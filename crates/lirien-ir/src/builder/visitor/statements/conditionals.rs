use super::*;

impl CFGBuilder {
    pub fn visit_if(&mut self, s: ast::StmtIf) -> BuilderResult<()> {
        let none_comp = self.get_none_comparison(&s.test);
        let cond = self.visit_expr(*s.test)?;
        let cond = self.auto_load(cond);

        // Constant pruning for If
        if let Some(val) = self.get_constant_int(cond) {
            if val != 0 {
                visit_block(self, &s.body)?;
            } else {
                visit_block(self, &s.orelse)?;
            }
            return Ok(());
        }

        let prev_block = self.current_block;
        let true_block = self.create_block();
        let false_block = self.create_block();
        let merge_block = self.create_block();

        push_inst!(self, InstructionKind::Branch(cond, true_block, false_block));
        self.link_blocks(prev_block, true_block);
        self.link_blocks(prev_block, false_block);

        self.seal_block(true_block)?;
        self.seal_block(false_block)?;

        self.start_block(true_block);
        push_inst!(self, InstructionKind::Nop()).add_constraint(format!("(= {} true)", cond));

        if let Some((ref var_name, true)) = none_comp {
            let old_val = self.read_variable(var_name.clone(), prev_block)?;
            let ty = self.func.get_type(old_val);
            if let Type::NullablePointer(inner) = ty {
                let new_val = self.func.next_value();
                self.func.set_type(new_val, Type::Pointer(inner.clone()));
                push_inst!(self, InstructionKind::Assign(new_val, old_val));
                self.write_variable(var_name.clone(), true_block, new_val);
            } else if let Type::Optional(inner) = ty {
                let new_val = self.func.next_value();
                self.func.set_type(new_val, *inner.clone());
                let align = inner.align(&self.func.struct_layouts);
                let payload_offset = (1 + align - 1) & !(align - 1);
                push_inst!(
                    self,
                    InstructionKind::StructLoad(new_val, old_val, payload_offset)
                );
                self.write_variable(var_name.clone(), true_block, new_val);
            }
        }

        visit_block(self, &s.body)?;
        if !self.is_terminated(self.current_block) {
            push_inst!(self, InstructionKind::Jump(merge_block));
            self.link_blocks(self.current_block, merge_block);
        }

        self.start_block(false_block);
        push_inst!(self, InstructionKind::Nop()).add_constraint(format!("(= {} false)", cond));

        if let Some((ref var_name, false)) = none_comp {
            let old_val = self.read_variable(var_name.clone(), prev_block)?;
            let ty = self.func.get_type(old_val);
            if let Type::NullablePointer(inner) = ty {
                let new_val = self.func.next_value();
                self.func.set_type(new_val, Type::Pointer(inner.clone()));
                push_inst!(self, InstructionKind::Assign(new_val, old_val));
                self.write_variable(var_name.clone(), false_block, new_val);
            } else if let Type::Optional(inner) = ty {
                let new_val = self.func.next_value();
                self.func.set_type(new_val, *inner.clone());
                let align = inner.align(&self.func.struct_layouts);
                let payload_offset = (1 + align - 1) & !(align - 1);
                push_inst!(
                    self,
                    InstructionKind::StructLoad(new_val, old_val, payload_offset)
                );
                self.write_variable(var_name.clone(), false_block, new_val);
            }
        }

        visit_block(self, &s.orelse)?;
        if !self.is_terminated(self.current_block) {
            push_inst!(self, InstructionKind::Jump(merge_block));
            self.link_blocks(self.current_block, merge_block);
        }

        self.seal_block(merge_block)?;
        self.start_block(merge_block);
        Ok(())
    }

    pub fn visit_match(&mut self, s: ast::StmtMatch) -> BuilderResult<()> {
        let subject_val = self.visit_expr(*s.subject.clone())?;
        let subject_ty = self.func.get_type(subject_val);

        if !matches!(subject_ty, Type::Enum(_)) {
            return self.compile_non_enum_match(s, subject_val);
        }

        let enum_name = match subject_ty {
            Type::Enum(name) => name,
            _ => unreachable!(),
        };

        let exit_block = self.create_block();
        let start_block = self.current_block;

        let tag_val = self.func.next_value();
        push_inst!(self, InstructionKind::EnumGetTag(tag_val, subject_val));
        self.func.set_type(tag_val, Type::U8);

        let variants = self
            .func
            .enum_layouts
            .get(&enum_name)
            .cloned()
            .ok_or_else(|| builder_error!(General, "Unknown enum layout for '{}'", enum_name))?;

        // Group cases by tag
        let mut tag_to_cases: HashMap<usize, Vec<ast::MatchCase>> = HashMap::new();
        let mut global_default_case: Option<ast::MatchCase> = None;

        for case in s.cases {
            match case.pattern {
                ast::Pattern::MatchAs(ref p) if p.pattern.is_none() && case.guard.is_none() => {
                    // Catch-all pattern without guard
                    if global_default_case.is_none() {
                        global_default_case = Some(case);
                    }
                    // Subsequent catch-alls are unreachable
                    break;
                }
                _ => {
                    let variant_res = match case.pattern {
                        ast::Pattern::MatchClass(ref p) => {
                            let attr = match &*p.cls {
                                ast::Expr::Attribute(a) => a,
                                _ => {
                                    return Err(builder_error!(
                                        UnsupportedStatement,
                                        "Expected Enum.Variant pattern"
                                    ));
                                }
                            };
                            Some(attr.attr.to_string())
                        }
                        ast::Pattern::MatchValue(ref p) => {
                            let attr = match &*p.value {
                                ast::Expr::Attribute(a) => a,
                                _ => {
                                    return Err(builder_error!(
                                        UnsupportedStatement,
                                        "Expected Enum.Variant pattern"
                                    ));
                                }
                            };
                            Some(attr.attr.to_string())
                        }
                        _ => None, // Might be MatchAs with a guard
                    };

                    if let Some(variant_name) = variant_res {
                        let tag_idx = variants.iter().position(|(name, _)| *name == variant_name);
                        if let Some(tag_idx) = tag_idx {
                            tag_to_cases.entry(tag_idx).or_default().push(case);
                        } else {
                            return Err(builder_error!(
                                General,
                                "Unknown variant '{}' for enum '{}'",
                                variant_name,
                                enum_name
                            ));
                        }
                    } else {
                        // This is a non-variant pattern (e.g. MatchAs with a name/guard).
                        // Python's `match` is strictly sequential.
                        // Currently, Lirien uses a single jump table for all variant-based cases.
                        // Interleaving generic patterns (like `case x if cond:`) between variants
                        // is not yet supported because it would require multiple sequential jump tables.
                        return Err(builder_error!(UnsupportedStatement, "Lirien currently requires all Enum variant patterns to come before any guarded catch-all patterns."));
                    }
                }
            }
        }

        let mut cases_map = HashMap::new();
        let default_block = self.create_block();

        // 1. Handle explicit variant cases
        for (tag_idx, tag_cases) in tag_to_cases {
            let dispatch_block = self.create_block();
            cases_map.insert(tag_idx, dispatch_block);

            self.start_block(dispatch_block);
            self.link_blocks(start_block, dispatch_block);

            let mut current_chain_block = dispatch_block;

            for (i, case) in tag_cases.iter().enumerate() {
                let next_in_chain = if i < tag_cases.len() - 1 {
                    self.create_block()
                } else {
                    default_block
                };

                let body_block = self.create_block();

                // 1.1. Pattern destructuring (bindings)
                let pattern_args = match &case.pattern {
                    ast::Pattern::MatchClass(p) => &p.patterns,
                    ast::Pattern::MatchValue(_) => &Vec::new(),
                    _ => unreachable!(),
                };

                if !pattern_args.is_empty() {
                    let payload = self.func.next_value();
                    push_inst!(
                        self,
                        InstructionKind::EnumExtract(payload, subject_val, tag_idx,)
                    );
                    let variant_ty = variants[tag_idx].1.clone();
                    self.func.set_type(payload, variant_ty.clone());

                    if pattern_args.len() == 1 {
                        self.handle_nested_pattern(&pattern_args[0], payload, current_chain_block)?;
                    } else {
                        if let Type::Tuple(ref types) = variant_ty {
                            for (j, p_arg) in pattern_args.iter().enumerate() {
                                let elt = self.func.next_value();
                                push_inst!(self, InstructionKind::TupleExtract(elt, payload, j,));
                                self.func.set_type(elt, types[j].clone());
                                self.handle_nested_pattern(p_arg, elt, current_chain_block)?;
                            }
                        }
                    }
                }

                // 1.2. Guard check
                if let Some(guard_expr) = &case.guard {
                    let cond = self.visit_expr(*guard_expr.clone())?;
                    push_inst!(
                        self,
                        InstructionKind::Branch(cond, body_block, next_in_chain,)
                    );
                    self.link_blocks(current_chain_block, body_block);
                    self.link_blocks(current_chain_block, next_in_chain);
                } else {
                    push_inst!(self, InstructionKind::Jump(body_block));
                    self.link_blocks(current_chain_block, body_block);
                }
                self.seal_block(current_chain_block)?;

                // 1.3. Body
                self.start_block(body_block);
                self.seal_block(body_block)?;
                for stmt in &case.body {
                    self.visit_stmt(stmt.clone())?;
                }
                if !self.is_terminated(self.current_block) {
                    push_inst!(self, InstructionKind::Jump(exit_block));
                    self.link_blocks(self.current_block, exit_block);
                }
                self.seal_block(body_block)?;

                if i < tag_cases.len() - 1 {
                    self.start_block(next_in_chain);
                    current_chain_block = next_in_chain;
                }
            }
        }

        // 2. Handle global default (catch-all)
        self.start_block(default_block);
        self.link_blocks(start_block, default_block);
        if let Some(ref case) = global_default_case {
            if let ast::Pattern::MatchAs(ref p) = case.pattern {
                if let Some(name) = &p.name {
                    self.write_variable(name.to_string(), default_block, subject_val);
                }
            }
            for stmt in &case.body {
                self.visit_stmt(stmt.clone())?;
            }
            if !self.is_terminated(self.current_block) {
                push_inst!(self, InstructionKind::Jump(exit_block));
                self.link_blocks(self.current_block, exit_block);
            }
        } else {
            push_inst!(self, InstructionKind::Jump(exit_block));
            self.link_blocks(default_block, exit_block);
        }
        self.seal_block(default_block)?;

        // Finalize start block
        self.start_block(start_block);
        push_inst!(
            self,
            InstructionKind::Match(
                tag_val,
                cases_map,
                default_block,
                global_default_case.is_none(),
            )
        );

        self.start_block(exit_block);
        self.seal_block(exit_block)?;
        Ok(())
    }

    pub(crate) fn compile_non_enum_match(
        &mut self,
        s: ast::StmtMatch,
        subject_val: Value,
    ) -> BuilderResult<()> {
        let exit_block = self.create_block();
        let start_block = self.current_block;

        let mut current_chain_block = start_block;

        for (i, case) in s.cases.iter().enumerate() {
            let body_block = self.create_block();
            let next_in_chain = if i < s.cases.len() - 1 {
                self.create_block()
            } else {
                exit_block
            };

            self.start_block(current_chain_block);

            // 1. Destructure pattern if applicable
            self.handle_nested_pattern(&case.pattern, subject_val, current_chain_block)?;

            // 2. Guard check
            if let Some(guard_expr) = &case.guard {
                let cond = self.visit_expr(*guard_expr.clone())?;
                let cond = self.auto_load(cond);
                push_inst!(
                    self,
                    InstructionKind::Branch(cond, body_block, next_in_chain)
                );
                self.link_blocks(current_chain_block, body_block);
                self.link_blocks(current_chain_block, next_in_chain);
            } else {
                push_inst!(self, InstructionKind::Jump(body_block));
                self.link_blocks(current_chain_block, body_block);
            }
            self.seal_block(current_chain_block)?;

            // 3. Visit body
            self.start_block(body_block);
            self.seal_block(body_block)?;
            for stmt in &case.body {
                self.visit_stmt(stmt.clone())?;
            }
            if !self.is_terminated(self.current_block) {
                push_inst!(self, InstructionKind::Jump(exit_block));
                self.link_blocks(self.current_block, exit_block);
            }
            self.seal_block(body_block)?;

            if i < s.cases.len() - 1 {
                current_chain_block = next_in_chain;
            }
        }

        self.start_block(exit_block);
        self.seal_block(exit_block)?;
        Ok(())
    }
}
