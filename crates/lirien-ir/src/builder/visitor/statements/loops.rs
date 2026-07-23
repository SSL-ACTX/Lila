use super::*;

fn body_has_break_or_continue(stmts: &[ast::Stmt]) -> bool {
    for stmt in stmts {
        match stmt {
            ast::Stmt::Break(_) | ast::Stmt::Continue(_) => return true,
            ast::Stmt::If(if_stmt) => {
                if body_has_break_or_continue(&if_stmt.body)
                    || body_has_break_or_continue(&if_stmt.orelse)
                {
                    return true;
                }
            }
            ast::Stmt::For(for_stmt) if body_has_break_or_continue(&for_stmt.body) => return true,
            _ => {}
        }
    }
    false
}

impl CFGBuilder {
    pub fn visit_while(&mut self, s: ast::StmtWhile) -> BuilderResult<()> {
        let none_comp = self.get_none_comparison(&s.test);
        let header_block = self.create_block();
        let body_block = self.create_block();
        let exit_block = self.create_block();

        let prev_block = self.current_block;
        push_inst!(self, InstructionKind::Jump(header_block));
        self.link_blocks(prev_block, header_block);

        self.loop_stack.push((header_block, exit_block));

        // Extract loop invariants before visiting the loop body
        let predicates = extract_loop_invariants(self, &s.body, header_block)?;
        let location = self.current_location;
        for predicate in predicates {
            self.func.loop_invariants.push(LoopInvariant {
                header_block,
                predicate,
                location,
            });
        }

        self.start_block(header_block);
        let cond = self.visit_expr(*s.test)?;
        push_inst!(self, InstructionKind::Branch(cond, body_block, exit_block));
        self.link_blocks(header_block, body_block);
        self.link_blocks(header_block, exit_block);

        self.seal_block(body_block)?;
        self.start_block(body_block);
        push_inst!(self, InstructionKind::Nop()).add_constraint(format!("(= {} true)", cond));

        if let Some((ref var_name, true)) = none_comp {
            let old_val = self.read_variable(var_name.clone(), header_block)?;
            let ty = self.func.get_type(old_val);
            if let Type::NullablePointer(inner) = ty {
                let new_val = self.func.next_value();
                self.func.set_type(new_val, Type::Pointer(inner.clone()));
                push_inst!(self, InstructionKind::Assign(new_val, old_val));
                self.write_variable(var_name.clone(), body_block, new_val);
            } else if let Type::Optional(inner) = ty {
                let new_val = self.func.next_value();
                self.func.set_type(new_val, *inner.clone());
                let align = inner.align(&self.func.struct_layouts);
                let payload_offset = (1 + align - 1) & !(align - 1);
                push_inst!(
                    self,
                    InstructionKind::StructLoad(new_val, old_val, payload_offset)
                );
                self.write_variable(var_name.clone(), body_block, new_val);
            }
        }

        let mut body_iter = s.body.iter().peekable();
        while let Some(stmt) = body_iter.peek() {
            if let ast::Stmt::Assert(_) = stmt {
                body_iter.next();
            } else if is_invariant_call(stmt) {
                body_iter.next();
            } else {
                break;
            }
        }
        let remaining: Vec<ast::Stmt> = body_iter.cloned().collect();
        visit_block(self, &remaining)?;
        if !self.is_terminated(self.current_block) {
            push_inst!(self, InstructionKind::Jump(header_block));
            self.link_blocks(self.current_block, header_block);
        }

        self.loop_stack.pop();
        self.seal_block(header_block)?;
        self.start_block(exit_block);
        push_inst!(self, InstructionKind::Nop()).add_constraint(format!("(= {} false)", cond));

        if let Some((ref var_name, false)) = none_comp {
            let old_val = self.read_variable(var_name.clone(), header_block)?;
            let ty = self.func.get_type(old_val);
            if let Type::NullablePointer(inner) = ty {
                let new_val = self.func.next_value();
                self.func.set_type(new_val, Type::Pointer(inner.clone()));
                push_inst!(self, InstructionKind::Assign(new_val, old_val));
                self.write_variable(var_name.clone(), exit_block, new_val);
            } else if let Type::Optional(inner) = ty {
                let new_val = self.func.next_value();
                self.func.set_type(new_val, *inner.clone());
                let align = inner.align(&self.func.struct_layouts);
                let payload_offset = (1 + align - 1) & !(align - 1);
                push_inst!(
                    self,
                    InstructionKind::StructLoad(new_val, old_val, payload_offset)
                );
                self.write_variable(var_name.clone(), exit_block, new_val);
            }
        }

        self.seal_block(exit_block)?;
        Ok(())
    }

    pub fn visit_for(&mut self, s: ast::StmtFor) -> BuilderResult<()> {
        let iter_expr = *s.iter;
        let target = *s.target;

        let mut is_enumerate = false;
        let mut enum_buf_expr = None;

        let (start_val, end_val, step_val, is_direct_iter) =
            if let ast::Expr::Call(call) = iter_expr.clone() {
                if let ast::Expr::Name(n) = *call.func {
                    if n.id.as_str() == "range" {
                        let (start, end, step) = match call.args.len() {
                            1 => (None, self.visit_expr(call.args[0].clone())?, None),
                            2 => (
                                Some(self.visit_expr(call.args[0].clone())?),
                                self.visit_expr(call.args[1].clone())?,
                                None,
                            ),
                            3 => (
                                Some(self.visit_expr(call.args[0].clone())?),
                                self.visit_expr(call.args[1].clone())?,
                                Some(self.visit_expr(call.args[2].clone())?),
                            ),
                            _ => {
                                return Err(builder_error!(
                                    UnsupportedStatement,
                                    "Unsupported range() signature"
                                ))
                            }
                        };

                        let start_v = if let Some(v) = start {
                            v
                        } else {
                            let zero = self.func.next_value();
                            push_inst!(self, InstructionKind::ConstInt(zero, 0));
                            zero
                        };
                        let step_v = if let Some(v) = step {
                            v
                        } else {
                            let one = self.func.next_value();
                            push_inst!(self, InstructionKind::ConstInt(one, 1));
                            one
                        };
                        (start_v, end, step_v, false)
                    } else if n.id.as_str() == "enumerate" && call.args.len() == 1 {
                        is_enumerate = true;
                        enum_buf_expr = Some(call.args[0].clone());
                        let buf_val = self.visit_expr(call.args[0].clone())?;
                        let buf_ty = self.func.get_type(buf_val);
                        let zero = self.func.next_value();
                        push_inst!(self, InstructionKind::ConstInt(zero, 0));
                        let one = self.func.next_value();
                        push_inst!(self, InstructionKind::ConstInt(one, 1));
                        let len = self.func.next_value();
                        if let Type::Buffer(_) = buf_ty {
                            push_inst!(self, InstructionKind::BufferLen(len, buf_val));
                        } else if let Type::Array(_, Some(size)) = buf_ty {
                            push_inst!(self, InstructionKind::ConstInt(len, size as i64));
                        } else {
                            return Err(builder_error!(
                                General,
                                "Cannot iterate over unknown size array"
                            ));
                        }
                        self.func.set_type(len, Type::I64);
                        (zero, len, one, true)
                    } else {
                        return Err(builder_error!(
                            UnsupportedStatement,
                            "Unsupported function in for loop: {}",
                            n.id
                        ));
                    }
                } else {
                    return Err(builder_error!(
                        UnsupportedStatement,
                        "Only range() or direct iteration supported"
                    ));
                }
            } else {
                // Potential direct iteration: for x in buf
                let buf_val = self.visit_expr(iter_expr.clone())?;
                let buf_ty = self.func.get_type(buf_val);
                match buf_ty {
                    Type::Buffer(_) | Type::Array(_, _) => {
                        let zero = self.func.next_value();
                        push_inst!(self, InstructionKind::ConstInt(zero, 0));
                        let one = self.func.next_value();
                        push_inst!(self, InstructionKind::ConstInt(one, 1));
                        let len = self.func.next_value();
                        if let Type::Buffer(_) = buf_ty {
                            push_inst!(self, InstructionKind::BufferLen(len, buf_val));
                        } else if let Type::Array(_, Some(size)) = buf_ty {
                            push_inst!(self, InstructionKind::ConstInt(len, size as i64));
                        } else {
                            return Err(builder_error!(
                                General,
                                "Cannot iterate over unknown size array"
                            ));
                        }
                        self.func.set_type(len, Type::I64);
                        (zero, len, one, true)
                    }
                    _ => {
                        return Err(builder_error!(
                            General,
                            "Cannot iterate over type {:?}",
                            buf_ty
                        ))
                    }
                }
            };

        if !is_direct_iter && !body_has_break_or_continue(&s.body) {
            if let (Some(st), Some(sp), Some(step)) = (
                self.get_constant_int(start_val),
                self.get_constant_int(end_val),
                self.get_constant_int(step_val),
            ) {
                if step > 0 && sp >= st {
                    let count = (sp - st + step - 1) / step;
                    if count <= 128 {
                        let idx_name = if let ast::Expr::Name(n) = target.clone() {
                            n.id.to_string()
                        } else {
                            return Err(builder_error!(
                                UnsupportedStatement,
                                "Unsupported loop target"
                            ));
                        };

                        for iteration in 0..count {
                            let cur_val_i64 = st + iteration * step;
                            let cur_val = self.func.next_value();
                            push_inst!(self, InstructionKind::ConstInt(cur_val, cur_val_i64));
                            self.func.set_type(cur_val, Type::I64);
                            self.write_variable(idx_name.clone(), self.current_block, cur_val);
                            visit_block(self, &s.body)?;
                        }
                        return Ok(());
                    }
                }
            }
        }

        let header_block = self.create_block();
        let body_block = self.create_block();
        let increment_block = self.create_block();
        let exit_block = self.create_block();

        // Iterator variable (index)
        let idx_name = if is_direct_iter {
            format!("_lirien_idx_{}", self.func.value_count)
        } else if let ast::Expr::Name(n) = target.clone() {
            n.id.to_string()
        } else {
            return Err(builder_error!(
                UnsupportedStatement,
                "Unsupported loop target"
            ));
        };

        self.write_variable(idx_name.clone(), self.current_block, start_val);

        let prev_block = self.current_block;
        push_inst!(self, InstructionKind::Jump(header_block));
        self.link_blocks(prev_block, header_block);

        self.loop_stack.push((increment_block, exit_block));

        self.start_block(header_block);
        let curr_idx = self.read_variable(idx_name.clone(), header_block)?;

        // Extract loop invariants after starting header_block and reading variable
        let predicates = extract_loop_invariants(self, &s.body, header_block)?;
        let location = self.current_location;
        for predicate in predicates {
            self.func.loop_invariants.push(LoopInvariant {
                header_block,
                predicate,
                location,
            });
        }
        let cond = self.func.next_value();

        // Determine if we should use SLt or SGt based on step if constant
        let mut use_sgt = false;
        if let Some(val) = self.get_constant_int(step_val) {
            if val < 0 {
                use_sgt = true;
            }
        }

        if use_sgt {
            push_inst!(self, InstructionKind::SGt(cond, curr_idx, end_val));
        } else {
            push_inst!(self, InstructionKind::SLt(cond, curr_idx, end_val));
        }
        push_inst!(self, InstructionKind::Branch(cond, body_block, exit_block));
        self.link_blocks(header_block, body_block);
        self.link_blocks(header_block, exit_block);

        self.seal_block(body_block)?;
        self.start_block(body_block);

        if is_direct_iter {
            // For direct iter, load the value into the target variable
            let buf_expr = if is_enumerate {
                enum_buf_expr
                    .ok_or_else(|| builder_error!(General, "Missing enum buffer expression"))?
            } else {
                iter_expr.clone()
            };
            let buf_val = self.visit_expr(buf_expr)?;
            let buf_ty = self.func.get_type(buf_val);
            let element = self.func.next_value();
            match buf_ty {
                Type::Buffer(inner) => {
                    push_inst!(
                        self,
                        InstructionKind::BufferLoad(element, buf_val, curr_idx,)
                    );
                    self.func.set_type(element, *inner);
                }
                Type::Array(inner, _) => {
                    push_inst!(
                        self,
                        InstructionKind::ArrayLoad(element, buf_val, curr_idx,)
                    );
                    self.func.set_type(element, *inner);
                }
                _ => unreachable!(),
            }

            if is_enumerate {
                if let ast::Expr::Tuple(t) = target {
                    if t.elts.len() != 2 {
                        return Err(builder_error!(
                            General,
                            "enumerate() requires a tuple of 2 elements"
                        ));
                    }
                    self.handle_assignment_target(&t.elts[0], curr_idx)?;
                    self.handle_assignment_target(&t.elts[1], element)?;
                } else {
                    return Err(builder_error!(
                        UnsupportedStatement,
                        "enumerate() requires a tuple target"
                    ));
                }
            } else if let ast::Expr::Name(n) = target {
                self.write_variable(n.id.to_string(), self.current_block, element);
            } else {
                return Err(builder_error!(
                    UnsupportedStatement,
                    "Unsupported loop target"
                ));
            }
        }

        let mut body_iter = s.body.iter().peekable();
        while let Some(stmt) = body_iter.peek() {
            if let ast::Stmt::Assert(_) = stmt {
                body_iter.next();
            } else if is_invariant_call(stmt) {
                body_iter.next();
            } else {
                break;
            }
        }
        let remaining: Vec<ast::Stmt> = body_iter.cloned().collect();
        visit_block(self, &remaining)?;

        if !self.is_terminated(self.current_block) {
            push_inst!(self, InstructionKind::Jump(increment_block));
            self.link_blocks(self.current_block, increment_block);
        }

        self.seal_block(increment_block)?;
        self.start_block(increment_block);
        let next_idx = self.func.next_value();
        let updated_idx = self.read_variable(idx_name.clone(), increment_block)?;
        push_inst!(self, InstructionKind::Add(next_idx, updated_idx, step_val));
        self.write_variable(idx_name, increment_block, next_idx);
        push_inst!(self, InstructionKind::Jump(header_block));
        self.link_blocks(increment_block, header_block);

        self.loop_stack.pop();
        self.seal_block(header_block)?;
        self.start_block(exit_block);
        self.seal_block(exit_block)?;
        Ok(())
    }

    pub fn visit_break(&mut self, _s: ast::StmtBreak) -> BuilderResult<()> {
        if let Some((_, exit_block)) = self.loop_stack.last() {
            let eb = *exit_block;
            push_inst!(self, InstructionKind::Jump(eb));
            self.link_blocks(self.current_block, eb);
            Ok(())
        } else {
            Err(builder_error!(General, "break outside of loop"))
        }
    }

    pub fn visit_continue(&mut self, _s: ast::StmtContinue) -> BuilderResult<()> {
        if let Some((header_block, _)) = self.loop_stack.last() {
            let hb = *header_block;
            push_inst!(self, InstructionKind::Jump(hb));
            self.link_blocks(self.current_block, hb);
            Ok(())
        } else {
            Err(builder_error!(General, "continue outside of loop"))
        }
    }
}
