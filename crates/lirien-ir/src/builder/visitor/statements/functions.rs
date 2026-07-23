use super::*;

impl CFGBuilder {
    pub fn visit_function_def(&mut self, s: ast::StmtFunctionDef) -> BuilderResult<()> {
        self.update_location(s.range().start().to_usize());
        self.func.arg_count = s.args.args.len() + 1;
        self.func.arg_names = vec!["__exception_ptr".to_string()];
        for arg in &s.args.args {
            self.func.arg_names.push(arg.def.arg.to_string());
        }

        // Prepend implicit exception pointer parameter at index 0
        let exc_val = self.func.next_value();
        self.func
            .set_type(exc_val, Type::Pointer(Box::new(Type::I64)));

        if let Some(returns) = &s.returns {
            let ret_ty = parse_type(
                returns,
                &self.type_aliases,
                &self.named_tuple_names,
                &self.typed_dict_names,
                &self.enum_names,
            )?;
            self.func.return_type = ret_ty;
            self.func.ret_refinement = extract_refinement(
                returns,
                &self.type_aliases,
                &self.func.struct_layouts,
                &self.named_tuple_names,
                &self.typed_dict_names,
                &self.enum_names,
            )?;
        }

        let mut param_map = HashMap::new();
        for arg in s.args.args {
            let val = self.func.next_value();
            if let Some(annotation) = &arg.def.annotation {
                let ty = parse_type(
                    annotation,
                    &self.type_aliases,
                    &self.named_tuple_names,
                    &self.typed_dict_names,
                    &self.enum_names,
                )?;
                self.func.set_type(val, ty);
                if let Some(refinement) = extract_refinement(
                    annotation,
                    &self.type_aliases,
                    &self.func.struct_layouts,
                    &self.named_tuple_names,
                    &self.typed_dict_names,
                    &self.enum_names,
                )? {
                    self.func.set_refinement(val, refinement);
                }
            }
            self.write_variable(arg.def.arg.to_string(), self.current_block, val);
            param_map.insert(arg.def.arg.to_string(), val);
        }

        // Process function decorators for preconditions and postconditions
        for dec in &s.decorator_list {
            if let ast::Expr::Call(c) = dec {
                if is_requires_decorator(&c.func) && !c.args.is_empty() {
                    let lambda_expr = &c.args[0];
                    let mut renamed_expr = lambda_expr.clone();
                    rename_parameters(&mut renamed_expr, &param_map, None);
                    if let Ok(pred_str) = expr_to_string(
                        &renamed_expr,
                        None,
                        &Type::Unknown,
                        &self.func.struct_layouts,
                    ) {
                        self.func.preconditions.push(pred_str);
                    }
                } else if is_ensures_decorator(&c.func) && !c.args.is_empty() {
                    let lambda_expr = &c.args[0];
                    let mut renamed_expr = lambda_expr.clone();
                    let ret_var_name = if let ast::Expr::Lambda(l) = lambda_expr {
                        if !l.args.args.is_empty() {
                            Some(l.args.args[0].def.arg.to_string())
                        } else {
                            None
                        }
                    } else {
                        None
                    };
                    rename_parameters(&mut renamed_expr, &param_map, ret_var_name.as_deref());
                    if let Ok(pred_str) = expr_to_string(
                        &renamed_expr,
                        None,
                        &Type::Unknown,
                        &self.func.struct_layouts,
                    ) {
                        self.func.postconditions.push(pred_str);
                    }
                }
            }
        }

        // Extract postconditions from assert statements
        let asserts_post = extract_postconditions(&s.body, &param_map, &self.func.struct_layouts);
        self.func.postconditions.extend(asserts_post);

        println!("[DEBUG] s.body size for {}: {}", s.name.as_str(), s.body.len());
        for (idx, stmt) in s.body.iter().enumerate() {
            println!("[DEBUG]   stmt {}: {:?}", idx, stmt);
        }

        // Process function assert preconditions at the top
        let mut body_iter = s.body.iter().peekable();
        while let Some(stmt) = body_iter.peek() {
            if let ast::Stmt::Assert(a) = stmt {
                let mut renamed_expr = *a.test.clone();
                rename_parameters(&mut renamed_expr, &param_map, None);
                match expr_to_string(
                    &renamed_expr,
                    None,
                    &Type::Unknown,
                    &self.func.struct_layouts,
                ) {
                    Ok(pred_str) => {
                        let msg_str = if let Some(ref msg_expr) = a.msg {
                            if let ast::Expr::Constant(ref c) = **msg_expr {
                                if let ast::Constant::Str(ref s) = c.value {
                                    Some(s.to_string())
                                } else {
                                    None
                                }
                            } else {
                                None
                            }
                        } else {
                            None
                        };
                        let pred_to_push = if let Some(ref m) = msg_str {
                            format!("{} :::msg::: {}", pred_str, m)
                        } else {
                            pred_str
                        };
                        self.func.preconditions.push(pred_to_push);
                    }
                    Err(e) => {
                        eprintln!("[DEBUG] expr_to_string failed for precondition assert: {:?}", e);
                    }
                }
                body_iter.next();
            } else {
                break;
            }
        }

        // Visit the remaining statements
        let remaining_stmts: Vec<ast::Stmt> = body_iter.cloned().collect();
        visit_block(self, &remaining_stmts)?;

        // Ensure the last block has a return if it's not terminated
        if !self.is_terminated(self.current_block) {
            let ret_val = if self.func.return_type != Type::Unknown {
                let zero = self.func.next_value();
                match self.func.return_type {
                    Type::F32 | Type::F64 => {
                        push_inst!(self, InstructionKind::ConstFloat(zero, 0.0));
                    }
                    _ => {
                        push_inst!(self, InstructionKind::ConstInt(zero, 0));
                    }
                }
                Some(zero)
            } else {
                None
            };
            push_inst!(self, InstructionKind::Return(ret_val));
        }

        Ok(())
    }

    pub fn visit_nested_function_def(&mut self, s: ast::StmtFunctionDef) -> BuilderResult<()> {
        use crate::builder::capture_analysis::CaptureVisitor;
        use rustpython_ast::Visitor;

        let next_val = self.func.next_value().0;
        let func_name = format!("{}_{}_{}", self.func.name, s.name, next_val);

        // 1. Capture Analysis
        let mut params = Vec::new();
        for arg in &s.args.args {
            params.push(arg.def.arg.to_string());
        }
        let mut capture_visitor = CaptureVisitor::new(params);
        for stmt in &s.body {
            capture_visitor.visit_stmt(stmt.clone());
        }

        let mut captures = Vec::new();
        let mut capture_types = Vec::new();
        for var_name in capture_visitor.captures {
            if self.variable_defs.contains_key(&var_name) {
                let val = self.read_variable(var_name.clone(), self.current_block)?;
                let ty = self.func.get_type(val);
                captures.push((var_name, val));
                capture_types.push(ty);
            }
        }

        // 2. Build Inner Function
        let mut inner_builder = self.new_sub_builder(func_name.clone());

        // Define arguments in inner function
        inner_builder.func.arg_count = 2 + s.args.args.len();
        inner_builder.func.value_count = inner_builder.func.arg_count;
        inner_builder
            .func
            .value_types
            .insert(Value(0), Type::Pointer(Box::new(Type::I64))); // _exc_ptr
        inner_builder
            .func
            .value_types
            .insert(Value(1), Type::Struct("ClosureEnv".to_string())); // ctx_ptr

        if let Some(returns) = &s.returns {
            inner_builder.func.return_type = parse_type(
                returns,
                &self.type_aliases,
                &self.named_tuple_names,
                &self.typed_dict_names,
                &self.enum_names,
            )?;
        }

        for (i, arg) in s.args.args.iter().enumerate() {
            let arg_ty = if let Some(annotation) = &arg.def.annotation {
                parse_type(
                    annotation,
                    &self.type_aliases,
                    &self.named_tuple_names,
                    &self.typed_dict_names,
                    &self.enum_names,
                )?
            } else {
                Type::Unknown
            };
            inner_builder.func.value_types.insert(Value(i + 2), arg_ty);
            inner_builder.write_variable(
                arg.def.arg.to_string(),
                inner_builder.current_block,
                Value(i + 2),
            );
        }

        // If there are captures, load them from ctx_ptr
        if !captures.is_empty() {
            let mut offset = 8; // Offset 0 is fn_ptr
            for (name, ty) in captures.iter().zip(capture_types.iter()) {
                let align = ty.align(&self.func.struct_layouts);
                offset = (offset + align - 1) & !(align - 1);

                let dest = inner_builder.func.next_value();
                push_inst!(
                    inner_builder,
                    InstructionKind::StructLoad(dest, Value(1), offset,)
                );
                inner_builder.func.set_type(dest, ty.clone());
                inner_builder.write_variable(name.0.clone(), inner_builder.current_block, dest);

                offset += ty.size(&self.func.struct_layouts);
            }
        }

        // Visit body
        for stmt in s.body {
            inner_builder.visit_stmt(stmt)?;
        }

        // Ensure the last block has a return if it's not terminated
        if !inner_builder.is_terminated(inner_builder.current_block) {
            let ret_val = if inner_builder.func.return_type != Type::Unknown {
                let zero = inner_builder.func.next_value();
                match inner_builder.func.return_type {
                    Type::F32 | Type::F64 => {
                        push_inst!(inner_builder, InstructionKind::ConstFloat(zero, 0.0));
                    }
                    _ => {
                        push_inst!(inner_builder, InstructionKind::ConstInt(zero, 0));
                    }
                }
                Some(zero)
            } else {
                None
            };
            push_inst!(inner_builder, InstructionKind::Return(ret_val));
        }

        let ret_ty = inner_builder.func.return_type.clone();
        if ret_ty != Type::Unknown && ret_ty != Type::Tuple(vec![]) {
            let block_ids: Vec<crate::ir::BlockId> = inner_builder
                .func
                .blocks
                .iter()
                .filter(|b| {
                    b.instructions
                        .last()
                        .is_some_and(|inst| matches!(inst.kind, InstructionKind::Return(None)))
                })
                .map(|b| b.id)
                .collect();

            for bid in block_ids {
                let prev_block = inner_builder.current_block;
                inner_builder.current_block = bid;

                if let Some(block) = inner_builder.func.blocks.iter_mut().find(|b| b.id == bid) {
                    block.instructions.pop();
                }

                let dummy_val = inner_builder.dummy_value(&ret_ty)?;
                push_inst!(inner_builder, InstructionKind::Return(Some(dummy_val)));

                inner_builder.current_block = prev_block;
            }
        }

        // Optimization for inner function
        crate::optimization::optimize(&mut inner_builder.func);

        // Store inner function for later compilation
        let inner_func = inner_builder.func;
        self.lambdas.push(inner_func.clone());
        // Collect nested lambdas from the sub-builder
        self.lambdas.extend(inner_builder.lambdas);

        // 3. Create Closure Instruction
        let dest = self.func.next_value();
        let capture_vals: Vec<Value> = captures.iter().map(|(_, v)| *v).collect();
        push_inst!(
            self,
            InstructionKind::Lambda(dest, func_name.clone(), capture_vals,)
        );

        let arg_types: Vec<Type> = (2..2 + s.args.args.len())
            .map(|i| inner_func.get_type(Value(i)))
            .collect();
        self.func.set_type(
            dest,
            Type::Closure(
                func_name.clone(),
                arg_types,
                Box::new(inner_func.return_type),
                Some(s.name.to_string()),
            ),
        );

        self.write_variable(s.name.to_string(), self.current_block, dest);

        Ok(())
    }

    pub(crate) fn handle_nested_pattern(
        &mut self,
        pattern: &ast::Pattern,
        val: Value,
        block: BlockId,
    ) -> BuilderResult<()> {
        let ty = self.func.get_type(val);
        if let Type::Pointer(inner) = ty {
            // Automatically dereference pointers for matching
            let deref_val = self.func.next_value();
            push_inst!(self, InstructionKind::PointerLoad(deref_val, val));
            self.func.set_type(deref_val, (*inner).clone());
            return self.handle_nested_pattern(pattern, deref_val, block);
        }

        match pattern {
            ast::Pattern::MatchAs(p) => {
                if p.pattern.is_some() {
                    return Err(builder_error!(
                        UnsupportedStatement,
                        "Nested patterns in MatchAs not yet supported"
                    ));
                }
                if let Some(name) = &p.name {
                    self.write_variable(name.to_string(), block, val);
                }
                Ok(())
            }
            ast::Pattern::MatchClass(p) => {
                // Nested struct or enum destructuring
                let ty = self.func.get_type(val);
                match ty {
                    Type::Struct(ref name) | Type::NamedTuple(ref name) => {
                        let fields =
                            self.func.struct_layouts.get(name).cloned().ok_or_else(|| {
                                builder_error!(General, "Unknown struct layout for '{}'", name)
                            })?;
                        if p.patterns.len() > fields.len() {
                            return Err(builder_error!(
                                General,
                                "Struct '{}' has {} fields, but pattern has {}",
                                name,
                                fields.len(),
                                p.patterns.len()
                            ));
                        }
                        let mut current_offset = 0;
                        for (i, sub_pattern) in p.patterns.iter().enumerate() {
                            let field_ty = &fields[i].1;
                            let align = field_ty.align(&self.func.struct_layouts);
                            current_offset = (current_offset + align - 1) & !(align - 1);

                            let field_val = self.func.next_value();
                            if field_ty.is_composite() {
                                push_inst!(
                                    self,
                                    InstructionKind::StructOffset(field_val, val, current_offset,)
                                );
                            } else {
                                push_inst!(
                                    self,
                                    InstructionKind::StructLoad(field_val, val, current_offset,)
                                );
                            }
                            self.func.set_type(field_val, field_ty.clone());
                            self.handle_nested_pattern(sub_pattern, field_val, block)?;

                            current_offset += field_ty.size(&self.func.struct_layouts);
                        }
                        Ok(())
                    }
                    Type::Enum(ref name) => {
                        let variant_name = match &*p.cls {
                            ast::Expr::Attribute(a) => a.attr.to_string(),
                            _ => {
                                return Err(builder_error!(
                                    UnsupportedStatement,
                                    "Expected Enum.Variant pattern"
                                ))
                            }
                        };
                        let variants = self
                            .func
                            .enum_layouts
                            .get(name)
                            .cloned() // Clone to avoid borrowing self.func
                            .ok_or_else(|| {
                                builder_error!(General, "Unknown enum layout for '{}'", name)
                            })?;
                        let tag_idx = variants
                            .iter()
                            .position(|(n, _)| *n == variant_name)
                            .ok_or_else(|| {
                                builder_error!(
                                    General,
                                    "Unknown variant '{}' for enum '{}'",
                                    variant_name,
                                    name
                                )
                            })?;

                        let payload = self.func.next_value();
                        push_inst!(self, InstructionKind::EnumExtract(payload, val, tag_idx));
                        let variant_ty = variants[tag_idx].1.clone();
                        self.func.set_type(payload, variant_ty.clone()); // Clone to avoid move

                        if p.patterns.len() == 1 {
                            self.handle_nested_pattern(&p.patterns[0], payload, block)?;
                        } else if !p.patterns.is_empty() {
                            if let Type::Tuple(ref types) = variant_ty {
                                if types.len() != p.patterns.len() {
                                    return Err(builder_error!(
                                        General,
                                        "Variant '{}' has {} fields, but pattern has {}",
                                        variant_name,
                                        types.len(),
                                        p.patterns.len()
                                    ));
                                }
                                for (i, sub_p) in p.patterns.iter().enumerate() {
                                    let elt = self.func.next_value();
                                    push_inst!(
                                        self,
                                        InstructionKind::TupleExtract(elt, payload, i,)
                                    );
                                    self.func.set_type(elt, types[i].clone());
                                    self.handle_nested_pattern(sub_p, elt, block)?;
                                }
                            } else {
                                return Err(builder_error!(General, "Variant '{}' has a non-tuple payload, but pattern has {} fields", variant_name, p.patterns.len()));
                            }
                        }
                        Ok(())
                    }
                    _ => Err(builder_error!(General, "Cannot destructure type {:?}", ty)),
                }
            }
            ast::Pattern::MatchSequence(p) => {
                let ty = self.func.get_type(val);
                if let Type::Tuple(ref types) = ty {
                    if p.patterns.len() != types.len() {
                        return Err(builder_error!(
                            General,
                            "Tuple has {} elements, but pattern has {}",
                            types.len(),
                            p.patterns.len()
                        ));
                    }
                    for (i, sub_pattern) in p.patterns.iter().enumerate() {
                        let elt_val = self.func.next_value();
                        push_inst!(self, InstructionKind::TupleExtract(elt_val, val, i));
                        self.func.set_type(elt_val, types[i].clone());
                        self.handle_nested_pattern(sub_pattern, elt_val, block)?;
                    }
                    Ok(())
                } else if let Type::NamedTuple(ref name) = ty {
                    let fields = self.func.struct_layouts.get(name).cloned().ok_or_else(|| {
                        builder_error!(General, "Unknown NamedTuple layout for '{}'", name)
                    })?;
                    if p.patterns.len() != fields.len() {
                        return Err(builder_error!(
                            General,
                            "NamedTuple has {} fields, but pattern has {}",
                            fields.len(),
                            p.patterns.len()
                        ));
                    }
                    for (i, sub_pattern) in p.patterns.iter().enumerate() {
                        let (field_name, field_ty) = &fields[i];
                        let field_offset =
                            self.get_field_offset(name, field_name).ok_or_else(|| {
                                builder_error!(General, "Field offset not found for {}", field_name)
                            })?;
                        let elt_val = self.func.next_value();
                        push_inst!(
                            self,
                            InstructionKind::StructLoad(elt_val, val, field_offset)
                        );
                        self.func.set_type(elt_val, field_ty.clone());
                        self.handle_nested_pattern(sub_pattern, elt_val, block)?;
                    }
                    Ok(())
                } else {
                    Err(builder_error!(
                        General,
                        "Sequence pattern expected Tuple or NamedTuple, found {:?}",
                        ty
                    ))
                }
            }
            ast::Pattern::MatchValue(_) => Err(builder_error!(
                UnsupportedStatement,
                "Literal matching not yet supported in nested patterns"
            )),
            _ => Err(builder_error!(
                UnsupportedStatement,
                "Unsupported nested pattern type: {:?}",
                pattern
            )),
        }
    }
}
