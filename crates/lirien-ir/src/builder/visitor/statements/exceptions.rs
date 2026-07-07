use super::*;

impl CFGBuilder {
    pub(crate) fn visit_raise(&mut self, s: ast::StmtRaise) -> BuilderResult<()> {
        let exc_name = match s.exc {
            Some(exc_expr) => match *exc_expr {
                ast::Expr::Call(c) => match *c.func {
                    ast::Expr::Name(n) => n.id.to_string(),
                    _ => {
                        return Err(builder_error!(
                            UnsupportedStatement,
                            "Unsupported raise expression"
                        ))
                    }
                },
                ast::Expr::Name(n) => n.id.to_string(),
                _ => {
                    return Err(builder_error!(
                        UnsupportedStatement,
                        "Unsupported raise expression"
                    ))
                }
            },
            None => "".to_string(),
        };

        if !exc_name.is_empty() {
            let exc_id = match exc_name.as_str() {
                "ValueError" => 1,
                "TypeError" => 2,
                "IndexError" => 3,
                "RuntimeError" => 4,
                "ZeroDivisionError" => 5,
                _ => {
                    return Err(builder_error!(
                        UnsupportedStatement,
                        "Unsupported exception type: {}",
                        exc_name
                    ))
                }
            };

            let exc_ptr = Value(0);
            let val_to_write = self.func.next_value();
            push_inst!(self, InstructionKind::ConstInt(val_to_write, exc_id));
            self.func.set_type(val_to_write, Type::I64);
            push_inst!(self, InstructionKind::PointerStore(exc_ptr, val_to_write));
        }

        if let Some(&handler) = self.try_stack.last() {
            push_inst!(self, InstructionKind::Jump(handler));
            self.link_blocks(self.current_block, handler);
        } else {
            let ret_val = if self.func.return_type != Type::Unknown
                && self.func.return_type != Type::Tuple(vec![])
            {
                Some(self.dummy_value(&self.func.return_type.clone())?)
            } else {
                None
            };
            push_inst!(self, InstructionKind::Return(ret_val));
        }

        Ok(())
    }

    pub(crate) fn visit_try(&mut self, s: ast::StmtTry) -> BuilderResult<()> {
        if !s.finalbody.is_empty() {
            return Err(builder_error!(
                UnsupportedStatement,
                "finally blocks in try statements are not yet supported"
            ));
        }

        let handler_dispatch_block = self.create_block();

        self.try_stack.push(handler_dispatch_block);

        visit_block(self, &s.body)?;

        self.try_stack.pop();

        let try_exit_block = self.current_block;
        let merge_block = self.create_block();

        if !self.is_terminated(try_exit_block) {
            if !s.orelse.is_empty() {
                visit_block(self, &s.orelse)?;
            }
            if !self.is_terminated(self.current_block) {
                push_inst!(self, InstructionKind::Jump(merge_block));
                self.link_blocks(self.current_block, merge_block);
            }
        }

        // --- Compile the Handler Dispatch Block ---
        self.seal_block(handler_dispatch_block)?;
        self.start_block(handler_dispatch_block);

        let exc_ptr = Value(0);
        let exc_val = self.func.next_value();
        push_inst!(self, InstructionKind::PointerLoad(exc_val, exc_ptr));
        self.func.set_type(exc_val, Type::I64);

        let mut current_dispatch_block = handler_dispatch_block;
        let current_exc_val = exc_val;

        for handler in s.handlers {
            let ast::ExceptHandler::ExceptHandler(h) = handler;

            let caught_ids = if let Some(type_expr) = &h.type_ {
                let mut ids = Vec::new();
                match &**type_expr {
                    ast::Expr::Tuple(t) => {
                        for elt in &t.elts {
                            if let ast::Expr::Name(n) = elt {
                                if let Some(id) = get_exception_id(n.id.as_str()) {
                                    ids.push(id);
                                }
                            }
                        }
                    }
                    ast::Expr::Name(n) => {
                        if let Some(id) = get_exception_id(n.id.as_str()) {
                            ids.push(id);
                        }
                    }
                    _ => {
                        return Err(builder_error!(
                            UnsupportedStatement,
                            "Unsupported except handler type"
                        ))
                    }
                }
                ids
            } else {
                vec![1, 2, 3, 4, 5]
            };

            let match_block = self.create_block();
            let next_handler_block = self.create_block();

            let is_match = if caught_ids.len() == 5 {
                let true_val = self.func.next_value();
                let zero = self.func.next_value();
                push_inst!(self, InstructionKind::ConstInt(zero, 0));
                self.func.set_type(zero, Type::I64);
                push_inst!(self, InstructionKind::Ne(true_val, current_exc_val, zero));
                self.func.set_type(true_val, Type::Bool);
                true_val
            } else {
                let mut last_cond = None;
                for &id in &caught_ids {
                    let id_val = self.func.next_value();
                    push_inst!(self, InstructionKind::ConstInt(id_val, id as i64));
                    self.func.set_type(id_val, Type::I64);

                    let eq_cond = self.func.next_value();
                    push_inst!(self, InstructionKind::Eq(eq_cond, current_exc_val, id_val));
                    self.func.set_type(eq_cond, Type::Bool);

                    if let Some(prev_cond) = last_cond {
                        let or_cond = self.func.next_value();
                        push_inst!(self, InstructionKind::Or(or_cond, prev_cond, eq_cond));
                        self.func.set_type(or_cond, Type::Bool);
                        last_cond = Some(or_cond);
                    } else {
                        last_cond = Some(eq_cond);
                    }
                }
                last_cond.unwrap()
            };

            push_inst!(
                self,
                InstructionKind::Branch(is_match, match_block, next_handler_block)
            );
            self.link_blocks(current_dispatch_block, match_block);
            self.link_blocks(current_dispatch_block, next_handler_block);

            // --- Match Block ---
            self.seal_block(match_block)?;
            self.start_block(match_block);

            let zero = self.func.next_value();
            push_inst!(self, InstructionKind::ConstInt(zero, 0));
            self.func.set_type(zero, Type::I64);
            push_inst!(self, InstructionKind::PointerStore(exc_ptr, zero));

            if let Some(name) = &h.name {
                self.write_variable(name.to_string(), match_block, current_exc_val);
            }

            visit_block(self, &h.body)?;
            if !self.is_terminated(self.current_block) {
                push_inst!(self, InstructionKind::Jump(merge_block));
                self.link_blocks(self.current_block, merge_block);
            }

            self.seal_block(next_handler_block)?;
            self.start_block(next_handler_block);
            current_dispatch_block = next_handler_block;
        }

        if !self.is_terminated(current_dispatch_block) {
            if let Some(&outer_handler) = self.try_stack.last() {
                push_inst!(self, InstructionKind::Jump(outer_handler));
                self.link_blocks(current_dispatch_block, outer_handler);
            } else {
                let ret_val = if self.func.return_type != Type::Unknown
                    && self.func.return_type != Type::Tuple(vec![])
                {
                    Some(self.dummy_value(&self.func.return_type.clone())?)
                } else {
                    None
                };
                push_inst!(self, InstructionKind::Return(ret_val));
            }
        }

        // --- Merge Block ---
        self.seal_block(merge_block)?;
        self.start_block(merge_block);

        Ok(())
    }
}
