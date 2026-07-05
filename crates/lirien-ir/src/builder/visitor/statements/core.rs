use super::*;

impl CFGBuilder {
    pub fn visit_assign(&mut self, s: ast::StmtAssign) -> BuilderResult<()> {
        if s.targets.len() != 1 {
            return Err(builder_error!(
                UnsupportedStatement,
                "Only single targets supported"
            ));
        }
        let target = &s.targets[0];
        let value = self.visit_expr(*s.value)?;

        self.handle_assignment_target(target, value)?;
        Ok(())
    }

    pub fn visit_aug_assign(&mut self, s: ast::StmtAugAssign) -> BuilderResult<()> {
        let target = *s.target;
        let value = self.visit_expr(*s.value)?;
        let lhs = self.visit_expr(target.clone())?;
        let dest = self.func.next_value();
        let kind = self.build_binop(s.op, lhs, value, dest)?;
        push_inst!(self, kind);
        self.handle_assignment_target(&target, dest)?;
        Ok(())
    }

    pub fn visit_ann_assign(&mut self, s: ast::StmtAnnAssign) -> BuilderResult<()> {
        if let Some(value_expr) = s.value {
            let mut value = self.visit_expr(*value_expr)?;
            value = self.auto_load(value);

            if let Ok(ann_ty) = parse_type(
                &s.annotation,
                &self.type_aliases,
                &self.named_tuple_names,
                &self.typed_dict_names,
                &self.enum_names,
            ) {
                let val_ty = self.func.get_type(value);
                if ann_ty.is_float() && val_ty.is_int() {
                    let converted = self.func.next_value();
                    push_inst!(
                        self,
                        InstructionKind::IToF(converted, value, ann_ty.clone(),)
                    );
                    self.func.set_type(converted, ann_ty);
                    value = converted;
                }
            }

            self.handle_assignment_target(&s.target, value)?;
        }
        Ok(())
    }

    pub fn visit_assert(&mut self, s: ast::StmtAssert) -> BuilderResult<()> {
        let test_val = self.visit_expr(*s.test)?;
        let test_val = self.auto_load(test_val);
        let msg_str = if let Some(ref msg_expr) = s.msg {
            if let ast::Expr::Constant(ref c) = **msg_expr {
                if let ast::Constant::Str(ref string) = c.value {
                    Some(string.to_string())
                } else {
                    None
                }
            } else {
                None
            }
        } else {
            None
        };
        push_inst!(self, InstructionKind::Assert(test_val, msg_str));
        Ok(())
    }

    pub fn visit_return(&mut self, s: ast::StmtReturn) -> BuilderResult<()> {
        let mut val = if let Some(expr) = s.value {
            let v = self.visit_expr(*expr)?;
            Some(self.auto_load(v))
        } else {
            None
        };

        // Auto-cast to return type if necessary
        if let Some(v) = val {
            let val_ty = self.func.get_type(v);
            let ret_ty = self.func.return_type.clone();
            if ret_ty.is_float() && val_ty.is_int() {
                let converted = self.func.next_value();
                push_inst!(self, InstructionKind::IToF(converted, v, ret_ty.clone()));
                self.func.set_type(converted, ret_ty);
                val = Some(converted);
            } else if ret_ty.is_int() && val_ty.is_float() {
                let converted = self.func.next_value();
                push_inst!(self, InstructionKind::FToI(converted, v, ret_ty.clone()));
                self.func.set_type(converted, ret_ty);
                val = Some(converted);
            }
        }

        push_inst!(self, InstructionKind::Return(val));
        Ok(())
    }

    pub fn visit_with(&mut self, s: ast::StmtWith) -> BuilderResult<()> {
        for item in s.items {
            let val = self.visit_expr(item.context_expr)?;
            if let Some(vars) = item.optional_vars {
                self.handle_assignment_target(&vars, val)?;
            }
        }

        for stmt in s.body {
            self.visit_stmt(stmt)?;
        }

        Ok(())
    }

    pub fn visit_expr_stmt(&mut self, s: ast::StmtExpr) -> BuilderResult<()> {
        self.visit_expr(*s.value)?;
        Ok(())
    }
}
