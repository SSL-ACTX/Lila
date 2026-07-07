pub mod conditionals;
pub mod core;
pub mod exceptions;
pub mod functions;
pub mod loops;

use crate::builder::error::BuilderResult;
use crate::builder::metadata::{expr_to_string, extract_refinement, parse_type};
use crate::builder::CFGBuilder;
use crate::ir::{BlockId, InstructionKind, LoopInvariant, Type, Value};
use crate::{builder_error, push_inst};
use rustpython_ast as ast;
use rustpython_ast::Ranged;
use std::collections::{HashMap, HashSet};

impl CFGBuilder {
    pub fn visit_stmt(&mut self, stmt: ast::Stmt) -> BuilderResult<()> {
        if is_invariant_call(&stmt) {
            return Ok(());
        }
        self.update_location(stmt.range().start().to_usize());
        match stmt {
            ast::Stmt::Assign(s) => self.visit_assign(s),
            ast::Stmt::AugAssign(s) => self.visit_aug_assign(s),
            ast::Stmt::AnnAssign(s) => self.visit_ann_assign(s),
            ast::Stmt::Assert(s) => self.visit_assert(s),
            ast::Stmt::Return(s) => self.visit_return(s),
            ast::Stmt::Try(s) => self.visit_try(s),
            ast::Stmt::Raise(s) => self.visit_raise(s),
            ast::Stmt::If(s) => self.visit_if(s),
            ast::Stmt::While(s) => self.visit_while(s),
            ast::Stmt::For(s) => self.visit_for(s),
            ast::Stmt::Match(s) => self.visit_match(s),
            ast::Stmt::With(s) => self.visit_with(s),
            ast::Stmt::Break(s) => self.visit_break(s),
            ast::Stmt::Continue(s) => self.visit_continue(s),
            ast::Stmt::Expr(s) => self.visit_expr_stmt(s),
            ast::Stmt::FunctionDef(s) => self.visit_nested_function_def(s),
            _ => Err(builder_error!(
                UnsupportedStatement,
                "Statement type {:?} not yet supported",
                stmt
            )),
        }
    }

    fn get_none_comparison(&self, test: &ast::Expr) -> Option<(String, bool)> {
        if let ast::Expr::Compare(ref s) = test {
            if s.ops.len() == 1 && s.comparators.len() == 1 {
                let op = s.ops[0];
                let is_none_comp = if let ast::Expr::Constant(ref c) = s.comparators[0] {
                    matches!(c.value, ast::Constant::None)
                } else {
                    false
                };

                if is_none_comp {
                    if let ast::Expr::Name(ref n) = *s.left {
                        let var_name = n.id.to_string();
                        match op {
                            ast::CmpOp::Is | ast::CmpOp::Eq => return Some((var_name, false)),
                            ast::CmpOp::IsNot | ast::CmpOp::NotEq => return Some((var_name, true)),
                            _ => {}
                        }
                    }
                }
            }
        }
        None
    }
}

pub(crate) fn get_exception_id(name: &str) -> Option<i32> {
    match name {
        "ValueError" => Some(1),
        "TypeError" => Some(2),
        "IndexError" => Some(3),
        "RuntimeError" => Some(4),
        "ZeroDivisionError" => Some(5),
        _ => None,
    }
}

pub(crate) fn visit_block(builder: &mut CFGBuilder, block: &[ast::Stmt]) -> BuilderResult<()> {
    for i in 0..block.len() {
        let stmt = &block[i];
        if i + 1 < block.len() {
            if let ast::Stmt::Assert(_) = stmt {
                if let ast::Stmt::Return(_) = &block[i + 1] {
                    continue;
                }
            }
        }
        builder.visit_stmt(stmt.clone())?;
    }
    Ok(())
}

fn is_requires_decorator(func: &ast::Expr) -> bool {
    match func {
        ast::Expr::Name(n) => n.id.as_str() == "requires",
        ast::Expr::Attribute(a) => {
            if let ast::Expr::Name(v) = &*a.value {
                v.id.as_str() == "verify" && a.attr.as_str() == "requires"
            } else {
                false
            }
        }
        _ => false,
    }
}

fn is_ensures_decorator(func: &ast::Expr) -> bool {
    match func {
        ast::Expr::Name(n) => n.id.as_str() == "ensures",
        ast::Expr::Attribute(a) => {
            if let ast::Expr::Name(v) = &*a.value {
                v.id.as_str() == "verify" && a.attr.as_str() == "ensures"
            } else {
                false
            }
        }
        _ => false,
    }
}

fn is_invariant_call(stmt: &ast::Stmt) -> bool {
    if let ast::Stmt::Expr(s) = stmt {
        if let ast::Expr::Call(c) = &*s.value {
            match &*c.func {
                ast::Expr::Name(n) => return n.id.as_str() == "invariant",
                ast::Expr::Attribute(a) => {
                    if let ast::Expr::Name(v) = &*a.value {
                        return v.id.as_str() == "verify" && a.attr.as_str() == "invariant";
                    }
                }
                _ => {}
            }
        }
    }
    false
}

fn collect_referenced_names(expr: &ast::Expr, names: &mut HashSet<String>) {
    match expr {
        ast::Expr::Name(n) => {
            names.insert(n.id.to_string());
        }
        ast::Expr::Lambda(l) => {
            collect_referenced_names(&l.body, names);
            for arg in &l.args.args {
                names.remove(arg.def.arg.as_str());
            }
        }
        ast::Expr::BoolOp(b) => {
            for val in &b.values {
                collect_referenced_names(val, names);
            }
        }
        ast::Expr::Compare(c) => {
            collect_referenced_names(&c.left, names);
            for comp in &c.comparators {
                collect_referenced_names(comp, names);
            }
        }
        ast::Expr::BinOp(b) => {
            collect_referenced_names(&b.left, names);
            collect_referenced_names(&b.right, names);
        }
        ast::Expr::UnaryOp(u) => {
            collect_referenced_names(&u.operand, names);
        }
        ast::Expr::IfExp(i) => {
            collect_referenced_names(&i.test, names);
            collect_referenced_names(&i.body, names);
            collect_referenced_names(&i.orelse, names);
        }
        ast::Expr::Call(c) => {
            collect_referenced_names(&c.func, names);
            for arg in &c.args {
                collect_referenced_names(arg, names);
            }
        }
        ast::Expr::Attribute(a) => {
            collect_referenced_names(&a.value, names);
        }
        ast::Expr::Subscript(s) => {
            collect_referenced_names(&s.value, names);
            collect_referenced_names(&s.slice, names);
        }
        ast::Expr::Tuple(t) => {
            for elt in &t.elts {
                collect_referenced_names(elt, names);
            }
        }
        ast::Expr::List(l) => {
            for elt in &l.elts {
                collect_referenced_names(elt, names);
            }
        }
        _ => {}
    }
}

fn rename_parameters(
    expr: &mut ast::Expr,
    param_map: &HashMap<String, Value>,
    return_var_name: Option<&str>,
) {
    match expr {
        ast::Expr::Name(n) => {
            if let Some(ret_name) = return_var_name {
                if n.id.as_str() == ret_name {
                    n.id = "{v}".to_string().into();
                    return;
                }
            }
            if let Some(val) = param_map.get(n.id.as_str()) {
                n.id = format!("v{}", val.0).into();
            }
        }
        ast::Expr::Lambda(l) => {
            let sub_ret = if return_var_name.is_none() && !l.args.args.is_empty() {
                None
            } else {
                return_var_name
            };
            rename_parameters(&mut l.body, param_map, sub_ret);
        }
        ast::Expr::BoolOp(b) => {
            for val in &mut b.values {
                rename_parameters(val, param_map, return_var_name);
            }
        }
        ast::Expr::Compare(c) => {
            rename_parameters(&mut c.left, param_map, return_var_name);
            for comp in &mut c.comparators {
                rename_parameters(comp, param_map, return_var_name);
            }
        }
        ast::Expr::BinOp(b) => {
            rename_parameters(&mut b.left, param_map, return_var_name);
            rename_parameters(&mut b.right, param_map, return_var_name);
        }
        ast::Expr::UnaryOp(u) => {
            rename_parameters(&mut u.operand, param_map, return_var_name);
        }
        ast::Expr::IfExp(i) => {
            rename_parameters(&mut i.test, param_map, return_var_name);
            rename_parameters(&mut i.body, param_map, return_var_name);
            rename_parameters(&mut i.orelse, param_map, return_var_name);
        }
        ast::Expr::Call(c) => {
            rename_parameters(&mut c.func, param_map, return_var_name);
            for arg in &mut c.args {
                rename_parameters(arg, param_map, return_var_name);
            }
        }
        ast::Expr::Attribute(a) => {
            rename_parameters(&mut a.value, param_map, return_var_name);
        }
        ast::Expr::Subscript(s) => {
            rename_parameters(&mut s.value, param_map, return_var_name);
            rename_parameters(&mut s.slice, param_map, return_var_name);
        }
        ast::Expr::Tuple(t) => {
            for elt in &mut t.elts {
                rename_parameters(elt, param_map, return_var_name);
            }
        }
        ast::Expr::List(l) => {
            for elt in &mut l.elts {
                rename_parameters(elt, param_map, return_var_name);
            }
        }
        _ => {}
    }
}

fn extract_loop_invariants(
    builder: &mut CFGBuilder,
    body: &[ast::Stmt],
    header_block: BlockId,
) -> BuilderResult<Vec<String>> {
    let mut invariants = Vec::new();
    let mut param_map = HashMap::new();

    for stmt in body {
        if let ast::Stmt::Assert(a) = stmt {
            let lambda_expr = &a.test;
            let mut ref_names = HashSet::new();
            collect_referenced_names(lambda_expr, &mut ref_names);

            for name in &ref_names {
                if builder.variable_defs.contains_key(name) {
                    let val = builder.read_variable(name.clone(), header_block)?;
                    param_map.insert(name.clone(), val);
                }
            }

            let mut renamed_expr = *lambda_expr.clone();
            rename_parameters(&mut renamed_expr, &param_map, None);

            if let Ok(pred_str) = expr_to_string(
                &renamed_expr,
                None,
                &Type::Unknown,
                &builder.func.struct_layouts,
            ) {
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
                invariants.push(pred_to_push);
            }
        } else if is_invariant_call(stmt) {
            if let ast::Stmt::Expr(s) = stmt {
                if let ast::Expr::Call(c) = &*s.value {
                    if !c.args.is_empty() {
                        let lambda_expr = &c.args[0];
                        let mut ref_names = HashSet::new();
                        collect_referenced_names(lambda_expr, &mut ref_names);

                        for name in &ref_names {
                            if builder.variable_defs.contains_key(name) {
                                let val = builder.read_variable(name.clone(), header_block)?;
                                param_map.insert(name.clone(), val);
                            }
                        }

                        let mut renamed_expr = lambda_expr.clone();
                        rename_parameters(&mut renamed_expr, &param_map, None);

                        if let Ok(pred_str) = expr_to_string(
                            &renamed_expr,
                            None,
                            &Type::Unknown,
                            &builder.func.struct_layouts,
                        ) {
                            invariants.push(pred_str);
                        }
                    }
                }
            }
        } else {
            break;
        }
    }
    Ok(invariants)
}

fn extract_postconditions(
    body: &[ast::Stmt],
    param_map: &HashMap<String, Value>,
    struct_layouts: &HashMap<String, Vec<(String, Type)>>,
) -> Vec<String> {
    let mut postconditions = Vec::new();
    for i in 0..body.len() {
        if i + 1 < body.len() {
            if let ast::Stmt::Assert(a) = &body[i] {
                if let ast::Stmt::Return(r) = &body[i + 1] {
                    let mut renamed_expr = *a.test.clone();
                    let ret_var_name = if let Some(ret_val) = &r.value {
                        if let ast::Expr::Name(n) = &**ret_val {
                            Some(n.id.to_string())
                        } else {
                            None
                        }
                    } else {
                        None
                    };
                    rename_parameters(&mut renamed_expr, param_map, ret_var_name.as_deref());
                    if let Ok(pred_str) =
                        expr_to_string(&renamed_expr, None, &Type::Unknown, struct_layouts)
                    {
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
                        postconditions.push(pred_to_push);
                    }
                }
            }
        }
        match &body[i] {
            ast::Stmt::If(s) => {
                postconditions.extend(extract_postconditions(&s.body, param_map, struct_layouts));
                postconditions.extend(extract_postconditions(&s.orelse, param_map, struct_layouts));
            }
            ast::Stmt::While(s) => {
                postconditions.extend(extract_postconditions(&s.body, param_map, struct_layouts));
            }
            ast::Stmt::For(s) => {
                postconditions.extend(extract_postconditions(&s.body, param_map, struct_layouts));
            }
            _ => {}
        }
    }
    postconditions
}
