use crate::refinement::resolver::Resolver;
use crate::refinement::utils::split_sexpr_parts;
use crate::refinement::utils::{float_eq, float_ge, float_gt, float_le, float_lt};
use z3::ast::Bool;

pub(crate) fn parse_bool_expr(
    sexpr: &str,
    v_int: Option<&z3::ast::Int>,
    v_real: Option<&z3::ast::Real>,
    v_float: Option<&z3::ast::Float>,
    v_arr: Option<&z3::ast::Array>,
    v_bv: Option<&z3::ast::BV>,
    resolver: Option<&Resolver>,
) -> Result<Bool, String> {
    let sexpr = sexpr.trim();
    if sexpr == "true" {
        return Ok(Bool::from_bool(true));
    }
    if sexpr == "false" {
        return Ok(Bool::from_bool(false));
    }

    if !sexpr.starts_with('(') {
        if let Some(r) = resolver {
            if let Some(b) = r.resolve_bool(sexpr) {
                return Ok(b);
            }
        }
        return Err(format!("Invalid boolean sexpr: {}", sexpr));
    }
    let inner = &sexpr[1..sexpr.len() - 1];
    let parts = split_sexpr_parts(inner);
    if parts.is_empty() {
        return Err("Empty sexpr".to_string());
    }

    match parts[0] {
        "and" | "&" => {
            let mut sub_exprs = Vec::new();
            for part in &parts[1..] {
                sub_exprs.push(parse_bool_expr(
                    part, v_int, v_real, v_float, v_arr, v_bv, resolver,
                )?);
            }
            let refs: Vec<&Bool> = sub_exprs.iter().collect();
            Ok(Bool::and(&refs))
        }
        "or" | "|" => {
            let mut sub_exprs = Vec::new();
            for part in &parts[1..] {
                sub_exprs.push(parse_bool_expr(
                    part, v_int, v_real, v_float, v_arr, v_bv, resolver,
                )?);
            }
            let refs: Vec<&Bool> = sub_exprs.iter().collect();
            Ok(Bool::or(&refs))
        }
        "xor" | "^" => {
            if parts.len() != 3 {
                return Err("xor expects 2 arguments".to_string());
            }
            let lhs = parse_bool_expr(parts[1], v_int, v_real, v_float, v_arr, v_bv, resolver)?;
            let rhs = parse_bool_expr(parts[2], v_int, v_real, v_float, v_arr, v_bv, resolver)?;
            Ok(lhs.xor(&rhs))
        }
        "not" | "~" => {
            if parts.len() != 2 {
                return Err("not expects exactly one argument".to_string());
            }
            Ok(parse_bool_expr(parts[1], v_int, v_real, v_float, v_arr, v_bv, resolver)?.not())
        }
        "ite" => {
            if parts.len() != 4 {
                return Err("ite (if-then-else) expects 3 arguments".to_string());
            }
            let cond = parse_bool_expr(parts[1], v_int, v_real, v_float, v_arr, v_bv, resolver)?;
            let then = parse_bool_expr(parts[2], v_int, v_real, v_float, v_arr, v_bv, resolver)?;
            let orelse = parse_bool_expr(parts[3], v_int, v_real, v_float, v_arr, v_bv, resolver)?;
            Ok(cond.ite(&then, &orelse))
        }
        "=" | "!=" | "<" | "<=" | ">" | ">=" => {
            if parts.len() != 3 {
                return Err(format!("Invalid comparison: {:?}", parts));
            }
            if parts[0] == "=" || parts[0] == "!=" {
                // Try parsing as boolean comparison first (for (= v_bool_dest (comparison ...)))
                let lhs_bool =
                    parse_bool_expr(parts[1], v_int, v_real, v_float, v_arr, v_bv, resolver);
                let rhs_bool =
                    parse_bool_expr(parts[2], v_int, v_real, v_float, v_arr, v_bv, resolver);
                if let (Ok(l), Ok(r)) = (lhs_bool, rhs_bool) {
                    return Ok(if parts[0] == "=" { l.eq(&r) } else { l.xor(&r) });
                }
            }

            if v_float.is_some() {
                let lhs = super::floats::parse_float_expr(parts[1], v_float, v_arr, resolver)?;
                let rhs = super::floats::parse_float_expr(parts[2], v_float, v_arr, resolver)?;
                match parts[0] {
                    "=" => Ok(float_eq(&lhs, &rhs)),
                    "!=" => Ok(float_eq(&lhs, &rhs).not()),
                    "<" => Ok(float_lt(&lhs, &rhs)),
                    "<=" => Ok(float_le(&lhs, &rhs)),
                    ">" => Ok(float_gt(&lhs, &rhs)),
                    ">=" => Ok(float_ge(&lhs, &rhs)),
                    _ => unreachable!(),
                }
            } else if v_real.is_some() {
                let lhs = super::reals::parse_real_expr(parts[1], v_real, v_arr, resolver)?;
                let rhs = super::reals::parse_real_expr(parts[2], v_real, v_arr, resolver)?;
                match parts[0] {
                    "=" => Ok(lhs.eq(&rhs)),
                    "!=" => Ok(lhs.eq(&rhs).not()),
                    "<" => Ok(lhs.lt(&rhs)),
                    "<=" => Ok(lhs.le(&rhs)),
                    ">" => Ok(lhs.gt(&rhs)),
                    ">=" => Ok(lhs.ge(&rhs)),
                    _ => unreachable!(),
                }
            } else {
                let can_parse_bv = resolver.is_some() || (v_bv.is_some() && !has_other_vars(sexpr));
                if can_parse_bv {
                    let lhs_bv = super::bitvectors::parse_bv_expr(parts[1], v_bv, v_int, resolver);
                    let rhs_bv = super::bitvectors::parse_bv_expr(parts[2], v_bv, v_int, resolver);
                    if let (Ok(l), Ok(r)) = (lhs_bv, rhs_bv) {
                        let size_l = l.get_size();
                        let size_r = r.get_size();
                        let (l_unified, r_unified) = if size_l > size_r {
                            (l.clone(), r.zero_ext(size_l - size_r))
                        } else if size_r > size_l {
                            (l.zero_ext(size_r - size_l), r.clone())
                        } else {
                            (l, r)
                        };
                        return match parts[0] {
                            "=" => Ok(l_unified.eq(&r_unified)),
                            "!=" => Ok(l_unified.eq(&r_unified).not()),
                            "<" => Ok(l_unified.bvslt(&r_unified)),
                            "<=" => Ok(l_unified.bvsle(&r_unified)),
                            ">" => Ok(l_unified.bvsgt(&r_unified)),
                            ">=" => Ok(l_unified.bvsge(&r_unified)),
                            _ => unreachable!(),
                        };
                    }
                }

                let lhs = super::integers::parse_int_expr(parts[1], v_int, v_arr, v_bv, resolver)?;
                let rhs = super::integers::parse_int_expr(parts[2], v_int, v_arr, v_bv, resolver)?;
                match parts[0] {
                    "=" => Ok(lhs.eq(&rhs)),
                    "!=" => Ok(lhs.eq(&rhs).not()),
                    "<" => Ok(lhs.lt(&rhs)),
                    "<=" => Ok(lhs.le(&rhs)),
                    ">" => Ok(lhs.gt(&rhs)),
                    ">=" => Ok(lhs.ge(&rhs)),
                    _ => unreachable!(),
                }
            }
        }
        _ => Err(format!("Unknown boolean operator: {}", parts[0])),
    }
}

fn has_other_vars(sexpr: &str) -> bool {
    let clean = sexpr.replace("VALUE_PLACEHOLDER", "");
    let mut chars = clean.chars().peekable();
    while let Some(c) = chars.next() {
        if c == 'v' {
            if let Some(&next_c) = chars.peek() {
                if next_c.is_ascii_digit() {
                    return true;
                }
            }
        }
    }
    false
}
