use super::CodegenContext;
use cranelift::prelude::*;
use cranelift_module::Module;
use lirien_ir::ir::Value as SsaValue;

pub fn get_val_with_ctx<M: Module>(ctx: &CodegenContext<M>, val: &SsaValue) -> Value {
    if let Some(&v) = ctx.values.get(val) {
        v
    } else if let Some(unpacked) = ctx.unpacked_values.get(val) {
        if !unpacked.is_empty() {
            unpacked[0]
        } else {
            panic!("Value v{} has empty unpacked_values", val.0);
        }
    } else {
        panic!("Value v{} not found in values map", val.0);
    }
}

pub fn get_val(values: &std::collections::HashMap<SsaValue, Value>, val: &SsaValue) -> Value {
    match values.get(val) {
        Some(&v) => v,
        None => panic!("Value v{} not found in values map", val.0),
    }
}

pub fn get_all_cl_values<M: Module>(ctx: &CodegenContext<M>, val: &SsaValue) -> Vec<Value> {
    let ty = ctx.ssa_func.get_type(*val);
    let base_ty = ty.base_type();
    match base_ty {
        lirien_ir::ir::Type::NamedTuple(_) | lirien_ir::ir::Type::Tuple(_) => {
            if let Some(unpacked) = ctx.unpacked_values.get(val) {
                unpacked.clone()
            } else {
                vec![get_val_with_ctx(ctx, val)]
            }
        }
        lirien_ir::ir::Type::Buffer(_) => vec![
            get_val_with_ctx(ctx, val),
            *ctx.buffer_lengths
                .get(val)
                .unwrap_or_else(|| panic!("Length for v{} not found", val.0)),
        ],
        lirien_ir::ir::Type::Tensor(_, ref _dims) => {
            let mut res = vec![get_val_with_ctx(ctx, val)];
            res.extend(
                ctx.tensor_dims
                    .get(val)
                    .unwrap_or_else(|| panic!("Dims for v{} not found", val.0)),
            );
            res
        }
        _ => vec![get_val_with_ctx(ctx, val)],
    }
}

pub fn get_len(lengths: &std::collections::HashMap<SsaValue, Value>, val: &SsaValue) -> Value {
    *lengths
        .get(val)
        .unwrap_or_else(|| panic!("Length for v{} not found", val.0))
}
