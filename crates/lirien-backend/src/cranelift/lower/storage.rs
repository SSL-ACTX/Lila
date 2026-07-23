use super::CodegenContext;
use cranelift::codegen::ir::{MemFlagsData, StackSlot};
use cranelift::prelude::*;
use cranelift_module::Module;

pub fn copy_memory(builder: &mut FunctionBuilder, src_ptr: Value, dest_ptr: Value, size: usize) {
    let mut curr_offset = 0;
    while curr_offset < size {
        let bytes_left = size - curr_offset;
        let (cl_ty, chunk_size) = if bytes_left >= 8 {
            (types::I64, 8)
        } else if bytes_left >= 4 {
            (types::I32, 4)
        } else if bytes_left >= 2 {
            (types::I16, 2)
        } else {
            (types::I8, 1)
        };

        let val = builder
            .ins()
            .load(cl_ty, MemFlagsData::new(), src_ptr, curr_offset as i32);
        builder
            .ins()
            .store(MemFlagsData::new(), val, dest_ptr, curr_offset as i32);
        curr_offset += chunk_size;
    }
}

pub fn stack_store_with_ptr_ty(
    builder: &mut FunctionBuilder,
    val: Value,
    slot: StackSlot,
    offset: i32,
    ptr_ty: Type,
) {
    let addr = builder.ins().stack_addr(ptr_ty, slot, offset);
    builder.ins().store(MemFlagsData::new(), val, addr, 0);
}

pub fn stack_store_safe<M: Module>(
    ctx: &mut CodegenContext<M>,
    val: Value,
    slot: StackSlot,
    offset: i32,
) {
    let ptr_ty = ctx.module.target_config().pointer_type();
    stack_store_with_ptr_ty(&mut ctx.builder, val, slot, offset, ptr_ty);
}

pub fn copy_to_stack(
    builder: &mut FunctionBuilder,
    src_ptr: Value,
    slot: StackSlot,
    slot_offset: i32,
    size: usize,
    ptr_ty: Type,
) {
    let mut curr_offset = 0;
    while curr_offset < size {
        let bytes_left = size - curr_offset;
        let (cl_ty, chunk_size) = if bytes_left >= 8 {
            (types::I64, 8)
        } else if bytes_left >= 4 {
            (types::I32, 4)
        } else if bytes_left >= 2 {
            (types::I16, 2)
        } else {
            (types::I8, 1)
        };

        let val = builder
            .ins()
            .load(cl_ty, MemFlagsData::new(), src_ptr, curr_offset as i32);
        stack_store_with_ptr_ty(builder, val, slot, slot_offset + curr_offset as i32, ptr_ty);
        curr_offset += chunk_size;
    }
}

pub enum StorageDest {
    Stack(StackSlot),
    Addr(Value),
}

impl StorageDest {
    pub fn store<M: Module>(&self, ctx: &mut CodegenContext<M>, val: Value, offset: i32) {
        match self {
            StorageDest::Stack(slot) => {
                stack_store_safe(ctx, val, *slot, offset);
            }
            StorageDest::Addr(ptr) => {
                ctx.builder
                    .ins()
                    .store(MemFlagsData::new(), val, *ptr, offset);
            }
        }
    }

    pub fn copy<M: Module>(
        &self,
        ctx: &mut CodegenContext<M>,
        src_val: Value,
        offset: i32,
        size: usize,
    ) {
        match self {
            StorageDest::Stack(slot) => {
                let ptr_ty = ctx.module.target_config().pointer_type();
                copy_to_stack(&mut ctx.builder, src_val, *slot, offset, size, ptr_ty);
            }
            StorageDest::Addr(ptr) => {
                let size_val = ctx.builder.ins().iconst(types::I64, size as i64);
                let dest_addr = ctx.builder.ins().iadd_imm(*ptr, offset as i64);
                ctx.builder
                    .call_memcpy(ctx.module.target_config(), dest_addr, src_val, size_val);
            }
        }
    }
}

pub fn store_recursive<M: Module>(
    ctx: &mut CodegenContext<M>,
    ty: &lirien_ir::ir::Type,
    flat_vals: &[Value],
    dest: &StorageDest,
    current_offset: &mut i32,
    val_idx: &mut usize,
) {
    let align = ty.align(&ctx.ssa_func.struct_layouts) as i32;
    *current_offset = (*current_offset + align - 1) & !(align - 1);
    let start_offset = *current_offset;

    match ty {
        lirien_ir::ir::Type::NamedTuple(ref name) => {
            let fields = ctx.ssa_func.struct_layouts.get(name).unwrap().clone();
            for (_, f_ty) in fields {
                store_recursive(ctx, &f_ty, flat_vals, dest, current_offset, val_idx);
            }
        }
        lirien_ir::ir::Type::Tuple(ref types) => {
            let types = types.clone();
            for t in types {
                store_recursive(ctx, &t, flat_vals, dest, current_offset, val_idx);
            }
        }
        lirien_ir::ir::Type::Buffer(_) => {
            let ptr_val = flat_vals[*val_idx];
            let len_val = flat_vals[*val_idx + 1];
            dest.store(ctx, ptr_val, start_offset);
            dest.store(ctx, len_val, start_offset + 8);
            *current_offset += 16;
            *val_idx += 2;
        }
        lirien_ir::ir::Type::Tensor(_, dims) => {
            let ptr_val = flat_vals[*val_idx];
            dest.store(ctx, ptr_val, start_offset);
            for i in 0..dims.len() {
                let dim_val = flat_vals[*val_idx + 1 + i];
                dest.store(ctx, dim_val, start_offset + 8 + 8 * i as i32);
            }
            *current_offset += 8 + 8 * dims.len() as i32;
            *val_idx += 1 + dims.len();
        }
        _ => {
            let val = flat_vals[*val_idx];
            let size = ty.size(&ctx.ssa_func.struct_layouts) as i32;
            if ty.is_composite() {
                dest.copy(ctx, val, start_offset, size as usize);
            } else {
                dest.store(ctx, val, start_offset);
            }
            *current_offset += size;
            *val_idx += 1;
        }
    }
}

pub fn store_to_stack<M: Module>(
    ctx: &mut CodegenContext<M>,
    src: lirien_ir::ir::Value,
    slot: StackSlot,
    slot_offset: i32,
) {
    let ty = ctx.ssa_func.get_type(src);
    let dest = StorageDest::Stack(slot);
    if let Some(flat_vals) = ctx.unpacked_values.get(&src).cloned() {
        let mut current_offset = slot_offset;
        let mut val_idx = 0;
        store_recursive(
            ctx,
            &ty,
            &flat_vals,
            &dest,
            &mut current_offset,
            &mut val_idx,
        );
    } else {
        let val = super::utils::get_val(&ctx.values, &src);
        let size = ty.size(&ctx.ssa_func.struct_layouts) as i32;
        if ty.is_composite() {
            dest.copy(ctx, val, slot_offset, size as usize);
        } else {
            dest.store(ctx, val, slot_offset);
        }
    }
}

pub fn store_to_memory<M: Module>(
    ctx: &mut CodegenContext<M>,
    src: lirien_ir::ir::Value,
    dest_ptr: Value,
    dest_offset: i32,
) {
    let ty = ctx.ssa_func.get_type(src);
    let dest = StorageDest::Addr(dest_ptr);
    if let Some(flat_vals) = ctx.unpacked_values.get(&src).cloned() {
        let mut current_offset = dest_offset;
        let mut val_idx = 0;
        store_recursive(
            ctx,
            &ty,
            &flat_vals,
            &dest,
            &mut current_offset,
            &mut val_idx,
        );
    } else {
        let val = super::utils::get_val(&ctx.values, &src);
        let size = ty.size(&ctx.ssa_func.struct_layouts) as i32;
        if ty.is_composite() {
            dest.copy(ctx, val, dest_offset, size as usize);
        } else {
            dest.store(ctx, val, dest_offset);
        }
    }
}
