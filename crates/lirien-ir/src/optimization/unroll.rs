//! Post-verification SSA Loop Unrolling Pass.
//!
//! Flattens static cyclic loops post-verification into straight-line SSA instructions
//! prior to machine code lowering with Cranelift.

use crate::ir::{BasicBlock, BlockId, Function, Instruction, InstructionKind, Type, Value};
use std::collections::HashMap;

const MAX_UNROLL_LIMIT: i64 = 128;

/// Unrolls static cyclic loops in a function post-verification.
pub fn unroll_loops(func: &mut Function) {
    let mut changed = true;
    let mut iterations = 0;
    while changed && iterations < 10 {
        changed = false;
        iterations += 1;
        if try_unroll_one_loop(func) {
            changed = true;
        }
    }
}

fn get_const_int_val(func: &Function, val: Value) -> Option<i64> {
    for block in &func.blocks {
        for inst in &block.instructions {
            if let InstructionKind::ConstInt(d, v) = inst.kind {
                if d == val {
                    return Some(v);
                }
            }
        }
    }
    None
}

struct LoopInfo {
    header_id: BlockId,
    entry_pred: BlockId,
    inc_pred: BlockId,
    body_id: BlockId,
    exit_id: BlockId,
    idx_var: Value,
    start_c: i64,
    _end_c: i64,
    step_c: i64,
    trip_count: i64,
}

fn detect_static_loop(func: &Function) -> Option<LoopInfo> {
    for block in &func.blocks {
        let header_id = block.id;

        // Header block must have instructions and end with a Branch
        let branch_inst = match block.instructions.last() {
            Some(inst) => match &inst.kind {
                InstructionKind::Branch(cond, t_block, f_block) => (*cond, *t_block, *f_block),
                _ => continue,
            },
            None => continue,
        };

        let (_cond_val, body_id, exit_id) = branch_inst;

        // Find Phi instruction for loop induction variable
        for inst in &block.instructions {
            if let InstructionKind::Phi(dest, incoming) = &inst.kind {
                if incoming.len() != 2 {
                    continue;
                }

                // Check if one incoming is backedge (from block that jumps to header) and one is entry
                let mut entry_pair = None;
                let mut backedge_pair = None;

                for (&pred, &v) in incoming {
                    let pred_block = func.blocks.iter().find(|b| b.id == pred);
                    let jumps_to_header = pred_block.is_some_and(|b| {
                        b.instructions.last().is_some_and(|last| match &last.kind {
                            InstructionKind::Jump(target) => *target == header_id,
                            _ => false,
                        })
                    });

                    if jumps_to_header {
                        backedge_pair = Some((pred, v));
                    } else {
                        entry_pair = Some((pred, v));
                    }
                }

                let (entry_pred, start_val) = match entry_pair {
                    Some(p) => p,
                    None => continue,
                };
                let (inc_pred, next_val) = match backedge_pair {
                    Some(p) => p,
                    None => continue,
                };

                let start_c = match get_const_int_val(func, start_val) {
                    Some(c) => c,
                    None => continue,
                };

                // Find condition instruction SLt/SLe/SGt/SGe/Ne
                let mut end_c_opt = None;
                for h_inst in &block.instructions {
                    match &h_inst.kind {
                        InstructionKind::SLt(_, l, r) | InstructionKind::ULt(_, l, r)
                            if *l == *dest =>
                        {
                            end_c_opt = get_const_int_val(func, *r);
                        }
                        InstructionKind::SGt(_, l, r) | InstructionKind::UGt(_, l, r)
                            if *l == *dest =>
                        {
                            end_c_opt = get_const_int_val(func, *r);
                        }
                        _ => {}
                    }
                }

                let end_c = match end_c_opt {
                    Some(c) => c,
                    None => continue,
                };

                // Find step value from Add in inc_pred
                let inc_block = match func.blocks.iter().find(|b| b.id == inc_pred) {
                    Some(b) => b,
                    None => continue,
                };

                let mut step_c = 1i64;
                for inc_inst in &inc_block.instructions {
                    if let InstructionKind::Add(d, l, r) = inc_inst.kind {
                        if d == next_val {
                            let step_val = if l == *dest {
                                r
                            } else if r == *dest {
                                l
                            } else {
                                continue;
                            };
                            if let Some(sc) = get_const_int_val(func, step_val) {
                                step_c = sc;
                            }
                        }
                    }
                }

                if step_c == 0 {
                    continue;
                }

                let trip_count = if step_c > 0 {
                    if end_c > start_c {
                        (end_c - start_c + step_c - 1) / step_c
                    } else {
                        0
                    }
                } else if start_c > end_c {
                    (start_c - end_c + (-step_c) - 1) / (-step_c)
                } else {
                    0
                };

                if trip_count <= MAX_UNROLL_LIMIT {
                    return Some(LoopInfo {
                        header_id,
                        entry_pred,
                        inc_pred,
                        body_id,
                        exit_id,
                        idx_var: *dest,
                        start_c,
                        _end_c: end_c,
                        step_c,
                        trip_count,
                    });
                }
            }
        }
    }
    None
}

fn try_unroll_one_loop(func: &mut Function) -> bool {
    let info = match detect_static_loop(func) {
        Some(i) => i,
        None => return false,
    };

    tracing::info!(
        target: "lirien::ssa::unroll",
        "Unrolling loop header b{:?} (trip_count={})",
        info.header_id,
        info.trip_count
    );

    let trip_count = info.trip_count;

    if trip_count == 0 {
        // Redirect entry_pred directly to exit_id
        if let Some(entry_b) = func.blocks.iter_mut().find(|b| b.id == info.entry_pred) {
            if let Some(last) = entry_b.instructions.last_mut() {
                if let InstructionKind::Jump(ref mut target) = last.kind {
                    if *target == info.header_id {
                        *target = info.exit_id;
                    }
                }
            }
        }
        return true;
    }

    // Collect body instructions before mutating func to avoid borrow checker conflicts
    let body_block_ids: Vec<BlockId> = vec![info.body_id, info.inc_pred];
    let mut orig_body_instructions: Vec<(BlockId, Vec<Instruction>)> = Vec::new();
    for &b_id in &body_block_ids {
        if let Some(orig_block) = func.blocks.iter().find(|b| b.id == b_id) {
            orig_body_instructions.push((b_id, orig_block.instructions.clone()));
        }
    }

    let mut iteration_entry_blocks = Vec::new();
    let mut last_iter_exit_block = info.entry_pred;

    for i in 0..trip_count {
        let current_idx_c = info.start_c + i * info.step_c;
        let iter_body_id = func.next_block();
        iteration_entry_blocks.push(iter_body_id);

        let mut val_map = HashMap::new();
        let curr_idx_val = func.next_value();
        func.set_type(curr_idx_val, Type::I64);
        val_map.insert(info.idx_var, curr_idx_val);

        let mut new_instructions = Vec::new();

        // Push const int for current index
        new_instructions.push(Instruction {
            kind: InstructionKind::ConstInt(curr_idx_val, current_idx_c),
            location: None,
            constraints: Vec::new(),
        });

        // Clone instructions from saved body block instructions
        for (_b_id, insts) in &orig_body_instructions {
            for inst in insts {
                match &inst.kind {
                    InstructionKind::Jump(target) => {
                        let next_target = if *target == info.inc_pred || *target == info.header_id {
                            if i == trip_count - 1 {
                                info.exit_id
                            } else {
                                BlockId(func.block_count) // Will be next iteration's iter_body_id
                            }
                        } else {
                            *target
                        };
                        new_instructions.push(Instruction {
                            kind: InstructionKind::Jump(next_target),
                            location: inst.location,
                            constraints: Vec::new(),
                        });
                    }
                    InstructionKind::Branch(cond, t, f) => {
                        let mapped_cond = val_map.get(cond).copied().unwrap_or(*cond);
                        new_instructions.push(Instruction {
                            kind: InstructionKind::Branch(mapped_cond, *t, *f),
                            location: inst.location,
                            constraints: Vec::new(),
                        });
                    }
                    _ => {
                        let mut cloned_kind = inst.kind.clone();
                        if let Some(def) = inst.get_def() {
                            let new_def = func.next_value();
                            let existing_ty = func.get_type(def);
                            func.set_type(new_def, existing_ty);
                            val_map.insert(def, new_def);
                            set_inst_def(&mut cloned_kind, new_def);
                        }
                        remap_inst_operands(&mut cloned_kind, &val_map);
                        new_instructions.push(Instruction {
                            kind: cloned_kind,
                            location: inst.location,
                            constraints: inst.constraints.clone(),
                        });
                    }
                }
            }
        }

        let iter_block = BasicBlock {
            id: iter_body_id,
            instructions: new_instructions,
            predecessors: vec![last_iter_exit_block],
            successors: Vec::new(),
        };

        func.blocks.push(iter_block);
        last_iter_exit_block = iter_body_id;
    }

    // Connect entry_pred to iteration 0
    if let Some(entry_b) = func.blocks.iter_mut().find(|b| b.id == info.entry_pred) {
        if let Some(last) = entry_b.instructions.last_mut() {
            if let InstructionKind::Jump(ref mut target) = last.kind {
                if *target == info.header_id {
                    *target = iteration_entry_blocks[0];
                }
            }
        }
    }

    true
}

fn set_inst_def(kind: &mut InstructionKind, new_def: Value) {
    match kind {
        InstructionKind::Add(ref mut d, ..)
        | InstructionKind::Sub(ref mut d, ..)
        | InstructionKind::Mul(ref mut d, ..)
        | InstructionKind::SDiv(ref mut d, ..)
        | InstructionKind::UDiv(ref mut d, ..)
        | InstructionKind::SRem(ref mut d, ..)
        | InstructionKind::URem(ref mut d, ..)
        | InstructionKind::FAdd(ref mut d, ..)
        | InstructionKind::FSub(ref mut d, ..)
        | InstructionKind::FMul(ref mut d, ..)
        | InstructionKind::FDiv(ref mut d, ..)
        | InstructionKind::ConstInt(ref mut d, ..)
        | InstructionKind::ConstFloat(ref mut d, ..)
        | InstructionKind::Assign(ref mut d, ..)
        | InstructionKind::Eq(ref mut d, ..)
        | InstructionKind::Ne(ref mut d, ..)
        | InstructionKind::SLt(ref mut d, ..)
        | InstructionKind::SLe(ref mut d, ..)
        | InstructionKind::SGt(ref mut d, ..)
        | InstructionKind::SGe(ref mut d, ..)
        | InstructionKind::ULt(ref mut d, ..)
        | InstructionKind::ULe(ref mut d, ..)
        | InstructionKind::UGt(ref mut d, ..)
        | InstructionKind::UGe(ref mut d, ..)
        | InstructionKind::FLt(ref mut d, ..)
        | InstructionKind::FLe(ref mut d, ..)
        | InstructionKind::FGt(ref mut d, ..)
        | InstructionKind::FGe(ref mut d, ..)
        | InstructionKind::BufferLoad(ref mut d, ..)
        | InstructionKind::BufferStore(ref mut d, ..)
        | InstructionKind::ArrayLoad(ref mut d, ..)
        | InstructionKind::ArrayStore(ref mut d, ..)
        | InstructionKind::StructLoad(ref mut d, ..)
        | InstructionKind::StructSet(ref mut d, ..)
        | InstructionKind::Phi(ref mut d, ..) => {
            *d = new_def;
        }
        _ => {}
    }
}

fn remap_inst_operands(kind: &mut InstructionKind, map: &HashMap<Value, Value>) {
    let remap = |v: &mut Value| {
        if let Some(&new_v) = map.get(v) {
            *v = new_v;
        }
    };

    match kind {
        InstructionKind::Add(_, l, r)
        | InstructionKind::Sub(_, l, r)
        | InstructionKind::Mul(_, l, r)
        | InstructionKind::SDiv(_, l, r)
        | InstructionKind::UDiv(_, l, r)
        | InstructionKind::SRem(_, l, r)
        | InstructionKind::URem(_, l, r)
        | InstructionKind::FAdd(_, l, r)
        | InstructionKind::FSub(_, l, r)
        | InstructionKind::FMul(_, l, r)
        | InstructionKind::FDiv(_, l, r)
        | InstructionKind::Eq(_, l, r)
        | InstructionKind::Ne(_, l, r)
        | InstructionKind::SLt(_, l, r)
        | InstructionKind::SLe(_, l, r)
        | InstructionKind::SGt(_, l, r)
        | InstructionKind::SGe(_, l, r)
        | InstructionKind::ULt(_, l, r)
        | InstructionKind::ULe(_, l, r)
        | InstructionKind::UGt(_, l, r)
        | InstructionKind::UGe(_, l, r)
        | InstructionKind::FLt(_, l, r)
        | InstructionKind::FLe(_, l, r)
        | InstructionKind::FGt(_, l, r)
        | InstructionKind::FGe(_, l, r) => {
            remap(l);
            remap(r);
        }
        InstructionKind::Assign(_, s) => {
            remap(s);
        }
        InstructionKind::BufferLoad(_, buf, idx) => {
            remap(buf);
            remap(idx);
        }
        InstructionKind::BufferStore(_, buf, idx, val, _) => {
            remap(buf);
            remap(idx);
            remap(val);
        }
        InstructionKind::ArrayLoad(_, arr, idx) => {
            remap(arr);
            remap(idx);
        }
        InstructionKind::ArrayStore(_, arr, idx, val, _) => {
            remap(arr);
            remap(idx);
            remap(val);
        }
        InstructionKind::StructLoad(_, obj, _) => {
            remap(obj);
        }
        InstructionKind::StructSet(_, obj, _, val, _) => {
            remap(obj);
            remap(val);
        }
        _ => {}
    }
}
