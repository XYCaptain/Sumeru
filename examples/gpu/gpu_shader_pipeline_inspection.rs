//! Example 3: Cross-Platform Shader Inspection (MSL / WGSL), Control Flow & Naga Validation.
//!
//! Demonstrates how `#[gpu]` translates Rust control flow (`if`/`else`, early `return;` with
//! automatic mutable buffer write-back) and math intrinsics (`abs`, `clamp`, `mul_add`) into
//! Apple Metal Shading Language (MSL) and WebGPU (WGSL), validated via `naga`.

use crate::common::*;
use crate::elog;

#[gpu]
pub fn bounded_relativistic_step(pos: &mut SimPosition, vel: &SimVelocity, drag: &SimDrag) {
    let damped_vx: f32 = vel.vx.mul_add(drag.factor, 0.0).clamp(-50.0, 50.0);
    let step_x: f32 = damped_vx.abs();
    pos.x += step_x;
    if pos.x >= 100.0 {
        pos.x = 100.0;
        return;
    }
    pos.y += vel.vy * drag.factor;
}

pub fn main() {
    clear_log();

    let msl_struct = SimPosition::msl_struct_definition();
    let wgsl_struct = SimPosition::wgsl_struct_definition();
    let desc = bounded_relativistic_step.descriptor();

    // Validate the generated WGSL & MSL pipeline through Naga's parser, IR validator, and MSL backend
    let compiled = compile_and_validate_pipeline(desc).unwrap();

    elog!(
        "kernel={}, bindings={}, validated={}",
        compiled.entry_point,
        desc.bindings().len(),
        compiled.shader_validated
    );
    elog!(
        "struct_check: msl_float={}, wgsl_f32={}, early_return_writeback={}",
        msl_struct.contains("float x;"),
        wgsl_struct.contains("x: f32,"),
        desc.wgsl_source().contains("sumeru_buf_pos[gid] = pos; return;")
    );

    // Execute kernel on entities that both trigger and bypass the early `return;` branch
    let mut store = Store::new().with_layout_optimization(LayoutOptimization::PageAlignedGpu);
    let e_clamped = EntityId::from_u128(1);
    let e_normal = EntityId::from_u128(2);

    store
        .insert((
            e_clamped,
            (
                SimPosition { x: 95.0, y: 10.0 },
                SimVelocity { vx: -20.0, vy: 30.0 },
                SimDrag { factor: 0.5 },
            ),
        ))
        .unwrap();
    store
        .insert((
            e_normal,
            (
                SimPosition { x: 10.0, y: 10.0 },
                SimVelocity { vx: -20.0, vy: 30.0 },
                SimDrag { factor: 0.5 },
            ),
        ))
        .unwrap();

    store
        .query_mut::<(&mut SimPosition, &SimVelocity, &SimDrag)>()
        .unwrap()
        .gpu()
        .for_each(bounded_relativistic_step)
        .unwrap();

    let p_clamped = *store.get::<SimPosition>(e_clamped).unwrap();
    let p_normal = *store.get::<SimPosition>(e_normal).unwrap();

    elog!(
        "clamped=({:.1}, {:.1}), normal=({:.1}, {:.1})",
        p_clamped.x,
        p_clamped.y,
        p_normal.x,
        p_normal.y
    );
}

#[test]
fn test_gpu_shader_pipeline_inspection() {
    main();
    assert_eq!(
        take_log(),
        vec![
            "kernel=bounded_relativistic_step, bindings=3, validated=true",
            "struct_check: msl_float=true, wgsl_f32=true, early_return_writeback=true",
            "clamped=(100.0, 10.0), normal=(20.0, 25.0)",
        ]
    );
}
