//! 13. Typed Shader IR: External `const` Evaluation, `#[gpu]` Helper Composition & Symbol-Table Inference
//!
//! Demonstrates Sumeru's Type-Driven Shader IR (`GpuBlock` / `GpuStmt` / `GpuExpr`):
//! 1. **External Rust `const` Constants (`GpuConstValue`)**: Directly reference module constants
//!    (`SUBSTEP_DT`, `SPEED_LIMIT`) and standard library constants (`std::f32::consts::FRAC_PI_4`)
//!    inside `#[gpu]` functions — evaluated at compile time by `rustc`;
//! 2. **Composable `#[gpu]` Pure Helper Functions (`GpuHelperFnDescriptor`)**: Define modular scalar
//!    device functions (`soft_clamp_axis` -> `damped_substep_displacement`) with tail-return expressions,
//!    automatically linked in topological dependency order into both MSL and WGSL;
//! 3. **Symbol-Table Type Inference (`ShaderTypeEnv`)**: Write `for _step in 0..cfg.substeps` with an
//!    unsuffixed `0` — `ShaderTypeEnv` inspects `SimStepConfig::FIELDS`, infers `cfg.substeps: u32`,
//!    and automatically promotes the loop counter and step to unsigned `0u` / `1u`.

use crate::common::*;

const SUBSTEP_DT: f32 = 0.25;
const SPEED_LIMIT: f32 = 8.0;

#[gpu]
pub struct SimStepConfig {
    pub damping: f32,
    pub substeps: u32,
}

#[gpu]
pub fn soft_clamp_axis(val: f32, limit: f32) -> f32 {
    if val > limit {
        limit
    } else if val < -limit {
        -limit
    } else {
        val
    }
}

#[gpu]
pub fn damped_substep_displacement(vel_axis: f32, damping: f32) -> f32 {
    let clamped = soft_clamp_axis(vel_axis, SPEED_LIMIT);
    clamped * SUBSTEP_DT * damping
}

#[gpu]
pub fn integrate_typed_ir_substeps(
    pos: &mut SimPosition,
    vel: &SimVelocity,
    cfg: &SimStepConfig,
) {
    const BIAS_SCALE: f32 = 0.0;
    let angle_bias = std::f32::consts::FRAC_PI_4 * BIAS_SCALE;
    // Unsuffixed `0..cfg.substeps`: `ShaderTypeEnv` infers `cfg.substeps` as `u32` from `SimStepConfig::FIELDS`
    for _step in 0..cfg.substeps {
        pos.x += damped_substep_displacement(vel.vx, cfg.damping) + angle_bias;
        pos.y += damped_substep_displacement(vel.vy, cfg.damping);
    }
}

pub fn main() {
    clear_log();

    let desc = integrate_typed_ir_substeps.descriptor();
    let compiled = compile_and_validate_pipeline(desc).unwrap();
    let msl = desc.msl_source();
    let wgsl = desc.wgsl_source();

    // Verify topological order of linked helper functions (callee before caller)
    let wgsl_callee_idx = wgsl.find("fn soft_clamp_axis").unwrap();
    let wgsl_caller_idx = wgsl.find("fn damped_substep_displacement").unwrap();
    let topo_ordered = wgsl_callee_idx < wgsl_caller_idx
        && msl.find("inline float soft_clamp_axis").unwrap()
            < msl.find("inline float damped_substep_displacement").unwrap();
    let inferred_u32_loop =
        wgsl.contains("for (var _step = 0u; _step < cfg.substeps; _step += 1u)");

    let mut store_gpu =
        Store::new().with_layout_optimization(LayoutOptimization::PageAlignedGpu);
    let e1 = EntityId::from_u128(1);
    store_gpu
        .insert((
            e1,
            (
                SimPosition { x: 2.0, y: 10.0 },
                SimVelocity { vx: 20.0, vy: -4.0 }, // vx=20 clamped to 8.0; vy=-4 stays -4.0
                SimStepConfig {
                    damping: 0.5,
                    substeps: 4,
                },
            ),
        ))
        .unwrap();
    let mut store_cpu = store_gpu.clone();

    store_gpu
        .query_mut::<(&mut SimPosition, &SimVelocity, &SimStepConfig)>()
        .unwrap()
        .gpu()
        .for_each(integrate_typed_ir_substeps)
        .unwrap();

    store_cpu
        .query_mut::<(&mut SimPosition, &SimVelocity, &SimStepConfig)>()
        .unwrap()
        .run(integrate_typed_ir_substeps);

    let p_gpu = *store_gpu.get::<SimPosition>(e1).unwrap();
    let p_cpu = *store_cpu.get::<SimPosition>(e1).unwrap();
    assert_eq!(p_gpu, p_cpu);

    elog!(
        "typed_ir: validated={}, helpers_topo={}, u32_loop_inferred={}",
        compiled.shader_validated,
        topo_ordered,
        inferred_u32_loop
    );
    elog!("typed_ir_pos=({:.1}, {:.1})", p_gpu.x, p_gpu.y);
}

#[test]
fn test_gpu_typed_ir_consts_and_helpers() {
    main();
    assert_eq!(
        take_log(),
        vec![
            "typed_ir: validated=true, helpers_topo=true, u32_loop_inferred=true",
            "typed_ir_pos=(6.0, 8.0)",
        ]
    );
}
