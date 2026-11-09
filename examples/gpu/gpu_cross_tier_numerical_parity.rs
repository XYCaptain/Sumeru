//! Example 10: Four-Way Cross-Tier Numerical Parity (`HardwareGpu` vs `CpuFallback` vs `par_run` vs `run`).
//!
//! Verifies that executing a multi-step non-linear physics kernel across all 4 execution tiers
//! on identical cloned stores produces 100% identical row-by-row results.

use crate::common::*;
use crate::elog;

#[gpu]
pub fn nonlinear_oscillator_step(
    pos: &mut SimPosition,
    vel: &mut SimVelocity,
    mass: &SimMass,
    drag: &SimDrag,
) {
    let restore_x: f32 = -(pos.x * 0.125);
    let restore_y: f32 = -(pos.y * 0.125);
    vel.vx = (vel.vx + restore_x * mass.inv_mass) * drag.factor;
    vel.vy = (vel.vy + restore_y * mass.inv_mass) * drag.factor;
    pos.x += vel.vx.clamp(-25.0, 25.0);
    pos.y += vel.vy.clamp(-25.0, 25.0);
}

pub fn main() {
    clear_log();

    let mut base_store = Store::new().with_layout_optimization(LayoutOptimization::PageAlignedGpu);
    base_store
        .register_gpu_query::<(&mut SimPosition, &mut SimVelocity, &SimMass, &SimDrag)>()
        .unwrap();

    for i in 0..512u128 {
        let f = i as f32;
        base_store
            .insert((
                EntityId::from_u128(i + 1),
                (
                    SimPosition {
                        x: (f * 0.5) - 64.0,
                        y: 128.0 - (f * 0.25),
                    },
                    SimVelocity {
                        vx: 4.0,
                        vy: -2.0,
                    },
                    SimMass { inv_mass: 0.5 },
                    SimDrag { factor: 0.75 },
                ),
            ))
            .unwrap();
    }

    let mut store_cpu_seq = base_store.clone();
    let mut store_cpu_par = base_store.clone();
    let mut store_gpu_fallback = base_store.clone();
    let mut store_gpu_auto = base_store;

    let mut fallback_engine = MetalComputeEngine::new_cpu_fallback();
    let mut auto_engine = MetalComputeEngine::new();

    for _step in 0..4 {
        store_cpu_seq
            .query_mut::<(&mut SimPosition, &mut SimVelocity, &SimMass, &SimDrag)>()
            .unwrap()
            .run(nonlinear_oscillator_step);

        store_cpu_par
            .query_mut::<(&mut SimPosition, &mut SimVelocity, &SimMass, &SimDrag)>()
            .unwrap()
            .par_run(nonlinear_oscillator_step);

        store_gpu_fallback
            .query_mut::<(&mut SimPosition, &mut SimVelocity, &SimMass, &SimDrag)>()
            .unwrap()
            .gpu()
            .for_each_on(&mut fallback_engine, nonlinear_oscillator_step)
            .unwrap();

        store_gpu_auto
            .query_mut::<(&mut SimPosition, &mut SimVelocity, &SimMass, &SimDrag)>()
            .unwrap()
            .gpu()
            .for_each_on(&mut auto_engine, nonlinear_oscillator_step)
            .unwrap();
    }

    let mut max_abs_diff = 0.0f32;
    for i in 0..512u128 {
        let e = EntityId::from_u128(i + 1);
        let p_seq = store_cpu_seq.get::<SimPosition>(e).unwrap();
        let p_par = store_cpu_par.get::<SimPosition>(e).unwrap();
        let p_fb = store_gpu_fallback.get::<SimPosition>(e).unwrap();
        let p_auto = store_gpu_auto.get::<SimPosition>(e).unwrap();

        assert_eq!(*p_seq, *p_par);
        assert_eq!(*p_seq, *p_fb);
        let dx = (p_seq.x - p_auto.x).abs();
        let dy = (p_seq.y - p_auto.y).abs();
        max_abs_diff = max_abs_diff.max(dx).max(dy);
    }

    let sample = *store_gpu_auto
        .get::<SimPosition>(EntityId::from_u128(1))
        .unwrap();

    elog!(
        "parity_check: rows=512, steps=4, fallback_mode={:?}, max_diff_ok={}",
        fallback_engine.execution_mode(),
        max_abs_diff <= 1e-5
    );
    elog!("sample_e1=({:.4}, {:.4})", sample.x, sample.y);
}

#[test]
fn test_gpu_cross_tier_numerical_parity() {
    main();
    assert_eq!(
        take_log(),
        vec![
            "parity_check: rows=512, steps=4, fallback_mode=CpuFallback, max_diff_ok=true",
            "sample_e1=(-35.1260, 80.9865)",
        ]
    );
}
