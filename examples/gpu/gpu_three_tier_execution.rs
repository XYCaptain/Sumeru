//! Example 2: Three-Tier Explicit Traversal (CPU Sequential / Rayon Morsel Parallel / GPU Zero-Copy).
//!
//! Demonstrates executing the exact same `#[gpu]` operator and closure logic across:
//! 1. Single-threaded CPU (`.for_each` / `.run`),
//! 2. Multi-core Rayon morsel-driven parallel CPU (`.par_for_each` / `.par().run`),
//! 3. 16KB page-aligned zero-copy GPU compute (`.gpu().for_each` / `.gpu().run`).

use crate::common::*;
use crate::elog;

pub fn main() {
    clear_log();

    let mut store = Store::new().with_layout_optimization(LayoutOptimization::PageAlignedGpu);

    for i in 0..2048u128 {
        let entity = EntityId::from_u128(i + 1);
        store
            .insert((
                entity,
                (
                    SimPosition {
                        x: i as f32,
                        y: (i * 2) as f32,
                    },
                    SimVelocity { vx: 1.0, vy: 2.0 },
                ),
            ))
            .unwrap();
    }

    // Tier 1: CPU sequential closure & kernel operator
    store
        .query_mut::<(&mut SimPosition, &SimVelocity)>()
        .unwrap()
        .for_each(|(pos, vel)| {
            pos.x += vel.vx;
            pos.y += vel.vy;
        });
    store
        .query_mut::<(&mut SimPosition, &SimVelocity)>()
        .unwrap()
        .run(integrate_motion);

    let after_cpu = *store.get::<SimPosition>(EntityId::from_u128(1)).unwrap();

    // Tier 2: Multi-core CPU (Rayon 1024-row morsel-driven parallel traversal)
    store
        .query_mut::<(&mut SimPosition, &SimVelocity)>()
        .unwrap()
        .par_for_each(|(pos, vel)| {
            pos.x += vel.vx;
            pos.y += vel.vy;
        });
    store
        .query_mut::<(&mut SimPosition, &SimVelocity)>()
        .unwrap()
        .par()
        .run(integrate_motion);

    let after_par = *store.get::<SimPosition>(EntityId::from_u128(1)).unwrap();

    // Tier 3: GPU zero-copy dispatch (.gpu().for_each & .gpu().run)
    store
        .query_mut::<(&mut SimPosition, &SimVelocity)>()
        .unwrap()
        .gpu()
        .for_each(integrate_motion)
        .unwrap();
    let stats = store
        .query_mut::<(&mut SimPosition, &SimVelocity)>()
        .unwrap()
        .gpu()
        .run(integrate_motion)
        .unwrap();

    let after_gpu = *store.get::<SimPosition>(EntityId::from_u128(1)).unwrap();
    let last_gpu = *store.get::<SimPosition>(EntityId::from_u128(2048)).unwrap();

    elog!(
        "tier1_cpu=({:.1}, {:.1}), tier2_par=({:.1}, {:.1}), tier3_gpu=({:.1}, {:.1})",
        after_cpu.x,
        after_cpu.y,
        after_par.x,
        after_par.y,
        after_gpu.x,
        after_gpu.y
    );
    elog!(
        "morsel_rows={}, last_entity=({:.1}, {:.1})",
        stats.total_rows,
        last_gpu.x,
        last_gpu.y
    );
}

#[test]
fn test_gpu_three_tier_execution() {
    main();
    assert_eq!(
        take_log(),
        vec![
            "tier1_cpu=(2.0, 4.0), tier2_par=(4.0, 8.0), tier3_gpu=(6.0, 12.0)",
            "morsel_rows=2048, last_entity=(2053.0, 4106.0)",
        ]
    );
}
