//! Example 1: `#[gpu]` Unified Macro & 16KB Page-Aligned Zero-Copy Dispatch.
//!
//! Demonstrates defining `#[gpu]` components and kernels, initializing a `Store` with
//! `LayoutOptimization::PageAlignedGpu`, and dispatching a zero-copy GPU compute pass.

use crate::common::*;
use crate::elog;

pub fn main() {
    clear_log();

    let mut store = Store::new().with_layout_optimization(LayoutOptimization::PageAlignedGpu);

    // Spawn 128 entities with SimPosition and SimVelocity
    for i in 0..128u128 {
        let entity = EntityId::from_u128(i + 1);
        store
            .insert((
                entity,
                (
                    SimPosition {
                        x: (i * 10) as f32,
                        y: (i * 20) as f32,
                    },
                    SimVelocity { vx: 1.5, vy: -0.5 },
                ),
            ))
            .unwrap();
    }

    // Dispatch `integrate_motion` via the 16KB page-aligned GPU runner
    let stats = store
        .query_mut::<(&mut SimPosition, &SimVelocity)>()
        .unwrap()
        .gpu()
        .for_each(integrate_motion)
        .unwrap();

    let e0 = *store.get::<SimPosition>(EntityId::from_u128(1)).unwrap();
    let e127 = *store.get::<SimPosition>(EntityId::from_u128(128)).unwrap();

    elog!(
        "gpu_hello_world: archetypes={}, rows={}, pages_16k={}",
        stats.dispatched_archetypes,
        stats.total_rows,
        stats.total_pages_16k
    );
    elog!(
        "e0=({:.1}, {:.1}), e127=({:.1}, {:.1})",
        e0.x,
        e0.y,
        e127.x,
        e127.y
    );
}

#[test]
fn test_gpu_hello_world() {
    main();
    assert_eq!(
        take_log(),
        vec![
            "gpu_hello_world: archetypes=1, rows=128, pages_16k=2",
            "e0=(1.5, -0.5), e127=(1271.5, 2539.5)",
        ]
    );
}
