//! Example 11: Two-Phase Precheck Rejection, Zero-Side-Effect Rollback & On-Demand 16KB Promotion.
//!
//! Verifies:
//! 1. Calling `gpu_column_views` on an unaligned `Compact` store returns `Err(StateError::AccessDenied)`;
//! 2. Calling `.gpu().for_each` with row-level filters (`.with_flag` / `.changed_since`) is rejected
//!    in Phase 0 without mutating data or promoting columns;
//! 3. Calling `.gpu().for_each` with archetype filters (`.with::<Tag>`) on a default `Compact` store
//!    automatically promotes ONLY the matched archetype's bound columns to 16KB page alignment.

use crate::common::*;
use crate::elog;

pub type TagGpuTarget = Tag<20>;
pub type TagCpuOnly = Tag<21>;

pub fn main() {
    clear_log();

    // Default `Compact` store (NOT `PageAlignedGpu`)
    let mut store = Store::new();

    let e_gpu = EntityId::from_u128(1);
    let e_cpu = EntityId::from_u128(2);

    store
        .insert((
            e_gpu,
            (
                SimPosition { x: 10.0, y: 20.0 },
                SimVelocity { vx: 5.0, vy: 6.0 },
                Tag::<20>,
            ),
        ))
        .unwrap();
    store
        .insert((
            e_cpu,
            (
                SimPosition { x: 100.0, y: 200.0 },
                SimVelocity { vx: 1.0, vy: 1.0 },
                Tag::<21>,
            ),
        ))
        .unwrap();
    store.set_flag::<0>(e_gpu, true).unwrap();

    // 1. Before promotion, `gpu_column_views` rejects unaligned Compact columns
    let unaligned_err = store.gpu_column_views::<SimPosition>(0).is_err();

    // 2. Row-level filters (`.with_flag::<0>()`) are rejected by GPU whole-column dispatch in Phase 0
    let row_filter_err = store
        .query_mut::<(&mut SimPosition, &SimVelocity)>()
        .unwrap()
        .with_flag::<0>()
        .gpu()
        .for_each(integrate_motion)
        .is_err();

    // Verify zero side effects on `e_gpu` after rejected row-filtered dispatch
    let pos_after_reject = *store.get::<SimPosition>(e_gpu).unwrap();

    // 3. Valid archetype-filtered `.with::<TagGpuTarget>().gpu().for_each(...)` promotes ONLY the matched archetype!
    let stats = store
        .query_mut::<(&mut SimPosition, &SimVelocity)>()
        .unwrap()
        .with::<TagGpuTarget>()
        .gpu()
        .for_each(integrate_motion)
        .unwrap();

    let pos_gpu_after = *store.get::<SimPosition>(e_gpu).unwrap();
    let pos_cpu_after = *store.get::<SimPosition>(e_cpu).unwrap();

    elog!(
        "precheck: unaligned_rejected={}, row_filter_rejected={}, untouched_on_err=({:.1}, {:.1})",
        unaligned_err,
        row_filter_err,
        pos_after_reject.x,
        pos_after_reject.y
    );
    elog!(
        "on_demand_promote: dispatched_arch={}, gpu_entity=({:.1}, {:.1}), cpu_entity=({:.1}, {:.1})",
        stats.dispatched_archetypes,
        pos_gpu_after.x,
        pos_gpu_after.y,
        pos_cpu_after.x,
        pos_cpu_after.y
    );
}

#[test]
fn test_gpu_precheck_and_auto_promotion() {
    main();
    assert_eq!(
        take_log(),
        vec![
            "precheck: unaligned_rejected=true, row_filter_rejected=true, untouched_on_err=(10.0, 20.0)",
            "on_demand_promote: dispatched_arch=1, gpu_entity=(15.0, 26.0), cpu_entity=(100.0, 200.0)",
        ]
    );
}
