//! Example 7: Multi-Archetype GPU Dispatch, Multi-Parameter Kernels & `.with` / `.without` Filtering.
//!
//! Demonstrates dispatching a 4-parameter `#[gpu]` kernel across multiple archetypes filtered
//! by zero-sized `Tag<N>` markers while ensuring non-matching archetypes remain untouched and clean.

use crate::common::*;
use crate::elog;

pub type TagSimulated = Tag<10>;
pub type TagFrozen = Tag<11>;

#[gpu]
pub fn integrate_damped_weighted(
    pos: &mut SimPosition,
    vel: &mut SimVelocity,
    mass: &SimMass,
    drag: &SimDrag,
) {
    vel.vx *= drag.factor;
    vel.vy *= drag.factor;
    pos.x += vel.vx * mass.inv_mass;
    pos.y += vel.vy * mass.inv_mass;
}

pub fn main() {
    clear_log();

    let mut store = Store::new().with_layout_optimization(LayoutOptimization::PageAlignedGpu);

    // Archetype A: Active simulated rigid bodies (matches `.with::<TagSimulated>().without::<TagFrozen>()`)
    let e_active = EntityId::from_u128(1);
    store
        .insert((
            e_active,
            (
                SimPosition { x: 10.0, y: 20.0 },
                SimVelocity { vx: 10.0, vy: 20.0 },
                SimMass { inv_mass: 0.5 },
                SimDrag { factor: 0.8 },
                Tag::<10>,
            ),
        ))
        .unwrap();

    // Archetype B: Active simulated bodies with an extra marker Tag<12> (also matches!)
    let e_active_extra = EntityId::from_u128(2);
    store
        .insert((
            e_active_extra,
            (
                SimPosition { x: 0.0, y: 0.0 },
                SimVelocity { vx: 5.0, vy: 10.0 },
                SimMass { inv_mass: 1.0 },
                SimDrag { factor: 0.5 },
                Tag::<10>,
                Tag::<12>,
            ),
        ))
        .unwrap();

    // Archetype C: Frozen bodies (excluded by `.without::<TagFrozen>()`)
    let e_frozen = EntityId::from_u128(3);
    store
        .insert((
            e_frozen,
            (
                SimPosition { x: 50.0, y: 50.0 },
                SimVelocity { vx: 100.0, vy: 100.0 },
                SimMass { inv_mass: 1.0 },
                SimDrag { factor: 1.0 },
                Tag::<10>,
                Tag::<11>,
            ),
        ))
        .unwrap();

    let epoch_before = store.epoch();

    let stats = store
        .query_mut::<(&mut SimPosition, &mut SimVelocity, &SimMass, &SimDrag)>()
        .unwrap()
        .with::<TagSimulated>()
        .without::<TagFrozen>()
        .gpu()
        .for_each(integrate_damped_weighted)
        .unwrap();

    let p_active = *store.get::<SimPosition>(e_active).unwrap();
    let p_extra = *store.get::<SimPosition>(e_active_extra).unwrap();
    let p_frozen = *store.get::<SimPosition>(e_frozen).unwrap();

    // Verify that the excluded Frozen archetype was NOT marked dirty by the GPU dispatch
    let changed_count = store
        .query::<&SimPosition>()
        .with::<TagFrozen>()
        .changed_since::<SimPosition>(epoch_before)
        .iter()
        .count();

    elog!(
        "filtered_dispatch: archetypes={}, rows={}, frozen_dirty_rows={}",
        stats.dispatched_archetypes,
        stats.total_rows,
        changed_count
    );
    elog!(
        "active=({:.1}, {:.1}), extra=({:.1}, {:.1}), frozen=({:.1}, {:.1})",
        p_active.x,
        p_active.y,
        p_extra.x,
        p_extra.y,
        p_frozen.x,
        p_frozen.y
    );
}

#[test]
fn test_gpu_archetype_filtering_and_multi_table() {
    main();
    assert_eq!(
        take_log(),
        vec![
            "filtered_dispatch: archetypes=2, rows=2, frozen_dirty_rows=0",
            "active=(14.0, 28.0), extra=(2.5, 5.0), frozen=(50.0, 50.0)",
        ]
    );
}
