//! Example 12: Persistent `Query::gpu_for_each_on`, `gpu_column_views_mut` & Buffer Cache Eviction.
//!
//! Demonstrates:
//! 1. Building a Level-2 cached `Query` via `QueryBuilder` and dispatching `gpu_for_each_on`;
//! 2. Mutating 16KB pages directly via `store.gpu_column_views_mut::<T>()` while respecting COW pins;
//! 3. Evicting archetype buffers (`evict_archetype_buffers`) and clearing `MetalBufferCache`.

use crate::common::*;
use crate::elog;

pub fn main() {
    clear_log();

    let mut store = Store::new().with_layout_optimization(LayoutOptimization::PageAlignedGpu);
    store
        .register_gpu_query::<(&mut SimPosition, &SimVelocity)>()
        .unwrap();

    for i in 0..64u128 {
        store
            .insert((
                EntityId::from_u128(i + 1),
                (
                    SimPosition {
                        x: (i * 2) as f32,
                        y: 1.0,
                    },
                    SimVelocity { vx: 3.0, vy: 4.0 },
                ),
            ))
            .unwrap();
    }

    let mut engine = MetalComputeEngine::new();
    let query = QueryBuilder::<(&mut SimPosition, &SimVelocity)>::new()
        .build(&store)
        .unwrap();

    // 1. Dispatch via persistent Level-2 `Query`
    let s1 = query
        .gpu_for_each_on(&mut store, &mut engine, integrate_motion)
        .unwrap();
    let s2 = query
        .gpu_for_each_on(&mut store, &mut engine, integrate_motion)
        .unwrap();

    // 2. Pin `SimPosition` and mutate through `gpu_column_views_mut`:
    //    COW allocates a new 16KB buffer so the pinned slice remains untouched!
    let pins = store.pin_gpu_columns::<SimPosition>().unwrap();
    let pinned_x0 = pins[0].as_slice()[0].x;
    let arch_id = pins[0].archetype_id();

    {
        let mut mut_views = store.gpu_column_views_mut::<SimPosition>().unwrap();
        mut_views[0].1.as_mut_slice()[0].x = 500.0;
    }

    let pinned_x0_after = pins[0].as_slice()[0].x;
    let live_x0_after = store.get::<SimPosition>(EntityId::from_u128(1)).unwrap().x;
    drop(pins);

    // 3. Evict archetype buffers from `MetalComputeEngine` and verify cache count drops to 0
    let cache_before_evict = engine.buffer_cache().len();
    engine.evict_archetype_buffers(arch_id);
    let cache_after_evict = engine.buffer_cache().len();

    elog!(
        "persistent_query: s1_misses={}, s2_hits={}, total_dispatches={}",
        s1.buffer_cache_misses,
        s2.buffer_cache_hits,
        engine.total_dispatches()
    );
    elog!(
        "views_mut_and_evict: pinned_x0={:.1}, live_x0={:.1}, cache_before={}, cache_after={}",
        pinned_x0_after,
        live_x0_after,
        cache_before_evict,
        cache_after_evict
    );
    assert_eq!(pinned_x0, pinned_x0_after);
}

#[test]
fn test_gpu_persistent_query_and_views_mut() {
    main();
    assert_eq!(
        take_log(),
        vec![
            "persistent_query: s1_misses=2, s2_hits=2, total_dispatches=2",
            "views_mut_and_evict: pinned_x0=6.0, live_x0=500.0, cache_before=2, cache_after=0",
        ]
    );
}
