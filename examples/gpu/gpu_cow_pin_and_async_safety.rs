//! Example 5: Zero-Copy Column Pinning (`GpuColumnPin`) & Copy-On-Write (COW) Async Isolation.
//!
//! Demonstrates how `store.pin_gpu_columns::<T>()` and `MetalAsyncCommandTicket` pin 16KB
//! physical pages for in-flight async GPU readers while subsequent CPU mutations or mutable
//! GPU dispatches transparently trigger Copy-On-Write (COW) without corrupting pinned memory.

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
                        x: (i * 10) as f32,
                        y: 5.0,
                    },
                    SimVelocity { vx: 2.0, vy: 3.0 },
                ),
            ))
            .unwrap();
    }

    let mut engine = MetalComputeEngine::new();
    store
        .query_mut::<(&mut SimPosition, &SimVelocity)>()
        .unwrap()
        .gpu()
        .for_each_on(&mut engine, integrate_motion)
        .unwrap();

    // Pin `SimPosition` columns for an in-flight async GPU command buffer
    let pins = store.pin_gpu_columns::<SimPosition>().unwrap();
    let pinned_gen = pins[0].generation();
    let pinned_first_x_before = pins[0].as_slice()[0].x;

    // Attach a cloned pin to an async command ticket
    let mut ticket = MetalAsyncCommandTicket::new(1);
    ticket.retain_pin(pins[0].clone());

    // Mutate via GPU while the pin & ticket are alive:
    // `SimPosition` (mutable) triggers COW into a new 16KB buffer generation,
    // while `SimVelocity` (read-only) reuses its cached zero-copy buffer!
    let stats_cow = store
        .query_mut::<(&mut SimPosition, &SimVelocity)>()
        .unwrap()
        .gpu()
        .for_each_on(&mut engine, integrate_motion)
        .unwrap();

    let pinned_first_x_after = pins[0].as_slice()[0].x;
    let store_first_x_after = store.get::<SimPosition>(EntityId::from_u128(1)).unwrap().x;
    let new_gen = store.gpu_column_views::<SimPosition>(0).unwrap()[0]
        .1
        .generation();

    elog!(
        "pin_isolation: ticket_pins={}, pinned_x={:.1}, live_store_x={:.1}",
        ticket.pinned_column_count(),
        pinned_first_x_after,
        store_first_x_after
    );
    elog!(
        "cow_stats: gen_advanced={}, cache_hits={}, cache_misses={}",
        new_gen > pinned_gen && pinned_first_x_before == pinned_first_x_after,
        stats_cow.buffer_cache_hits,
        stats_cow.buffer_cache_misses
    );

    drop(ticket);
    drop(pins);
}

#[test]
fn test_gpu_cow_pin_and_async_safety() {
    main();
    assert_eq!(
        take_log(),
        vec![
            "pin_isolation: ticket_pins=1, pinned_x=2.0, live_store_x=4.0",
            "cow_stats: gen_advanced=true, cache_hits=1, cache_misses=1",
        ]
    );
}
