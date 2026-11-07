//! Example 6: `MetalComputeEngine` Cross-Frame Buffer Cache, SIMD32 Grid & 16KB Page Table.
//!
//! Demonstrates reusing `MetalComputeEngine` across frames for 100% zero-copy `MTLBuffer`
//! cache hits, computing 32-wide Apple SIMD-group threadgroup grids, and building a
//! cross-archetype `GpuPageTable`.

use crate::common::*;
use crate::elog;

pub fn main() {
    clear_log();

    let mut store = Store::new().with_layout_optimization(LayoutOptimization::PageAlignedGpu);
    store
        .register_gpu_query::<(&mut SimPosition, &SimVelocity)>()
        .unwrap();

    for i in 0..500u128 {
        store
            .insert((
                EntityId::from_u128(i + 1),
                (
                    SimPosition {
                        x: i as f32,
                        y: 0.0,
                    },
                    SimVelocity { vx: 1.0, vy: 2.0 },
                ),
            ))
            .unwrap();
    }

    let mut engine = MetalComputeEngine::new();

    // Frame 1: Cold cache -> registers 2 zero-copy 16KB buffers (SimPosition & SimVelocity)
    let frame1 = store
        .query_mut::<(&mut SimPosition, &SimVelocity)>()
        .unwrap()
        .gpu()
        .for_each_on(&mut engine, integrate_motion)
        .unwrap();

    // Frame 2: Warm cache -> 100% cache hit on both 16KB buffers without reallocation
    let frame2 = store
        .query_mut::<(&mut SimPosition, &SimVelocity)>()
        .unwrap()
        .gpu()
        .run_on(&mut engine, integrate_motion)
        .unwrap();

    // Compute 1D SIMD32-aligned threadgroup dispatch grid for 500 active rows
    let grid = MetalThreadgroupDispatch::compute_1d(500, 256);

    // Build cross-archetype 16KB GPU page table
    let page_table = store.build_gpu_page_table::<SimPosition>().unwrap();

    elog!(
        "frame1: hits={}, misses={} | frame2: hits={}, misses={}",
        frame1.buffer_cache_hits,
        frame1.buffer_cache_misses,
        frame2.buffer_cache_hits,
        frame2.buffer_cache_misses
    );
    elog!(
        "grid_500: groups={}, threads_per_group={}, total_threads={}, page_table_rows={}",
        grid.threadgroup_count,
        grid.threads_per_threadgroup,
        grid.total_grid_threads,
        page_table.total_active_rows()
    );
}

#[test]
fn test_gpu_engine_buffer_cache_and_grid() {
    main();
    assert_eq!(
        take_log(),
        vec![
            "frame1: hits=0, misses=2 | frame2: hits=2, misses=0",
            "grid_500: groups=2, threads_per_group=256, total_threads=512, page_table_rows=500",
        ]
    );
}
