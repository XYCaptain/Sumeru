//! Example 4: 16KB Hardware Page Geometry, 64-Row Chunk Epochs & Coalesced `DirtyPageRun`.
//!
//! Demonstrates inspecting 16KB physical page alignment, zero-padded tail safety,
//! fine-grained 16KB dirty page bitmaps derived from 64-row chunk epochs, and `swap_remove`
//! physical dirty page tracking.

use crate::common::*;
use crate::elog;

pub fn main() {
    clear_log();

    let mut store = Store::new().with_layout_optimization(LayoutOptimization::PageAlignedGpu);
    store
        .register_gpu_query::<(&mut SimPosition, &SimVelocity)>()
        .unwrap();

    // Each `SimPosition` is 8 bytes -> 2048 rows per 16KB (16384B) hardware page.
    // Inserting 2300 entities spans 2 hardware pages (Page 0: rows 0..2048, Page 1: rows 2048..2300).
    let mut entities = Vec::with_capacity(2300);
    for i in 0..2300u128 {
        let e = EntityId::from_u128(i + 1);
        store
            .insert((
                e,
                (
                    SimPosition {
                        x: i as f32,
                        y: 1.0,
                    },
                    SimVelocity { vx: 1.0, vy: 1.0 },
                ),
            ))
            .unwrap();
        entities.push(e);
    }

    let baseline_epoch = store.epoch();
    let views = store.gpu_column_views::<SimPosition>(baseline_epoch).unwrap();
    let (_, view0) = &views[0];
    let geom = view0.geometry();

    elog!(
        "geometry: aligned={}, elem_count={}, page_count={}, allocated_bytes={}",
        view0.is_zero_copy_aligned(),
        geom.elem_count,
        geom.page_count,
        geom.allocated_bytes
    );

    // Modify row 2100 (resides in Page 1: byte offset 16800 >= 16384). Page 0 remains clean.
    store
        .modify::<SimPosition, _>(entities[2100], |pos| pos.x += 500.0)
        .unwrap();

    let views_after_mod = store.gpu_column_views::<SimPosition>(baseline_epoch).unwrap();
    let dirty_mod = views_after_mod[0].1.dirty_bitmap();
    let runs = dirty_mod.coalesced_runs();

    elog!(
        "after_page1_mod: page0_dirty={}, page1_dirty={}, run_offset={}, run_len={}",
        dirty_mod.is_page_dirty(0),
        dirty_mod.is_page_dirty(1),
        runs[0].byte_offset,
        runs[0].byte_len
    );

    // Record a new epoch and delete row 10 (in Page 0).
    // `swap_remove` moves the last row (from Page 1) into row 10 (Page 0), marking Page 0 dirty.
    let pre_delete_epoch = store.epoch();
    store.delete_entity(entities[10]).unwrap();

    let views_after_del = store
        .gpu_column_views::<SimPosition>(pre_delete_epoch)
        .unwrap();
    let dirty_del = views_after_del[0].1.dirty_bitmap();

    elog!(
        "after_swap_remove: len={}, page0_dirty={}",
        views_after_del[0].1.len(),
        dirty_del.is_page_dirty(0)
    );
}

#[test]
fn test_gpu_page_aligned_dirty_tracking() {
    main();
    assert_eq!(
        take_log(),
        vec![
            "geometry: aligned=true, elem_count=2300, page_count=2, allocated_bytes=32768",
            "after_page1_mod: page0_dirty=false, page1_dirty=true, run_offset=16384, run_len=16384",
            "after_swap_remove: len=2299, page0_dirty=true",
        ]
    );
}
