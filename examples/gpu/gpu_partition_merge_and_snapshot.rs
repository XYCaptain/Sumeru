//! Example 8: Partition Merge Promotion, Zero-Copy `rkyv` Snapshot & `ChangedSince` Loop.
//!
//! Demonstrates merging a worker partition (`Compact` layout) into a `PageAlignedGpu` store
//! (automatically promoting columns to 16KB page alignment), running a GPU compute pass,
//! taking a whole-column `rkyv` snapshot, restoring it, and tracking 64-row dirty chunks.

use crate::common::*;
use crate::elog;

pub fn main() {
    clear_log();

    // Primary simulation store configured for 16KB page-aligned GPU zero-copy
    let mut main_store = Store::new().with_layout_optimization(LayoutOptimization::PageAlignedGpu);
    main_store
        .register_gpu_query::<(&mut SimPosition, &SimVelocity)>()
        .unwrap();

    // Worker partition uses default Compact layout to stage 128 entities independently
    let mut worker_partition = Store::new();
    for i in 0..128u128 {
        worker_partition
            .insert((
                EntityId::from_u128(i + 1000),
                (
                    SimPosition {
                        x: i as f32,
                        y: 10.0,
                    },
                    SimVelocity { vx: 3.0, vy: 4.0 },
                ),
            ))
            .unwrap();
    }

    // Merge worker partition into main_store: columns are automatically promoted to 16KB alignment
    main_store.merge_partition(worker_partition).unwrap();
    let merged_aligned = main_store.gpu_column_views::<SimPosition>(0).unwrap()[0]
        .1
        .is_zero_copy_aligned();

    // Run GPU step on the merged state
    main_store
        .query_mut::<(&mut SimPosition, &SimVelocity)>()
        .unwrap()
        .gpu()
        .for_each(integrate_motion)
        .unwrap();

    // Capture a zero-copy `rkyv` snapshot and restore into a fresh PageAlignedGpu store
    let snapshot = main_store.snapshot().unwrap();
    let mut restored = Store::new().with_layout_optimization(LayoutOptimization::PageAlignedGpu);
    restored
        .register_gpu_query::<(&mut SimPosition, &SimVelocity)>()
        .unwrap();
    restored.restore_snapshot(&snapshot).unwrap();

    let baseline_epoch = restored.epoch();
    // Mutate only entity 1000 (row 0, inside 64-row chunk 0: rows 0..64)
    restored
        .modify::<SimPosition, _>(EntityId::from_u128(1000), |p| p.x += 10.0)
        .unwrap();

    // `changed_since` visits only the dirty 64-row chunk (64 rows out of 128)
    let dirty_chunk_rows = restored
        .query::<&SimPosition>()
        .changed_since::<SimPosition>(baseline_epoch)
        .iter()
        .count();

    let e1000 = *restored
        .get::<SimPosition>(EntityId::from_u128(1000))
        .unwrap();

    elog!(
        "merge_and_snapshot: merged_aligned={}, restored_len={}, dirty_chunk_rows={}",
        merged_aligned,
        restored.len(),
        dirty_chunk_rows
    );
    elog!("e1000_after_restore_and_mod=({:.1}, {:.1})", e1000.x, e1000.y);
}

#[test]
fn test_gpu_partition_merge_and_snapshot() {
    main();
    assert_eq!(
        take_log(),
        vec![
            "merge_and_snapshot: merged_aligned=true, restored_len=128, dirty_chunk_rows=64",
            "e1000_after_restore_and_mod=(13.0, 14.0)",
        ]
    );
}
