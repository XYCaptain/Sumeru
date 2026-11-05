//! ColumnBacking 纯连续内存后备单元测试

use super::*;

#[test]
fn test_column_block_and_direct_block_lifecycle() {
    let block = ColumnBlock::new(vec![10u64, 20, 30]);
    assert_eq!(block.len(), 3);
    assert!(!block.is_empty());
    assert_eq!(block.as_slice(), &[10, 20, 30]);
    assert_eq!(block.into_vec(), vec![10, 20, 30]);

    let mut direct = DirectBlock::Owned(vec![1u32, 2, 3]);
    assert_eq!(direct.as_slice(), &[1, 2, 3]);
    let frozen = direct.freeze();
    assert_eq!(frozen.as_slice(), &[1, 2, 3]);
    assert_eq!(direct.as_slice(), &[1, 2, 3]);

    // 共享引用下 make_owned 触发写时复制
    let vec_ref = direct.make_owned();
    vec_ref.push(4);
    assert_eq!(direct.as_slice(), &[1, 2, 3, 4]);
    assert_eq!(frozen.as_slice(), &[1, 2, 3]);

    // 唯一持有者下 make_owned 零拷贝解包
    drop(frozen);
    let frozen_unique = direct.freeze();
    drop(frozen_unique);
    let vec_unique = direct.make_owned();
    vec_unique.push(5);
    assert_eq!(direct.as_slice(), &[1, 2, 3, 4, 5]);
}

#[test]
fn test_column_backing_contiguous_ops() {
    let mut backing = ColumnBacking::<u64>::with_capacity(8);
    assert!(backing.is_empty());
    assert_eq!(backing.len(), 0);
    assert!(!backing.is_segmented());
    assert_eq!(backing.segment_count(), 0);

    backing.push(100);
    backing.push(200);
    backing.push(300);

    assert_eq!(backing.len(), 3);
    assert!(!backing.is_empty());
    assert!(!backing.is_segmented());
    assert_eq!(backing.segment_count(), 1);
    assert_eq!(backing.as_contiguous_slice(), &[100, 200, 300]);
    assert_eq!(backing.try_as_contiguous_slice(), Some(&[100, 200, 300][..]));
    assert_eq!(backing.get_ref(1), Some(&200));
    assert_eq!(backing.get_clone(2), Some(300));
    assert_eq!(backing.get_ref(99), None);

    let arc_block = backing.freeze_contiguous();
    assert_eq!(arc_block.as_slice(), &[100, 200, 300]);

    let owned = backing.make_contiguous_owned();
    owned[1] = 250;
    assert_eq!(backing.as_contiguous_slice(), &[100, 250, 300]);
    assert_eq!(arc_block.as_slice(), &[100, 200, 300]);

    let removed = backing.swap_remove(0);
    assert_eq!(removed, Some(100));
    assert_eq!(backing.as_contiguous_slice(), &[300, 250]);

    backing.clear();
    assert!(backing.is_empty());
    assert_eq!(backing.segment_count(), 0);
}
