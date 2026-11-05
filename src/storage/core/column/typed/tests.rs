use super::*;

mod native_block_handoff_tests {
    use super::*;
    use crate::storage::column::ErasedBlock;

    #[test]
    fn test_apply_native_block_scatter() {
        let mut col = TypedColumn::<i64>::new("i64");
        for v in [10i64, 20, 30, 40, 50] {
            col.push(v);
        }

        let block: Box<dyn ErasedBlock> = Box::new(ColumnBlock::new(vec![100i64, 200i64]));
        col.apply_native_block_scatter(block, 0..2, &[1, 3]).unwrap();

        assert_eq!(col.get(0), Some(&10));
        assert_eq!(col.get(1), Some(&100));
        assert_eq!(col.get(2), Some(&30));
        assert_eq!(col.get(3), Some(&200));
        assert_eq!(col.get(4), Some(&50));

        // Error: range mismatch
        let block2: Box<dyn ErasedBlock> = Box::new(ColumnBlock::new(vec![300i64]));
        assert!(col.apply_native_block_scatter(block2, 0..1, &[0, 1]).is_err());

        // Error: type mismatch
        let block3: Box<dyn ErasedBlock> = Box::new(ColumnBlock::new(vec![1.0f64]));
        assert!(col.apply_native_block_scatter(block3, 0..1, &[0]).is_err());
    }

    #[test]
    fn test_apply_native_block_range() {
        let mut col = TypedColumn::<i64>::new("i64");
        for v in [10i64, 20, 30, 40, 50] {
            col.push(v);
        }

        let block: Box<dyn ErasedBlock> = Box::new(ColumnBlock::new(vec![99i64, 98i64]));
        col.apply_native_block_range(block, 0..2, 2).unwrap();

        assert_eq!(col.get(0), Some(&10));
        assert_eq!(col.get(1), Some(&20));
        assert_eq!(col.get(2), Some(&99));
        assert_eq!(col.get(3), Some(&98));
        assert_eq!(col.get(4), Some(&50));

        // Error: out of bounds
        let block2: Box<dyn ErasedBlock> = Box::new(ColumnBlock::new(vec![1i64, 2, 3]));
        assert!(col.apply_native_block_range(block2, 0..4, 0).is_err());

        // Error: type mismatch
        let block3: Box<dyn ErasedBlock> = Box::new(ColumnBlock::new(vec![1.0f64]));
        assert!(col.apply_native_block_range(block3, 0..1, 0).is_err());
    }
}
