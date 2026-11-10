//! 连续列式状态存储基础计算原语与切分索引 (Primitives & Indexing)
//!
//! 包含零外部业务依赖的高性能底层计算原语：
//! - [`RowBitMask`] 与连续掩码段合并扫描 [`for_each_contiguous_mask_run`]
//! - [`ArchetypeBitSet`] 倒排原型位图索引
//! - [`ChunkEpochs`] 与 [`AtomicChunkEpochs`] 64 行物理块变更纪元
//! - [`PageDirtyBitmap`] 与 [`DirtyPageRun`] 硬件物理页级脏追踪
//! - [`SubchunkPlanner`] 基于 MonetDB/X100 缓存驻留模型的自适应切块规划器

pub mod bitmask;
pub mod bitset;
pub mod dirty;
pub mod epochs;
pub mod planner;

pub use self::bitmask::{for_each_contiguous_mask_run, mask_has_any_in_range, RowBitMask};
pub use self::bitset::ArchetypeBitSet;
pub use self::dirty::{
    DirtyPageRun, PageDirtyBitmap, GRACE_HOPPER_64K_PAGE_SIZE, HARDWARE_PAGE_SIZE,
};
pub use self::epochs::{AtomicChunkEpochs, ChunkEpochs, CHUNK_ROWS};
pub use self::planner::{
    detect_l1d_cache_bytes, SubchunkPlanner, SubchunkRangeIter, DEFAULT_TARGET_BYTES,
    MAX_SUBCHUNK_ROWS, MIN_SUBCHUNK_ROWS,
};
