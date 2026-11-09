//! 64 行物理块变更纪元索引与无锁原子推进原语 (`ChunkEpochs` & `AtomicChunkEpochs`)
//!
//! # 性能设计与文献支撑
//!
//! 在高性能列式引擎与实体组件系统（ECS）中，细粒度脏检查通常面临两难：
//! - 逐实体/逐行维护版本号：内存膨胀 100%~200%，严重破坏缓存局部性；
//! - 整表/整列单版本号：无法感知局部写入，导致查询流式过滤与增量快照全面退化。
//!
//! Sumeru 采用 **64 行块纪元 (64-Row Chunk Epochs)**：
//! 1. **每 64 行对齐物理块** 分配单个 64 位版本号（`u64`）。
//! 2. 内存元数据开销极低：每行平均仅占 $\frac{8 \text{ Bytes}}{64} = 1 \text{ bit}$，内存膨胀率 $< 0.2\%$。
//! 3. 与 `RowBitMask`（64 位机器字）及 CPU 64 字节缓存行完美几何对齐，
//!    单个 `u64` 掩码字对应的行区间恰好对应一个块纪元槽位。
//! 4. 提供常规 [`ChunkEpochs`]（单所有者栈优化）与 [`AtomicChunkEpochs`]（多核流水线无锁并发并发更新）。

use std::sync::atomic::{AtomicU64, Ordering};
use crate::storage::primitives::bitmask::RowBitMask;

/// 物理块对齐行数：64 行（对应单个 `u64` 机器字与 64B/128B 缓存行整数倍）
pub const CHUNK_ROWS: usize = 64;

/// 单列 64 行物理块变更纪元索引
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct ChunkEpochs {
    chunks: smallvec::SmallVec<[u64; 4]>,
}

impl ChunkEpochs {
    /// 创建空的块纪元索引
    #[inline]
    pub fn new() -> Self {
        Self {
            chunks: smallvec::SmallVec::new(),
        }
    }

    /// 预分配容量
    #[inline]
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            chunks: smallvec::SmallVec::with_capacity(capacity),
        }
    }

    /// 底层切片引用
    #[inline(always)]
    pub fn as_slice(&self) -> &[u64] {
        &self.chunks
    }

    /// 底层可变切片引用
    #[inline(always)]
    pub fn as_mut_slice(&mut self) -> &mut [u64] {
        &mut self.chunks
    }

    /// 当前记录的块数
    #[inline(always)]
    pub fn len(&self) -> usize {
        self.chunks.len()
    }

    /// 是否为空
    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.chunks.is_empty()
    }

    /// 清空所有块纪元
    #[inline]
    pub fn clear(&mut self) {
        self.chunks.clear();
    }

    /// 调整块纪元大小，新块填充 `def_ep`
    #[inline]
    pub fn resize(&mut self, new_len: usize, def_ep: u64) {
        self.chunks.resize(new_len, def_ep);
    }

    /// 截断到指定块数
    #[inline]
    pub fn truncate(&mut self, len: usize) {
        self.chunks.truncate(len);
    }

    /// 获取指定索引的元素引用
    #[inline(always)]
    pub fn get(&self, index: usize) -> Option<&u64> {
        self.chunks.get(index)
    }

    /// 获取指定块索引的变更纪元；若超出范围则回退到 `default_epoch`
    #[inline(always)]
    pub fn get_chunk_epoch(&self, chunk_idx: usize, default_epoch: u64) -> u64 {
        self.chunks.get(chunk_idx).copied().unwrap_or(default_epoch)
    }

    /// 标记第 `row` 行所在 64 行块的变更纪元
    #[inline]
    pub fn mark_row(&mut self, row: usize, epoch: u64, total_rows: usize, default_epoch: u64) {
        let chunk_idx = row >> 6;
        let num_chunks = (total_rows.max(row + 1) + 63) >> 6;
        if self.chunks.is_empty() {
            self.chunks.resize(num_chunks, default_epoch);
        } else if self.chunks.len() <= chunk_idx {
            self.chunks.resize(num_chunks.max(chunk_idx + 1), default_epoch);
        }
        self.chunks[chunk_idx] = epoch;
    }

    /// 标记全列所有 64 行块的变更纪元
    #[inline]
    pub fn mark_all(&mut self, total_rows: usize, epoch: u64) {
        let num_chunks = (total_rows + 63) >> 6;
        self.chunks.clear();
        self.chunks.resize(num_chunks, epoch);
    }

    /// 仅标记掩码非 0 的 64 行块变更纪元
    #[inline]
    pub fn mark_masked(
        &mut self,
        mask: &RowBitMask,
        epoch: u64,
        total_rows: usize,
        default_epoch: u64,
    ) {
        let num_chunks = (total_rows + 63) >> 6;
        if self.chunks.len() < num_chunks {
            self.chunks.resize(num_chunks, default_epoch);
        }
        for (w_idx, &word) in mask.words().iter().enumerate() {
            if word != 0 && w_idx < self.chunks.len() {
                self.chunks[w_idx] = epoch;
            }
        }
    }
}

impl std::ops::Index<usize> for ChunkEpochs {
    type Output = u64;
    #[inline(always)]
    fn index(&self, index: usize) -> &Self::Output {
        &self.chunks[index]
    }
}

impl std::ops::IndexMut<usize> for ChunkEpochs {
    #[inline(always)]
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.chunks[index]
    }
}

/// 多核流水线无锁并发块纪元集合 (`AtomicChunkEpochs`)
///
/// 用于 Stage 并行系统执行期间，多个 worker 线程在无全局锁、无原型表写锁的情况下，
/// 并发递增所负责 64 行子块的变更版本号。
pub struct AtomicChunkEpochs {
    chunks: Vec<AtomicU64>,
}

impl AtomicChunkEpochs {
    /// 使用指定的块数量与初始纪元创建原子块集合
    pub fn new(num_chunks: usize, initial_epoch: u64) -> Self {
        let mut chunks = Vec::with_capacity(num_chunks);
        for _ in 0..num_chunks {
            chunks.push(AtomicU64::new(initial_epoch));
        }
        Self { chunks }
    }

    /// 从普通切片快速转换
    pub fn from_slice(slice: &[u64]) -> Self {
        let mut chunks = Vec::with_capacity(slice.len());
        for &ep in slice {
            chunks.push(AtomicU64::new(ep));
        }
        Self { chunks }
    }

    /// 原子更新指定块的纪元（取最大值）
    #[inline]
    pub fn mark_chunk(&self, chunk_idx: usize, epoch: u64) {
        if let Some(atomic_ep) = self.chunks.get(chunk_idx) {
            atomic_ep.fetch_max(epoch, Ordering::Relaxed);
        }
    }

    /// 原子更新指定行所在 64 行块的纪元
    #[inline]
    pub fn mark_row(&self, row: usize, epoch: u64) {
        self.mark_chunk(row >> 6, epoch);
    }

    /// 读取指定块的纪元
    #[inline]
    pub fn get_chunk(&self, chunk_idx: usize) -> u64 {
        self.chunks
            .get(chunk_idx)
            .map(|a| a.load(Ordering::Relaxed))
            .unwrap_or(0)
    }

    /// 拷贝回普通 `Vec<u64>`
    pub fn to_vec(&self) -> Vec<u64> {
        self.chunks.iter().map(|a| a.load(Ordering::Relaxed)).collect()
    }

    /// 将当前原子版本号写回普通切片
    pub fn copy_into_slice(&self, out: &mut [u64]) {
        let count = out.len().min(self.chunks.len());
        for i in 0..count {
            out[i] = self.chunks[i].load(Ordering::Relaxed);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chunk_epochs_basic() {
        let mut epochs = ChunkEpochs::new();
        assert!(epochs.is_empty());

        // 130 行实体，对应 3 个 64 行块：[0..64], [64..128], [128..130]
        epochs.mark_row(70, 10, 130, 1);
        assert_eq!(epochs.len(), 3);
        assert_eq!(epochs.get_chunk_epoch(0, 1), 1);
        assert_eq!(epochs.get_chunk_epoch(1, 1), 10);
        assert_eq!(epochs.get_chunk_epoch(2, 1), 1);
        assert_eq!(epochs.get_chunk_epoch(3, 1), 1); // 越界回退 default_epoch
    }

    #[test]
    fn test_chunk_epochs_masked() {
        let mut epochs = ChunkEpochs::new();
        let mut mask = RowBitMask::new_empty(200);
        mask.set(5);   // chunk 0
        mask.set(150); // chunk 2

        epochs.mark_masked(&mask, 42, 200, 5);
        assert_eq!(epochs.len(), 4);
        assert_eq!(epochs.get_chunk_epoch(0, 0), 42);
        assert_eq!(epochs.get_chunk_epoch(1, 0), 5);
        assert_eq!(epochs.get_chunk_epoch(2, 0), 42);
        assert_eq!(epochs.get_chunk_epoch(3, 0), 5);
    }

    #[test]
    fn test_atomic_chunk_epochs() {
        let atomic = AtomicChunkEpochs::new(4, 10);
        assert_eq!(atomic.get_chunk(1), 10);

        atomic.mark_chunk(1, 20);
        assert_eq!(atomic.get_chunk(1), 20);

        // fetch_max 不会降低版本号
        atomic.mark_chunk(1, 15);
        assert_eq!(atomic.get_chunk(1), 20);

        let mut out = vec![0u64; 4];
        atomic.copy_into_slice(&mut out);
        assert_eq!(out, vec![10, 20, 10, 10]);
    }
}
