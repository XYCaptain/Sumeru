//! 硬件物理页级脏追踪与 DMA 合并段原语 (`PageDirtyBitmap` & `DirtyPageRun`)
//!
//! # 性能设计与文献支撑
//!
//! 在异构计算（GPU Compute Pass、Apple Metal、NVIDIA GH200 NVLink-C2C）与增量 Checkpoint 中，
//! 细粒度的数据同步必须对齐到底层操作系统与 GPU MMU 的硬件物理页大小。
//!
//! 1. **物理页对齐折叠 (Hardware Page Folding)**:
//!    CPU 侧记录的 64 行块变更纪元（`ChunkEpochs`）通过位运算直接折叠到 16KB（Apple Silicon / 通用硬件页）
//!    或 64KB（ARM64 超算 / NVIDIA Grace Hopper）硬件页位图。
//! 2. **连续脏页段合并 (Contiguous Run Merging)**:
//!    单次扫描生成紧凑的 [`DirtyPageRun`] 列表，避免将碎片化的离散微小传输分发给硬件 DMA 控制器，
//!    最大化 PCIe / NVLink 总线突发传输（Burst Transfer）吞吐。
//!
//! # 唯一类型约定
//!
//! 本模块是 `PageDirtyBitmap` / `DirtyPageRun` / `HARDWARE_PAGE_SIZE` 的**唯一定义点**。
//! `gpu` feature 仅通过同 crate 内的扩展 `impl` 增加构造方法，绝不替换类型本身，
//! 以保证 Cargo feature 的可加性（开启 `gpu` 不会改变 `sumeru::PageDirtyBitmap` 的字段与方法集合）。

use smallvec::SmallVec;
use std::ops::Range;

/// Apple Silicon M 系列 GPU / Linux 64 位系统标准硬件物理页粒度（16KB = 16384 字节）
pub const HARDWARE_PAGE_SIZE: usize = 16384;

/// NVIDIA Grace Hopper (GH200) NVLink-C2C / ARM64 超算节点 64KB 系统页粒度
pub const GRACE_HOPPER_64K_PAGE_SIZE: usize = 65536;

/// 连续硬件脏页段描述符（由 64 行物理块纪元折叠合并生成，直接作为 PCIe / RDMA 增量 DMA 传输段）
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DirtyPageRun {
    /// 起始硬件页号（0-indexed）
    pub first_page: usize,
    /// 连续脏页数量
    pub page_count: usize,
    /// 在列缓冲区内的起始字节偏移（`first_page * page_size`）
    pub byte_offset: usize,
    /// 本次连续传输的字节跨度（`page_count * page_size`，按硬件页边界对齐）
    pub byte_len: usize,
}

impl DirtyPageRun {
    /// 起始硬件页号（`first_page` 的语义别名）
    #[inline]
    pub fn start_page(&self) -> usize {
        self.first_page
    }

    /// 该连续脏页段覆盖的活跃元素行范围。
    #[inline]
    pub fn row_range(&self, elem_size: usize, total_rows: usize) -> Range<usize> {
        if elem_size == 0 || total_rows == 0 {
            return 0..0;
        }
        let start_row = (self.byte_offset / elem_size).min(total_rows);
        let end_byte = self.byte_offset.saturating_add(self.byte_len);
        let end_row = end_byte.div_ceil(elem_size).min(total_rows);
        start_row..end_row
    }
}

/// 硬件页级脏位图 (`PageDirtyBitmap`)
///
/// 以 `u64` 位图字记录每个硬件页（默认 16KB，可配置 4KB / 64KB）的脏状态，
/// 并通过 [`coalesced_runs`](Self::coalesced_runs) / [`dirty_runs`](Self::dirty_runs) 合并为连续 DMA 段。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PageDirtyBitmap {
    page_size: usize,
    page_count: usize,
    words: SmallVec<[u64; 2]>,
}

impl PageDirtyBitmap {
    /// 创建具有指定页数的全净位图（默认 16KB 页）。
    #[inline]
    pub fn new_clean(page_count: usize) -> Self {
        Self::new_clean_with_page_size(page_count, HARDWARE_PAGE_SIZE)
    }

    /// 创建具有指定页数与页粒度的全净位图。
    #[inline]
    pub fn new_clean_with_page_size(page_count: usize, page_size: usize) -> Self {
        let word_count = page_count.div_ceil(64);
        let mut words = SmallVec::with_capacity(word_count);
        words.resize(word_count, 0);
        Self {
            page_size: page_size.max(4096),
            page_count,
            words,
        }
    }

    /// 创建具有指定页数的全脏位图（默认 16KB 页）。
    #[inline]
    pub fn new_all_dirty(page_count: usize) -> Self {
        Self::new_all_dirty_with_page_size(page_count, HARDWARE_PAGE_SIZE)
    }

    /// 创建具有指定页数与页粒度的全脏位图。
    #[inline]
    pub fn new_all_dirty_with_page_size(page_count: usize, page_size: usize) -> Self {
        let mut bitmap = Self::new_clean_with_page_size(page_count, page_size);
        for w in bitmap.words.iter_mut() {
            *w = !0u64;
        }
        if page_count % 64 != 0 {
            if let Some(last) = bitmap.words.last_mut() {
                *last &= (1u64 << (page_count % 64)) - 1;
            }
        }
        bitmap
    }

    /// 按默认 16KB 硬件页粒度（[`HARDWARE_PAGE_SIZE`]），将 64 行物理块纪元折叠为硬件脏页段
    #[inline]
    pub fn from_phys_chunk_epochs(
        chunk_epochs: &[u64],
        default_epoch: u64,
        since_epoch: u64,
        row_len: usize,
        elem_size: usize,
    ) -> Self {
        Self::from_phys_chunk_epochs_with_page_size(
            chunk_epochs,
            default_epoch,
            since_epoch,
            row_len,
            elem_size,
            HARDWARE_PAGE_SIZE,
        )
    }

    /// 按指定硬件页粒度（如 4KB、16KB 或 64KB），将 64 行物理块纪元精确折叠为硬件脏页位图
    ///
    /// 任何 `chunk_epoch > since_epoch` 的 64 行块，其字节跨度触及的所有硬件页均被标记为脏页；
    /// 缺失的块纪元以 `default_epoch` 补齐。
    pub fn from_phys_chunk_epochs_with_page_size(
        chunk_epochs: &[u64],
        default_epoch: u64,
        since_epoch: u64,
        row_len: usize,
        elem_size: usize,
        page_size: usize,
    ) -> Self {
        let page_size = page_size.max(4096);
        let total_bytes = row_len.saturating_mul(elem_size);
        if total_bytes == 0 {
            return Self::new_clean_with_page_size(0, page_size);
        }

        let total_pages = total_bytes.div_ceil(page_size);
        let mut bitmap = Self::new_clean_with_page_size(total_pages, page_size);

        let num_chunks = row_len.div_ceil(64);
        for chunk_idx in 0..num_chunks {
            let ep = chunk_epochs
                .get(chunk_idx)
                .copied()
                .unwrap_or(default_epoch);
            if ep > since_epoch {
                let start_row = chunk_idx * 64;
                let end_row = ((chunk_idx + 1) * 64).min(row_len);
                let start_byte = start_row * elem_size;
                let end_byte = (end_row * elem_size).max(start_byte + 1);

                let first_page = start_byte / page_size;
                let last_page = (end_byte - 1) / page_size;
                for p in first_page..=last_page.min(total_pages - 1) {
                    bitmap.words[p >> 6] |= 1u64 << (p & 63);
                }
            }
        }
        bitmap
    }

    /// 标记指定页为脏。
    #[inline]
    pub fn mark_dirty(&mut self, page_idx: usize) {
        if page_idx < self.page_count {
            self.words[page_idx >> 6] |= 1u64 << (page_idx & 63);
        }
    }

    /// 检查指定页是否为脏。
    #[inline]
    pub fn is_page_dirty(&self, page_idx: usize) -> bool {
        if page_idx >= self.page_count {
            return false;
        }
        (self.words[page_idx >> 6] & (1u64 << (page_idx & 63))) != 0
    }

    /// 硬件页粒度（字节）。
    #[inline]
    pub fn page_size(&self) -> usize {
        self.page_size
    }

    /// 总页数。
    #[inline]
    pub fn page_count(&self) -> usize {
        self.page_count
    }

    /// 总页数（[`page_count`](Self::page_count) 的别名）。
    #[inline]
    pub fn total_pages(&self) -> usize {
        self.page_count
    }

    /// 脏页总数。
    #[inline]
    pub fn dirty_page_count(&self) -> usize {
        self.words.iter().map(|w| w.count_ones() as usize).sum()
    }

    /// 是否存在至少一个脏页。
    #[inline]
    pub fn has_any_dirty(&self) -> bool {
        self.words.iter().any(|&w| w != 0)
    }

    /// 脏页覆盖的总字节数（按页粒度对齐）。
    #[inline]
    pub fn total_dirty_bytes(&self) -> usize {
        self.dirty_page_count() * self.page_size
    }

    /// 底层 `u64` 位图字切片。
    #[inline]
    pub fn words(&self) -> &[u64] {
        &self.words
    }

    /// 将相邻脏页合并为连续的 `DirtyPageRun` 列表（栈上小向量，常见场景零堆分配）。
    pub fn coalesced_runs(&self) -> SmallVec<[DirtyPageRun; 4]> {
        let mut runs = SmallVec::new();
        self.for_each_run(|run| runs.push(run));
        runs
    }

    /// 将相邻脏页合并为连续的 `DirtyPageRun` 列表（堆分配 `Vec`，便于跨线程持有）。
    pub fn dirty_runs(&self) -> Vec<DirtyPageRun> {
        let mut runs = Vec::new();
        self.for_each_run(|run| runs.push(run));
        runs
    }

    #[inline]
    fn for_each_run(&self, mut f: impl FnMut(DirtyPageRun)) {
        let mut page = 0;
        while page < self.page_count {
            // 整字跳过全净区域
            let word = self.words[page >> 6] >> (page & 63);
            if word == 0 {
                page = ((page >> 6) + 1) << 6;
                continue;
            }
            page += word.trailing_zeros() as usize;
            if page >= self.page_count {
                break;
            }
            let first_page = page;
            page += 1;
            while page < self.page_count && self.is_page_dirty(page) {
                page += 1;
            }
            let count = page - first_page;
            f(DirtyPageRun {
                first_page,
                page_count: count,
                byte_offset: first_page * self.page_size,
                byte_len: count * self.page_size,
            });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_page_dirty_bitmap_folding() {
        let mut chunk_epochs = vec![10u64; 32];
        let row_len = 32 * 64; // 2048 行
        let elem_size = 64; // 64 字节/行 -> 每块 64 * 64 = 4096 字节
        // 总字节数: 2048 * 64 = 131072 字节 (8 个 16KB 页)

        chunk_epochs[2] = 15; // chunk 2: 8KB..12KB -> page 0 (0..16KB)
        chunk_epochs[5] = 15; // chunk 5: 20KB..24KB -> page 1 (16KB..32KB)
        chunk_epochs[24] = 16; // chunk 24: 96KB..100KB -> page 6 (96KB..112KB)

        let dirty = PageDirtyBitmap::from_phys_chunk_epochs(&chunk_epochs, 10, 10, row_len, elem_size);
        assert_eq!(dirty.total_pages(), 8);
        assert_eq!(dirty.dirty_page_count(), 3);
        assert!(dirty.is_page_dirty(0));
        assert!(dirty.is_page_dirty(1));
        assert!(dirty.is_page_dirty(6));
        assert!(!dirty.is_page_dirty(2));

        let runs = dirty.dirty_runs();
        assert_eq!(runs.len(), 2);
        assert_eq!(runs[0].start_page(), 0);
        assert_eq!(runs[0].page_count, 2); // page 0, 1 合并
        assert_eq!(runs[1].first_page, 6);
        assert_eq!(runs[1].page_count, 1);
        assert_eq!(dirty.coalesced_runs().as_slice(), runs.as_slice());
    }

    #[test]
    fn test_page_dirty_bitmap_all_dirty_and_word_skip() {
        let all = PageDirtyBitmap::new_all_dirty(130);
        assert_eq!(all.dirty_page_count(), 130);
        let runs = all.dirty_runs();
        assert_eq!(runs.len(), 1);
        assert_eq!(runs[0].page_count, 130);

        let mut sparse = PageDirtyBitmap::new_clean_with_page_size(200, GRACE_HOPPER_64K_PAGE_SIZE);
        sparse.mark_dirty(3);
        sparse.mark_dirty(129);
        sparse.mark_dirty(130);
        sparse.mark_dirty(199);
        sparse.mark_dirty(500); // 越界忽略
        let runs = sparse.dirty_runs();
        assert_eq!(runs.len(), 3);
        assert_eq!((runs[1].first_page, runs[1].page_count), (129, 2));
        assert_eq!(runs[2].byte_offset, 199 * GRACE_HOPPER_64K_PAGE_SIZE);
        assert_eq!(sparse.total_dirty_bytes(), 4 * GRACE_HOPPER_64K_PAGE_SIZE);
    }
}
