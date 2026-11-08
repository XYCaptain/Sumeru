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

/// Apple Silicon M 系列 GPU / Linux 64 位系统标准硬件物理页粒度（16KB = 16384 字节）
pub const HARDWARE_PAGE_SIZE: usize = 16384;

/// NVIDIA Grace Hopper (GH200) NVLink-C2C / ARM64 超算节点 64KB 系统页粒度
pub const GRACE_HOPPER_64K_PAGE_SIZE: usize = 65536;

/// 连续硬件脏页段描述符（由 64 行物理块纪元折叠合并生成，直接作为 PCIe / RDMA 增量 DMA 传输段）
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DirtyPageRun {
    /// 起始硬件页号（0-indexed）
    pub start_page: usize,
    /// 连续脏页数量
    pub page_count: usize,
    /// 在列缓冲区内的起始字节偏移（`start_page * page_size`）
    pub byte_offset: usize,
    /// 本次连续传输的字节跨度（向上对齐至硬件页边界或尾页有效载荷）
    pub byte_len: usize,
}

/// 硬件页级脏位图与合并段清单 (`PageDirtyBitmap`)
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PageDirtyBitmap {
    page_size: usize,
    total_pages: usize,
    dirty_page_count: usize,
    page_bits: smallvec::SmallVec<[u64; 2]>,
    runs: Vec<DirtyPageRun>,
}

impl PageDirtyBitmap {
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

    /// 按指定硬件页粒度（如 4KB、16KB 或 64KB），将 64 行物理块纪元精确折叠为连续硬件脏页段 [`DirtyPageRun`]
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
            return Self {
                page_size,
                total_pages: 0,
                dirty_page_count: 0,
                page_bits: smallvec::SmallVec::new(),
                runs: Vec::new(),
            };
        }

        let total_pages = total_bytes.div_ceil(page_size);
        let num_words = total_pages.div_ceil(64);
        let mut page_bits = smallvec::smallvec![0u64; num_words];

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
                    page_bits[p >> 6] |= 1u64 << (p & 63);
                }
            }
        }

        let mut runs = Vec::new();
        let mut dirty_page_count = 0usize;
        let mut cur_start: Option<usize> = None;

        for p in 0..total_pages {
            let is_dirty = (page_bits[p >> 6] & (1u64 << (p & 63))) != 0;
            if is_dirty {
                dirty_page_count += 1;
                if cur_start.is_none() {
                    cur_start = Some(p);
                }
            } else if let Some(start) = cur_start.take() {
                let count = p - start;
                runs.push(DirtyPageRun {
                    start_page: start,
                    page_count: count,
                    byte_offset: start * page_size,
                    byte_len: count * page_size,
                });
            }
        }

        if let Some(start) = cur_start.take() {
            let count = total_pages - start;
            runs.push(DirtyPageRun {
                start_page: start,
                page_count: count,
                byte_offset: start * page_size,
                byte_len: count * page_size,
            });
        }

        Self {
            page_size,
            total_pages,
            dirty_page_count,
            page_bits,
            runs,
        }
    }

    #[inline]
    pub fn page_size(&self) -> usize {
        self.page_size
    }

    #[inline]
    pub fn total_pages(&self) -> usize {
        self.total_pages
    }

    #[inline]
    pub fn dirty_page_count(&self) -> usize {
        self.dirty_page_count
    }

    #[inline]
    pub fn dirty_runs(&self) -> &[DirtyPageRun] {
        &self.runs
    }

    #[inline]
    pub fn total_dirty_bytes(&self) -> usize {
        self.dirty_page_count * self.page_size
    }

    #[inline]
    pub fn is_page_dirty(&self, page_idx: usize) -> bool {
        if page_idx >= self.total_pages {
            return false;
        }
        (self.page_bits[page_idx >> 6] & (1u64 << (page_idx & 63))) != 0
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
        assert_eq!(runs[0].start_page, 0);
        assert_eq!(runs[0].page_count, 2); // page 0, 1 合并
        assert_eq!(runs[1].start_page, 6);
        assert_eq!(runs[1].page_count, 1);
    }
}
