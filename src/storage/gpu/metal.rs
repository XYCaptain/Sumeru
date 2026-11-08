//! Apple Metal UMA 零拷贝描述符与关系目标 GPU 查找表 (Metal UMA & PairTargetGpuLut)
//!
//! 核心职责：
//! 1. 封装 16KB 硬件页对齐（`HARDWARE_PAGE_SIZE = 16384`）且尾页零填充的 UMA 零拷贝显存描述符
//!    [`MetalNoCopyDescriptor`]，直接对接 Metal `newBufferWithBytesNoCopy`；
//! 2. 将 `RowBitMask` / `RowFlags` 行级位平面直接包装为 16KB 页对齐的 GPU 谓词掩码缓冲区；
//! 3. 提供 [`PairTargetGpuLut`]：将关系对 `Pair<R, EntityId>` 的目标实体 ID 在 CPU 侧预解析为
//!    目标原型表内的密集行号 `u32`（`u32::MAX` 代表无效或跨表未命中），使 GPU Kernel 能够以
//!    $O(1)$ 连续偏移直接执行跨行 Gather 与 Atomic Scatter-Add，无需在 GPU 侧维护哈希表。

use crate::storage::core::archetype::RowBitMask;
use crate::storage::core::column::PageAlignedBuffer;
use crate::storage::gpu::HARDWARE_PAGE_SIZE;

/// 专供 Apple Metal `newBufferWithBytesNoCopy` 或统一内存（UMA）映射使用的物理页描述符
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MetalNoCopyDescriptor {
    /// 16KB 硬件页对齐的首地址指针（`base_ptr as usize % 16384 == 0`）
    pub base_ptr: *mut u8,
    /// 向上圆整为 16KB 整数倍的物理分配字节数（`allocated_bytes % 16384 == 0`）
    pub allocated_bytes: usize,
    /// 有效载荷的实际字节数（`logical_len * elem_size`）
    pub used_bytes: usize,
    /// 逻辑元素数量
    pub logical_len: usize,
    /// 单个元素字节大小
    pub elem_size: usize,
}

// SAFETY: 描述符仅携带物理地址与长度元数据，跨线程传递由上层借用或 Pin 守卫保障。
unsafe impl Send for MetalNoCopyDescriptor {}
unsafe impl Sync for MetalNoCopyDescriptor {}

impl MetalNoCopyDescriptor {
    /// 从 `PageAlignedBuffer<T>` 提取零拷贝描述符，并校验 16KB 硬件页对齐契约
    #[inline]
    pub fn from_page_aligned_buffer<T>(buf: &PageAlignedBuffer<T>) -> Self {
        let (base_ptr, allocated_bytes) = buf.as_metal_raw_parts();
        let elem_size = std::mem::size_of::<T>();
        let logical_len = buf.len();
        let used_bytes = buf.used_bytes();
        if allocated_bytes > 0 {
            debug_assert_eq!(base_ptr as usize % HARDWARE_PAGE_SIZE, 0);
            debug_assert_eq!(allocated_bytes % HARDWARE_PAGE_SIZE, 0);
        }
        Self {
            base_ptr,
            allocated_bytes,
            used_bytes,
            logical_len,
            elem_size,
        }
    }

    /// 验证指针与分配长度是否严格满足 16KB 硬件页对齐
    #[inline]
    pub fn is_page_aligned(&self) -> bool {
        if self.allocated_bytes == 0 {
            return true;
        }
        (self.base_ptr as usize).is_multiple_of(HARDWARE_PAGE_SIZE)
            && self.allocated_bytes.is_multiple_of(HARDWARE_PAGE_SIZE)
    }

    /// 占用的 16KB 硬件物理页总数
    #[inline]
    pub fn page_count(&self) -> usize {
        self.allocated_bytes / HARDWARE_PAGE_SIZE
    }
}

/// 将 `RowBitMask`（来自 `RowFlags`、`PairTargetIndex` 或 `ChangedSince`）封装为 16KB 页对齐的 GPU 谓词缓冲区
#[derive(Debug, Clone)]
pub struct GpuPredicationMaskBuffer {
    words_buf: PageAlignedBuffer<u64>,
    row_len: usize,
    active_rows: usize,
}

impl GpuPredicationMaskBuffer {
    /// 从 `RowBitMask` 构建 16KB 页对齐且尾页清零的 GPU 位平面掩码缓冲区
    pub fn from_row_bit_mask(mask: &RowBitMask) -> Self {
        let words_buf = PageAlignedBuffer::from_slice_zero_padded(mask.words());
        let active_rows = mask.count_ones();
        Self {
            words_buf,
            row_len: mask.row_len(),
            active_rows,
        }
    }

    #[inline]
    pub fn row_len(&self) -> usize {
        self.row_len
    }

    #[inline]
    pub fn active_rows(&self) -> usize {
        self.active_rows
    }

    #[inline]
    pub fn words(&self) -> &[u64] {
        self.words_buf.as_slice()
    }

    #[inline]
    pub fn page_aligned_buffer(&self) -> &PageAlignedBuffer<u64> {
        &self.words_buf
    }

    #[inline]
    pub fn metal_descriptor(&self) -> MetalNoCopyDescriptor {
        MetalNoCopyDescriptor::from_page_aligned_buffer(&self.words_buf)
    }
}

/// 关系对 `Pair<R, Target>` 的 GPU 密集目标行索引查找表（LUT）
///
/// 将源原型表中每一行的 `Pair<R, EntityId>.target` 映射为目标原型表内的连续行号 `u32`。
/// 若目标实体不存在或不属于指定目标原型表，则对应槽位记为 `u32::MAX`（`INVALID_TARGET_ROW`）。
#[derive(Debug, Clone)]
pub struct PairTargetGpuLut {
    pub source_archetype_id: u32,
    pub target_archetype_id: u32,
    pub target_row_count: usize,
    pub valid_link_count: usize,
    pub target_rows: PageAlignedBuffer<u32>,
}

impl PairTargetGpuLut {
    /// 无效目标行哨兵值（在 WGSL / MSL / CUDA 中通过 `target_row < target_row_count` 自动拦截）
    pub const INVALID_TARGET_ROW: u32 = u32::MAX;

    /// 从预解析的目标行切片构造 16KB 页对齐且尾页零填充的查找表
    pub fn from_target_rows(
        source_archetype_id: u32,
        target_archetype_id: u32,
        target_row_count: usize,
        rows: &[u32],
    ) -> Self {
        let mut buf = PageAlignedBuffer::with_capacity(rows.len());
        let mut valid_link_count = 0usize;
        for &r in rows {
            if (r as usize) < target_row_count {
                valid_link_count += 1;
            }
            buf.push(r);
        }
        buf.zero_pad_tail();
        Self {
            source_archetype_id,
            target_archetype_id,
            target_row_count,
            valid_link_count,
            target_rows: buf,
        }
    }

    #[inline]
    pub fn as_slice(&self) -> &[u32] {
        self.target_rows.as_slice()
    }

    #[inline]
    pub fn metal_descriptor(&self) -> MetalNoCopyDescriptor {
        MetalNoCopyDescriptor::from_page_aligned_buffer(&self.target_rows)
    }
}
