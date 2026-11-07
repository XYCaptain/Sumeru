//! 16KB 硬件页对齐 GPU 零拷贝视图、脏页位图与统一内存抽象 (Sumeru GPU Compute Layer)
//!
//! # 架构概览
//!
//! 1. **M0 — 16KB 硬件页对齐分配器 (`PageAlignedBuffer<T>`)**：
//!    物理列直接以 `16384` 字节对齐并在尾部补齐至 `16384` 字节整数倍，零填充尾页以防跨行越界读取。
//! 2. **M1 — 零拷贝 GPU 物理页视图与 16KB 脏页位图 (`ColumnGpuView` / `PageDirtyBitmap`)**：
//!    将 64 行物理修改纪元 `phys_chunk_epochs`（覆盖 `swap_remove` 尾行回填）折叠映射为 16KB 硬件页级脏位图与合并连续传输段 `DirtyPageRun`。
//! 3. **M2 — Apple Metal UMA 零拷贝与双后端管线 (`metal`)**：
//!    直接将 16KB 页对齐指针封装为 `MTLBuffer (StorageModeShared, NoCopy)`，按 `(ptr, bytes, generation)` 缓存句柄，并计算最优 Threadgroup 网格。
//! 4. **M3 — 多平台着色器内核与三层遍历体系 (`kernel`)**：
//!    通过 `#[gpu]` 统一宏同时生成 Metal Shading Language (MSL)、WebGPU (WGSL) 与 CPU 算子，支持：
//!    - `.for_each(...)`（CPU 单线程顺序执行）
//!    - `.par_for_each(...)` / `.par().for_each(...)`（CPU 多核 Rayon 并行执行）
//!    - `.gpu().for_each(...)`（GPU 硬件 16KB 页对齐零拷贝派发）

pub mod ir;
pub mod kernel;
pub mod metal;

#[cfg(test)]
mod tests;

pub use ir::*;
pub use kernel::*;
pub use metal::*;

use crate::context::ModelComponent;
use crate::error::StateError;
use crate::id::{ArchetypeId, ComponentTypeId};
use crate::storage::column::page_aligned::HARDWARE_PAGE_SIZE;
use crate::storage::column::ColumnBlock;
use crate::storage::store::Store;
use bytemuck::{Pod, Zeroable};
use serde::de::DeserializeOwned;
use serde::Serialize;
use smallvec::SmallVec;
use std::marker::PhantomData;
use std::ops::Range;
use std::ptr::NonNull;
use std::sync::Arc;

/// 可直接映射至 GPU 缓冲区的 `#[repr(C)]` POD 组件契约。
///
/// # Safety
/// 实现类型必须为无填充未初始化字节的 `Pod + Zeroable` 平坦结构，且大小 > 0、对齐不超过 16KB。
pub unsafe trait GpuComponent:
    crate::storage::column::ArchiveComponent
    + ModelComponent
    + Copy
    + Serialize
    + DeserializeOwned
    + Pod
    + Zeroable
    + Send
    + Sync
    + 'static
{
    /// 编译期断言组件满足 GPU 页对齐要求。
    const IS_GPU_SAFE: bool = {
        assert!(
            std::mem::size_of::<Self>() > 0,
            "GpuComponent cannot be a zero-sized type (ZST)"
        );
        assert!(
            std::mem::align_of::<Self>() <= HARDWARE_PAGE_SIZE,
            "GpuComponent alignment exceeds 16KB hardware page size"
        );
        true
    };

    /// 触发编译期布局校验。
    #[inline(always)]
    fn assert_gpu_layout() {
        let _ = Self::IS_GPU_SAFE;
    }
}

/// 列的 16KB 硬件页几何信息。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PageGeometry {
    /// 单个组件元素的字节大小 (`size_of::<T>()`)
    pub elem_size: usize,
    /// 当前活跃元素行数
    pub elem_count: usize,
    /// 活跃数据占用的精确字节数 (`elem_count * elem_size`)
    pub active_bytes: usize,
    /// 底层 16KB 页对齐分配的总字节数（16384 的整数倍）
    pub allocated_bytes: usize,
    /// 活跃数据覆盖的 16KB 硬件页数
    pub page_count: usize,
    /// 底层已分配的 16KB 硬件页总数
    pub allocated_page_count: usize,
}

impl PageGeometry {
    /// 计算给定元素大小、活跃行数与分配字节数的 16KB 页几何描述。
    #[inline]
    pub fn compute(elem_size: usize, elem_count: usize, allocated_bytes: usize) -> Self {
        let active_bytes = elem_count.saturating_mul(elem_size);
        let page_count = if active_bytes == 0 {
            0
        } else {
            (active_bytes + HARDWARE_PAGE_SIZE - 1) / HARDWARE_PAGE_SIZE
        };
        let allocated_page_count = allocated_bytes / HARDWARE_PAGE_SIZE;
        Self {
            elem_size,
            elem_count,
            active_bytes,
            allocated_bytes,
            page_count,
            allocated_page_count,
        }
    }

    /// 单个 16KB 页可完整容纳的元素个数（若元素跨页则至少返回 1）。
    #[inline]
    pub fn elems_per_page(&self) -> usize {
        if self.elem_size == 0 {
            0
        } else {
            (HARDWARE_PAGE_SIZE / self.elem_size).max(1)
        }
    }

    /// 给定行索引所在的首个 16KB 页索引。
    #[inline]
    pub fn page_for_row(&self, row: usize) -> usize {
        (row.saturating_mul(self.elem_size)) / HARDWARE_PAGE_SIZE
    }

    /// 给定 16KB 页索引覆盖的活跃行索引范围 `start..end`（按行起始字节归属，跨页不重复计数）。
    #[inline]
    pub fn row_range_for_page(&self, page_idx: usize) -> Range<usize> {
        if self.elem_size == 0 || self.elem_count == 0 || page_idx >= self.page_count {
            return 0..0;
        }
        let page_start_byte = page_idx * HARDWARE_PAGE_SIZE;
        let page_end_byte = ((page_idx + 1) * HARDWARE_PAGE_SIZE).min(self.active_bytes);
        let start_row = ((page_start_byte + self.elem_size - 1) / self.elem_size).min(self.elem_count);
        let end_row = ((page_end_byte + self.elem_size - 1) / self.elem_size).min(self.elem_count);
        start_row..end_row.max(start_row)
    }
}

/// 合并后的连续 16KB 脏页区间（用于独显 PCIe DMA `Queue::write_buffer` 最小化分段传输）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DirtyPageRun {
    /// 起始 16KB 页号
    pub first_page: usize,
    /// 连续脏页数量
    pub page_count: usize,
    /// 相对于列缓冲区起始地址的字节偏移（必为 16384 的整数倍）
    pub byte_offset: usize,
    /// 连续脏页覆盖的对齐字节长度（必为 16384 的整数倍）
    pub byte_len: usize,
}

impl DirtyPageRun {
    /// 该连续脏页段覆盖的活跃元素行范围。
    #[inline]
    pub fn row_range(&self, elem_size: usize, total_rows: usize) -> Range<usize> {
        if elem_size == 0 || total_rows == 0 {
            return 0..0;
        }
        let start_row = (self.byte_offset / elem_size).min(total_rows);
        let end_byte = self.byte_offset.saturating_add(self.byte_len);
        let end_row = ((end_byte + elem_size - 1) / elem_size).min(total_rows);
        start_row..end_row
    }
}

/// 16KB 硬件页级脏位图（由 Archetype 的 64 行物理块纪元 `phys_chunk_epochs` 映射生成）。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PageDirtyBitmap {
    words: SmallVec<[u64; 2]>,
    page_count: usize,
}

impl PageDirtyBitmap {
    /// 创建具有指定页数的全净位图。
    #[inline]
    pub fn new_clean(page_count: usize) -> Self {
        let word_count = (page_count + 63) >> 6;
        let mut words = SmallVec::with_capacity(word_count);
        words.resize(word_count, 0);
        Self { words, page_count }
    }

    /// 创建具有指定页数的全脏位图。
    #[inline]
    pub fn new_all_dirty(page_count: usize) -> Self {
        let word_count = (page_count + 63) >> 6;
        let mut words = SmallVec::with_capacity(word_count);
        words.resize(word_count, !0u64);
        if page_count % 64 != 0 {
            if let Some(last) = words.last_mut() {
                *last &= (1u64 << (page_count % 64)) - 1;
            }
        }
        Self { words, page_count }
    }

    /// 从 64 行粒度的物理修改纪元数组 (`phys_chunk_epochs`) 构建 16KB 硬件页级脏位图。
    ///
    /// 任何 `chunk_epoch > since_epoch` 的 64 行块，其字节跨度 `[chunk_start_byte, chunk_end_byte)`
    /// 触及的所有 16KB 硬件页均被标记为脏页。
    pub fn from_phys_chunk_epochs(
        elem_size: usize,
        elem_count: usize,
        allocated_bytes: usize,
        chunk_epochs: Option<&[u64]>,
        default_epoch: u64,
        since_epoch: u64,
    ) -> Self {
        let geom = PageGeometry::compute(elem_size, elem_count, allocated_bytes);
        if geom.page_count == 0 {
            return Self::new_clean(0);
        }
        let Some(epochs) = chunk_epochs else {
            return if default_epoch > since_epoch {
                Self::new_all_dirty(geom.page_count)
            } else {
                Self::new_clean(geom.page_count)
            };
        };

        let mut bitmap = Self::new_clean(geom.page_count);
        let num_chunks = (elem_count + 63) >> 6;
        for chunk_idx in 0..num_chunks {
            let ep = epochs.get(chunk_idx).copied().unwrap_or(default_epoch);
            if ep > since_epoch {
                let start_row = chunk_idx << 6;
                let end_row = ((chunk_idx + 1) << 6).min(elem_count);
                if start_row < end_row {
                    let start_byte = start_row * elem_size;
                    let end_byte = end_row * elem_size;
                    let first_page = start_byte / HARDWARE_PAGE_SIZE;
                    let last_page = (end_byte.saturating_sub(1)) / HARDWARE_PAGE_SIZE;
                    for page in first_page..=last_page.min(geom.page_count.saturating_sub(1)) {
                        bitmap.mark_dirty(page);
                    }
                }
            }
        }
        bitmap
    }

    /// 标记指定 16KB 页为脏。
    #[inline]
    pub fn mark_dirty(&mut self, page_idx: usize) {
        if page_idx < self.page_count {
            self.words[page_idx >> 6] |= 1u64 << (page_idx & 63);
        }
    }

    /// 检查指定 16KB 页是否为脏。
    #[inline]
    pub fn is_page_dirty(&self, page_idx: usize) -> bool {
        if page_idx >= self.page_count {
            return false;
        }
        (self.words[page_idx >> 6] & (1u64 << (page_idx & 63))) != 0
    }

    /// 总 16KB 页数。
    #[inline]
    pub fn page_count(&self) -> usize {
        self.page_count
    }

    /// 脏 16KB 页总数。
    #[inline]
    pub fn dirty_page_count(&self) -> usize {
        self.words.iter().map(|w| w.count_ones() as usize).sum()
    }

    /// 是否存在至少一个脏页。
    #[inline]
    pub fn has_any_dirty(&self) -> bool {
        self.words.iter().any(|&w| w != 0)
    }

    /// 底层 `u64` 位图字切片。
    #[inline]
    pub fn words(&self) -> &[u64] {
        &self.words
    }

    /// 将相邻脏页合并为连续的 `DirtyPageRun` 列表，减少 DMA 拷贝调用次数。
    pub fn coalesced_runs(&self) -> SmallVec<[DirtyPageRun; 4]> {
        let mut runs = SmallVec::new();
        let mut page = 0;
        while page < self.page_count {
            if !self.is_page_dirty(page) {
                page += 1;
                continue;
            }
            let first_page = page;
            page += 1;
            while page < self.page_count && self.is_page_dirty(page) {
                page += 1;
            }
            let count = page - first_page;
            runs.push(DirtyPageRun {
                first_page,
                page_count: count,
                byte_offset: first_page * HARDWARE_PAGE_SIZE,
                byte_len: count * HARDWARE_PAGE_SIZE,
            });
        }
        runs
    }
}

/// 单列的只读 16KB 页对齐 GPU 视图。
#[derive(Debug, Clone)]
pub struct ColumnGpuView<'a, T: GpuComponent> {
    base_ptr: NonNull<u8>,
    allocated_bytes: usize,
    generation: u64,
    len: usize,
    geometry: PageGeometry,
    dirty_bitmap: PageDirtyBitmap,
    _marker: PhantomData<&'a [T]>,
}

impl<'a, T: GpuComponent> ColumnGpuView<'a, T> {
    /// 构造列的只读 GPU 视图。
    #[inline]
    pub(crate) fn new(
        base_ptr: NonNull<u8>,
        allocated_bytes: usize,
        generation: u64,
        len: usize,
        dirty_bitmap: PageDirtyBitmap,
    ) -> Self {
        T::assert_gpu_layout();
        let geometry = PageGeometry::compute(std::mem::size_of::<T>(), len, allocated_bytes);
        Self {
            base_ptr,
            allocated_bytes,
            generation,
            len,
            geometry,
            dirty_bitmap,
            _marker: PhantomData,
        }
    }

    /// 物理基地址指针（16KB 对齐）。
    #[inline]
    pub fn base_ptr(&self) -> *const u8 {
        self.base_ptr.as_ptr()
    }

    /// 检查底层缓冲区是否严格满足 16KB (`16384` bytes) 首地址与容量对齐（可直接用于 Metal `newBufferWithBytesNoCopy`）。
    #[inline]
    pub fn is_zero_copy_aligned(&self) -> bool {
        (self.base_ptr.as_ptr() as usize) % HARDWARE_PAGE_SIZE == 0
            && self.allocated_bytes % HARDWARE_PAGE_SIZE == 0
    }

    /// 底层 16KB 页对齐分配的总字节数。
    #[inline]
    pub fn allocated_bytes(&self) -> usize {
        self.allocated_bytes
    }

    /// 缓冲区重分配代际号（用于 `MTLBuffer` 缓存命中校验）。
    #[inline]
    pub fn generation(&self) -> u64 {
        self.generation
    }

    /// 活跃元素行数。
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    /// 列是否为空。
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// 16KB 页几何信息。
    #[inline]
    pub fn geometry(&self) -> PageGeometry {
        self.geometry
    }

    /// 16KB 硬件页级脏位图。
    #[inline]
    pub fn dirty_bitmap(&self) -> &PageDirtyBitmap {
        &self.dirty_bitmap
    }

    /// 活跃元素的强类型切片。
    #[inline]
    pub fn as_slice(&self) -> &'a [T] {
        if self.len == 0 {
            return &[];
        }
        unsafe { std::slice::from_raw_parts(self.base_ptr.as_ptr() as *const T, self.len) }
    }

    /// 完整 16KB 页对齐分配区的字节切片（含已零初始化的尾页安全填充区）。
    #[inline]
    pub fn as_padded_page_bytes(&self) -> &'a [u8] {
        if self.allocated_bytes == 0 {
            return &[];
        }
        unsafe { std::slice::from_raw_parts(self.base_ptr.as_ptr(), self.allocated_bytes) }
    }

    /// 获取指定 16KB 页的 `16384` 字节切片。
    #[inline]
    pub fn page_bytes(&self, page_idx: usize) -> Option<&'a [u8]> {
        if page_idx >= self.geometry.allocated_page_count {
            return None;
        }
        let offset = page_idx * HARDWARE_PAGE_SIZE;
        Some(&self.as_padded_page_bytes()[offset..offset + HARDWARE_PAGE_SIZE])
    }

    /// 导出合并后的连续脏页段及其对应的对齐字节切片（专用于独显增量上传）。
    pub fn dirty_runs(&self) -> SmallVec<[(DirtyPageRun, &'a [u8]); 4]> {
        let padded = self.as_padded_page_bytes();
        self.dirty_bitmap
            .coalesced_runs()
            .into_iter()
            .filter_map(|run| {
                let end = run.byte_offset.checked_add(run.byte_len)?;
                if end <= padded.len() {
                    Some((run, &padded[run.byte_offset..end]))
                } else {
                    None
                }
            })
            .collect()
    }
}

/// 单列的可变 16KB 页对齐 GPU 视图（用于 GPU Compute Shader 原位写回或 CPU/GPU 统一内存交互）。
#[derive(Debug)]
pub struct ColumnGpuMutView<'a, T: GpuComponent> {
    base_ptr: NonNull<u8>,
    allocated_bytes: usize,
    generation: u64,
    len: usize,
    geometry: PageGeometry,
    write_epoch: u64,
    _marker: PhantomData<&'a mut [T]>,
}

impl<'a, T: GpuComponent> ColumnGpuMutView<'a, T> {
    #[inline]
    pub(crate) fn new(
        base_ptr: NonNull<u8>,
        allocated_bytes: usize,
        generation: u64,
        len: usize,
        write_epoch: u64,
    ) -> Self {
        T::assert_gpu_layout();
        let geometry = PageGeometry::compute(std::mem::size_of::<T>(), len, allocated_bytes);
        Self {
            base_ptr,
            allocated_bytes,
            generation,
            len,
            geometry,
            write_epoch,
            _marker: PhantomData,
        }
    }

    /// 物理基地址可变指针（16KB 对齐）。
    #[inline]
    pub fn base_mut_ptr(&mut self) -> *mut u8 {
        self.base_ptr.as_ptr()
    }

    /// 检查底层缓冲区是否严格满足 16KB (`16384` bytes) 首地址与容量对齐。
    #[inline]
    pub fn is_zero_copy_aligned(&self) -> bool {
        (self.base_ptr.as_ptr() as usize) % HARDWARE_PAGE_SIZE == 0
            && self.allocated_bytes % HARDWARE_PAGE_SIZE == 0
    }

    /// 底层 16KB 页对齐分配的总字节数。
    #[inline]
    pub fn allocated_bytes(&self) -> usize {
        self.allocated_bytes
    }

    /// 缓冲区重分配代际号。
    #[inline]
    pub fn generation(&self) -> u64 {
        self.generation
    }

    /// 活跃元素行数。
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    /// 列是否为空。
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// 16KB 页几何信息。
    #[inline]
    pub fn geometry(&self) -> PageGeometry {
        self.geometry
    }

    /// 本次可变视图绑定的 Store 写入纪元。
    #[inline]
    pub fn write_epoch(&self) -> u64 {
        self.write_epoch
    }

    /// 活跃元素的强类型只读切片。
    #[inline]
    pub fn as_slice(&self) -> &[T] {
        if self.len == 0 {
            return &[];
        }
        unsafe { std::slice::from_raw_parts(self.base_ptr.as_ptr() as *const T, self.len) }
    }

    /// 活跃元素的强类型可变切片。
    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        if self.len == 0 {
            return &mut [];
        }
        unsafe { std::slice::from_raw_parts_mut(self.base_ptr.as_ptr() as *mut T, self.len) }
    }

    /// 完整 16KB 页对齐分配区的可变字节切片。
    #[inline]
    pub fn as_padded_page_bytes_mut(&mut self) -> &mut [u8] {
        if self.allocated_bytes == 0 {
            return &mut [];
        }
        unsafe { std::slice::from_raw_parts_mut(self.base_ptr.as_ptr(), self.allocated_bytes) }
    }
}

/// 异步 GPU 派发期间固定底层 16KB 页对齐内存块生命周期的零拷贝句柄。
///
/// 通过持有 `Arc<ColumnBlock<T>>` 引用计数：
/// 1. 即使在 GPU 异步执行期间 CPU 端对 `Store` 发起写入，COW 机制也会为 CPU 克隆新块，
///    而 GPU 正在读取的物理内存页保持绝对不可变与有效，杜绝悬垂指针（Use-After-Free）与数据竞争。
#[derive(Debug, Clone)]
pub struct GpuColumnPin<T: GpuComponent> {
    archetype_id: ArchetypeId,
    block: Arc<ColumnBlock<T>>,
    base_ptr: *const u8,
    allocated_bytes: usize,
    generation: u64,
    len: usize,
}

// SAFETY: `T: GpuComponent` 约束了 `Send + Sync`，且 `Arc<ColumnBlock<T>>` 在ピン留期间只读共享。
unsafe impl<T: GpuComponent> Send for GpuColumnPin<T> {}
unsafe impl<T: GpuComponent> Sync for GpuColumnPin<T> {}

impl<T: GpuComponent> GpuColumnPin<T> {
    /// 所属 Archetype ID。
    #[inline]
    pub fn archetype_id(&self) -> ArchetypeId {
        self.archetype_id
    }

    /// 16KB 页对齐基地址指针。
    #[inline]
    pub fn base_ptr(&self) -> *const u8 {
        self.base_ptr
    }

    /// 16KB 页对齐分配字节数。
    #[inline]
    pub fn allocated_bytes(&self) -> usize {
        self.allocated_bytes
    }

    /// 物理分配代际号。
    #[inline]
    pub fn generation(&self) -> u64 {
        self.generation
    }

    /// 活跃元素数量。
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    /// 是否为空。
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// 强类型元素只读切片。
    #[inline]
    pub fn as_slice(&self) -> &[T] {
        let _ = &self.block;
        if self.len == 0 {
            return &[];
        }
        unsafe { std::slice::from_raw_parts(self.base_ptr as *const T, self.len) }
    }
}

/// 跨 Archetype 16KB 虚拟页表项（供 Compute Shader 在单次 Dispatch 中跨多张原型表寻址）。
#[repr(C)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Pod, Zeroable)]
pub struct GpuPageDescriptor {
    /// 所属 Archetype ID
    pub archetype_id: u32,
    /// 该页在对应 Archetype 列缓冲区内的页偏移（单位：16KB 页）
    pub column_page_index: u32,
    /// 该页起始行在全局逻辑线程空间中的起始索引
    pub global_row_start: u32,
    /// 该页内的有效活跃行数
    pub active_row_count: u32,
}

/// 跨 Archetype 汇总的 GPU 16KB 物理页表。
#[derive(Debug, Clone, Default)]
pub struct GpuPageTable {
    descriptors: Vec<GpuPageDescriptor>,
    total_active_rows: u32,
}

impl GpuPageTable {
    /// 页描述符切片（可直接作为 Uniform / Storage Buffer 传入 Compute Shader）。
    #[inline]
    pub fn descriptors(&self) -> &[GpuPageDescriptor] {
        &self.descriptors
    }

    /// 汇总的全局有效行总数。
    #[inline]
    pub fn total_active_rows(&self) -> u32 {
        self.total_active_rows
    }

    /// 汇总的 16KB 物理页总数。
    #[inline]
    pub fn total_pages(&self) -> usize {
        self.descriptors.len()
    }
}

/// 支持批量注册 GPU 查询中涉及的所有组件为 16KB 页对齐模式的辅助 Trait。
pub trait RegisterGpuQuery {
    /// 将查询涉及的所有组件类型注册到 `Store` 并提升为 16KB 硬件页对齐。
    fn register_in_store(store: &mut Store) -> Result<(), StateError>;
}

impl<A: GpuComponent> RegisterGpuQuery for &A {
    #[inline]
    fn register_in_store(store: &mut Store) -> Result<(), StateError> {
        store.register_gpu_component::<A>()
    }
}

impl<A: GpuComponent> RegisterGpuQuery for &mut A {
    #[inline]
    fn register_in_store(store: &mut Store) -> Result<(), StateError> {
        store.register_gpu_component::<A>()
    }
}

impl<A: RegisterGpuQuery> RegisterGpuQuery for (A,) {
    #[inline]
    fn register_in_store(store: &mut Store) -> Result<(), StateError> {
        A::register_in_store(store)
    }
}

impl<A: RegisterGpuQuery, B: RegisterGpuQuery> RegisterGpuQuery for (A, B) {
    #[inline]
    fn register_in_store(store: &mut Store) -> Result<(), StateError> {
        A::register_in_store(store)?;
        B::register_in_store(store)
    }
}

impl<A: RegisterGpuQuery, B: RegisterGpuQuery, C: RegisterGpuQuery> RegisterGpuQuery for (A, B, C) {
    #[inline]
    fn register_in_store(store: &mut Store) -> Result<(), StateError> {
        A::register_in_store(store)?;
        B::register_in_store(store)?;
        C::register_in_store(store)
    }
}

impl<A: RegisterGpuQuery, B: RegisterGpuQuery, C: RegisterGpuQuery, D: RegisterGpuQuery>
    RegisterGpuQuery for (A, B, C, D)
{
    #[inline]
    fn register_in_store(store: &mut Store) -> Result<(), StateError> {
        A::register_in_store(store)?;
        B::register_in_store(store)?;
        C::register_in_store(store)?;
        D::register_in_store(store)
    }
}

impl Store {
    /// 注册 GPU 友好的 `#[repr(C)]` POD 组件（统一采用 `ColumnCodecId::Rkyv` 归档），并将其在所有原型中的物理列提升为 16KB (`16384` bytes) 页对齐缓冲区。
    pub fn register_gpu_component<T: GpuComponent>(&mut self) -> Result<(), StateError> {
        T::assert_gpu_layout();
        self.check_poisoned()?;
        let mut reg = crate::storage::column::ComponentRegistration::of::<T>();
        reg.column.promote_to_page_aligned();
        self.register_component_prototypes(vec![reg.column])
    }

    /// 一次性注册查询元组 `Q`（如 `(&mut Pos, &Vel)`）涉及的所有 GPU 组件并开启 16KB 页对齐。
    #[inline]
    pub fn register_gpu_query<Q: RegisterGpuQuery>(&mut self) -> Result<(), StateError> {
        Q::register_in_store(self)
    }

    /// 提取所有包含组件 `T` 的非空 Archetype 的只读 16KB 页对齐 GPU 视图与自 `since_epoch` 以来的 16KB 脏页位图。
    pub fn gpu_column_views<T: GpuComponent>(
        &self,
        since_epoch: u64,
    ) -> Result<Vec<(ArchetypeId, ColumnGpuView<'_, T>)>, StateError> {
        T::assert_gpu_layout();
        self.check_poisoned()?;
        let cid = ComponentTypeId::of::<T>();
        let elem_size = std::mem::size_of::<T>();
        let mut out = Vec::new();
        for arch in &self.archetypes {
            let len = arch.len();
            if len == 0 || !arch.component_types_slice().contains(&cid) {
                continue;
            }
            let Some(col) = arch.columns().get(&cid) else {
                return Err(StateError::ComponentNotFound(format!("{cid}")));
            };
            if col.len() != len {
                return Err(StateError::AccessDenied(format!(
                    "Component column {} length {} does not match archetype {} row count {}",
                    std::any::type_name::<T>(),
                    col.len(),
                    arch.id().0,
                    len
                )));
            }
            let Some((ptr, allocated_bytes)) = col.metal_raw_parts() else {
                return Err(StateError::AccessDenied(format!(
                    "Component column {} in archetype {} is not 16KB page-aligned; call store.register_gpu_component::<{}>() or use LayoutOptimization::PageAlignedGpu",
                    std::any::type_name::<T>(),
                    arch.id().0,
                    std::any::type_name::<T>(),
                )));
            };
            let active_bytes = len.checked_mul(elem_size).ok_or_else(|| {
                StateError::CapacityExhausted("GPU column active byte size overflow".into())
            })?;
            if active_bytes > allocated_bytes {
                return Err(StateError::AccessDenied(format!(
                    "Component column {} active_bytes ({active_bytes}) exceeds allocated_bytes ({allocated_bytes})",
                    std::any::type_name::<T>()
                )));
            }
            let generation = col.physical_version();
            let (merged_epochs, def_ep) = arch.effective_gpu_chunk_epochs(cid);
            let dirty = PageDirtyBitmap::from_phys_chunk_epochs(
                elem_size,
                len,
                allocated_bytes,
                if merged_epochs.is_empty() {
                    None
                } else {
                    Some(merged_epochs.as_slice())
                },
                def_ep,
                since_epoch,
            );
            let base_ptr = NonNull::new(ptr).expect("PageAlignedBuffer pointer is non-null");
            out.push((
                arch.id(),
                ColumnGpuView::new(base_ptr, allocated_bytes, generation, len, dirty),
            ));
        }
        Ok(out)
    }

    /// 提取所有包含组件 `T` 的非空 Archetype 的可变 16KB 页对齐 GPU 视图，强制执行 COW 独占分离并自动推进 Store 纪元与块纪元。
    pub fn gpu_column_views_mut<T: GpuComponent>(
        &mut self,
    ) -> Result<Vec<(ArchetypeId, ColumnGpuMutView<'_, T>)>, StateError> {
        T::assert_gpu_layout();
        self.check_poisoned()?;
        let next_epoch = self.next_epoch()?;
        let cid = ComponentTypeId::of::<T>();
        let elem_size = std::mem::size_of::<T>();

        // 阶段 1a：全表只读预校验（确保任意 Archetype 布局异常时绝不产生半边提升或 COW 剥离副作用）
        for arch in &self.archetypes {
            let len = arch.len();
            if len == 0 || !arch.component_types_slice().contains(&cid) {
                continue;
            }
            let arch_id = arch.id().0;
            let col = arch
                .columns()
                .get(&cid)
                .ok_or_else(|| StateError::ComponentNotFound(format!("{cid}")))?;
            if col.len() != len {
                return Err(StateError::AccessDenied(format!(
                    "Component column {} length {} does not match archetype {} row count {}",
                    std::any::type_name::<T>(),
                    col.len(),
                    arch_id,
                    len
                )));
            }
            let _ = len.checked_mul(elem_size).ok_or_else(|| {
                StateError::CapacityExhausted("GPU column active byte size overflow".into())
            })?;
        }

        // 阶段 1b：16KB 页对齐提升与 COW 独占分离（确保被 Pin / Clone 共享的块绝不被原位覆写）
        for arch in &mut self.archetypes {
            let len = arch.len();
            if len == 0 || !arch.component_types_slice().contains(&cid) {
                continue;
            }
            let arch_id = arch.id().0;
            let col = arch
                .get_erased_column_mut(cid)
                .ok_or_else(|| StateError::ComponentNotFound(format!("{cid}")))?;
            if !col.is_page_aligned() {
                col.promote_to_page_aligned();
            }
            col.ensure_contiguous()?;
            let Some((_ptr, allocated_bytes)) = col.metal_raw_parts() else {
                return Err(StateError::AccessDenied(format!(
                    "Component column {} in archetype {} failed to promote to 16KB page-aligned buffer",
                    std::any::type_name::<T>(),
                    arch_id
                )));
            };
            let active_bytes = len * elem_size;
            if active_bytes > allocated_bytes {
                return Err(StateError::AccessDenied(format!(
                    "Component column {} active_bytes ({active_bytes}) exceeds allocated_bytes ({allocated_bytes})",
                    std::any::type_name::<T>()
                )));
            }
            arch.rebuild_flat_columns();
        }

        // 阶段 2：原子推进纪元并导出独占可变视图
        self.epoch = next_epoch;
        let mut out = Vec::new();
        for arch in &mut self.archetypes {
            let len = arch.len();
            if len == 0 || !arch.component_types_slice().contains(&cid) {
                continue;
            }
            arch.mark_column_row_range_epoch(cid, 0..len, next_epoch);
            let col = arch
                .columns()
                .get(&cid)
                .expect("checked component presence");
            let (ptr, allocated_bytes) = col
                .metal_raw_parts()
                .expect("column promoted and COW-unfrozen to 16KB page-aligned");
            let generation = col.physical_version();
            let base_ptr = NonNull::new(ptr).expect("PageAlignedBuffer pointer is non-null");
            out.push((
                arch.id(),
                ColumnGpuMutView::new(base_ptr, allocated_bytes, generation, len, next_epoch),
            ));
        }
        Ok(out)
    }

    /// 为所有包含组件 `T` 的非空 Archetype 创建 COW 安全的零拷贝 `GpuColumnPin<T>`，供异步 GPU 命令缓冲区安全读取。
    pub fn pin_gpu_columns<T: GpuComponent>(&mut self) -> Result<Vec<GpuColumnPin<T>>, StateError> {
        T::assert_gpu_layout();
        self.check_poisoned()?;
        let cid = ComponentTypeId::of::<T>();
        let elem_size = std::mem::size_of::<T>();

        // 阶段 1a：全表只读预校验（确保任意 Archetype 布局异常时绝不产生半边提升或冻结副作用）
        for arch in &self.archetypes {
            if arch.is_empty() || !arch.component_types_slice().contains(&cid) {
                continue;
            }
            let arch_id = arch.id();
            let arch_len = arch.len();
            let typed = arch
                .get_column::<T>()
                .ok_or_else(|| StateError::ComponentNotFound(format!("{cid}")))?;
            if typed.len() != arch_len {
                return Err(StateError::AccessDenied(format!(
                    "Component column {} length {} does not match archetype {} row count {}",
                    std::any::type_name::<T>(),
                    typed.len(),
                    arch_id.0,
                    arch_len
                )));
            }
            let _ = arch_len.checked_mul(elem_size).ok_or_else(|| {
                StateError::CapacityExhausted("GPU column active byte size overflow".into())
            })?;
        }

        // 阶段 1b：提升为 16KB 页对齐并导出 COW 零拷贝 Pin
        let mut pins = Vec::new();
        for arch in &mut self.archetypes {
            if arch.is_empty() || !arch.component_types_slice().contains(&cid) {
                continue;
            }
            let mut promoted = false;
            if let Some(col) = arch.get_erased_column_mut(cid) {
                if !col.is_page_aligned() {
                    col.promote_to_page_aligned();
                    promoted = true;
                }
            }
            if promoted {
                arch.rebuild_flat_columns();
            }
            let arch_id = arch.id();
            let arch_len = arch.len();
            let typed = arch
                .get_column_mut::<T>()
                .ok_or_else(|| StateError::ComponentNotFound(format!("{cid}")))?;
            let (base_ptr, allocated_bytes) = typed
                .as_metal_raw_parts()
                .expect("column promoted to 16KB page-aligned");
            let active_bytes = arch_len * elem_size;
            if active_bytes > allocated_bytes {
                return Err(StateError::AccessDenied(format!(
                    "Component column {} active_bytes ({active_bytes}) exceeds allocated_bytes ({allocated_bytes})",
                    std::any::type_name::<T>()
                )));
            }
            let generation = typed.physical_version();
            let block = typed.share_root_block();
            pins.push(GpuColumnPin {
                archetype_id: arch_id,
                block,
                base_ptr,
                allocated_bytes,
                generation,
                len: arch_len,
            });
        }
        Ok(pins)
    }

    /// 构建组件 `T` 跨所有匹配 Archetype 的 16KB 硬件页表（供单次 GPU Dispatch 跨表寻址）。
    pub fn build_gpu_page_table<T: GpuComponent>(&self) -> Result<GpuPageTable, StateError> {
        let views = self.gpu_column_views::<T>(0)?;
        let mut descriptors = Vec::new();
        let mut global_row_start = 0u32;
        for (arch_id, view) in views {
            let geom = view.geometry();
            for page_idx in 0..geom.page_count {
                let range = geom.row_range_for_page(page_idx);
                let active_row_count = (range.end - range.start) as u32;
                descriptors.push(GpuPageDescriptor {
                    archetype_id: arch_id.0,
                    column_page_index: page_idx as u32,
                    global_row_start,
                    active_row_count,
                });
                global_row_start = global_row_start.saturating_add(active_row_count);
            }
        }
        Ok(GpuPageTable {
            descriptors,
            total_active_rows: global_row_start,
        })
    }
}
