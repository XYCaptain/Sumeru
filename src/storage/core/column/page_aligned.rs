//! 专为 Apple Silicon UMA (16KB 页对齐) 与 CPU SIMD 硬件对齐设计的平坦连续缓冲区 (PageAlignedBuffer)
//!
//! 核心设计契约：
//! 1. 物理首地址严格对齐到 16KB (16,384 字节) 硬件页边界 (`ptr as usize % 16384 == 0`)；
//! 2. 物理分配的总字节容量始终向上圆整为 16KB 物理页的整数倍 (`allocated_bytes % 16384 == 0`)；
//! 3. 内部元素连续无间隙平铺（例如 16 字节的 `Position` 紧密排列，绝不在元素间填充空洞）；
//! 4. 完美兼容 Metal `newBufferWithBytesNoCopy` 零拷贝显存包装；
//! 5. 纯标准库实现，零第三方依赖，安全处理 ZST（零大小类型）与异常丢弃（Panic/Drop Safety）。

use std::alloc::{alloc, dealloc, realloc, Layout};
use std::fmt;
use std::ops::{Deref, DerefMut};
use std::ptr::NonNull;

/// 硬件物理页大小常数：Apple Silicon (ARM64) 默认采用 16KB 虚拟内存页
pub const PAGE_SIZE: usize = 16384;

/// 专为 Apple Silicon GPU 零拷贝与 CPU SIMD 硬件对齐设计的平坦连续缓冲区
pub struct PageAlignedBuffer<T> {
    ptr: NonNull<T>,
    len: usize,
    capacity: usize,
}

#[allow(dead_code)]
impl<T> PageAlignedBuffer<T> {
    /// 创建一个空的未分配缓冲区（0 字节开销，悬挂指针）
    #[inline]
    pub const fn new() -> Self {
        Self {
            ptr: NonNull::dangling(),
            len: 0,
            capacity: 0,
        }
    }

    /// 创建一个预分配至少 `capacity` 个元素空间的页对齐缓冲区
    pub fn with_capacity(capacity: usize) -> Self {
        let mut buf = Self::new();
        if capacity > 0 {
            buf.reserve(capacity);
        }
        buf
    }

    /// 当前有效元素个数
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    /// 缓冲区是否为空
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// 当前已分配的元素容量上限（在不触发 realloc 的情况下能容纳的最多元素数）
    #[inline]
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// 物理分配的真实字节总容量（保证为 16KB 的整数倍）
    #[inline]
    pub fn allocated_bytes(&self) -> usize {
        if std::mem::size_of::<T>() == 0 || self.capacity == 0 {
            0
        } else {
            let elem_size = std::mem::size_of::<T>();
            let raw_bytes = self.capacity.checked_mul(elem_size).expect("capacity overflow");
            (raw_bytes + PAGE_SIZE - 1) & !(PAGE_SIZE - 1)
        }
    }

    /// 确保至少有额外 `additional` 个元素的空闲物理容量
    pub fn reserve(&mut self, additional: usize) {
        let elem_size = std::mem::size_of::<T>();
        if elem_size == 0 {
            // ZST (零大小类型) 处理：无需物理分配，容量视为无限
            self.capacity = usize::MAX;
            return;
        }

        let min_cap = self.len.checked_add(additional).expect("capacity overflow");
        if min_cap <= self.capacity {
            return;
        }

        // 扩容策略：至少翻倍成长，且必须满足 min_cap
        let target_cap = std::cmp::max(self.capacity.saturating_mul(2), min_cap);
        let raw_bytes = target_cap.checked_mul(elem_size).expect("byte overflow");
        
        // 向上圆整为 16KB 页边界的整数倍
        let page_bytes = (raw_bytes + PAGE_SIZE - 1) & !(PAGE_SIZE - 1);
        let actual_cap = page_bytes / elem_size;

        let new_layout = Layout::from_size_align(page_bytes, PAGE_SIZE)
            .expect("Valid page-aligned layout");

        let new_ptr = if self.capacity == 0 {
            unsafe { alloc(new_layout) as *mut T }
        } else {
            let old_page_bytes = self.allocated_bytes();
            let old_layout = Layout::from_size_align(old_page_bytes, PAGE_SIZE).unwrap();
            unsafe { realloc(self.ptr.as_ptr() as *mut u8, old_layout, page_bytes) as *mut T }
        };

        self.ptr = NonNull::new(new_ptr).expect("Allocation failed");
        self.capacity = actual_cap;
    }

    /// 追加单个元素到末尾
    #[inline]
    pub fn push(&mut self, value: T) {
        if self.len == self.capacity {
            self.reserve(1);
        }
        unsafe {
            std::ptr::write(self.ptr.as_ptr().add(self.len), value);
            self.len += 1;
        }
    }

    /// 弹出末尾元素
    #[inline]
    pub fn pop(&mut self) -> Option<T> {
        if self.len == 0 {
            None
        } else {
            self.len -= 1;
            unsafe { Some(std::ptr::read(self.ptr.as_ptr().add(self.len))) }
        }
    }

    /// 以 O(1) 尾部互换方式删除指定下标元素（ECS 密集列存的核心删除算法）
    #[inline]
    pub fn swap_remove(&mut self, index: usize) -> Option<T> {
        if index >= self.len {
            return None;
        }
        unsafe {
            let last_idx = self.len - 1;
            let val = std::ptr::read(self.ptr.as_ptr().add(index));
            if index != last_idx {
                std::ptr::copy(self.ptr.as_ptr().add(last_idx), self.ptr.as_ptr().add(index), 1);
            }
            self.len -= 1;
            Some(val)
        }
    }

    /// 安全读取指定下标元素
    #[inline]
    pub fn get(&self, index: usize) -> Option<&T> {
        if index < self.len {
            unsafe { Some(&*self.ptr.as_ptr().add(index)) }
        } else {
            None
        }
    }

    /// 安全可变借用指定下标元素
    #[inline]
    pub fn get_mut(&mut self, index: usize) -> Option<&mut T> {
        if index < self.len {
            unsafe { Some(&mut *self.ptr.as_ptr().add(index)) }
        } else {
            None
        }
    }

    /// 原生只读切片透传（零开销）
    #[inline]
    pub fn as_slice(&self) -> &[T] {
        if self.len == 0 {
            &[]
        } else {
            unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
        }
    }

    /// 原生可变切片透传（直接供 Rayon / SIMD 直写）
    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        if self.len == 0 {
            &mut []
        } else {
            unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len) }
        }
    }

    /// 清空所有活跃元素，保留已分配物理容量
    pub fn clear(&mut self) {
        if self.len > 0 {
            unsafe {
                let slice = std::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len);
                self.len = 0;
                std::ptr::drop_in_place(slice);
            }
        }
    }

    /// 当前有效元素占用的实际字节数（`len * size_of::<T>()`）
    #[inline]
    pub fn used_bytes(&self) -> usize {
        self.len.saturating_mul(std::mem::size_of::<T>())
    }

    /// 尾页未使用的空闲填充字节数（`allocated_bytes() - used_bytes()`）
    #[inline]
    pub fn tail_padding_bytes(&self) -> usize {
        self.allocated_bytes().saturating_sub(self.used_bytes())
    }

    /// 将有效元素末尾至 16KB 物理页边界之间的尾页空闲字节全部清零。
    ///
    /// 在 UMA 零拷贝提交给 GPU Compute Kernel 前调用，确保 Threadgroup / SIMD 越界预取
    /// 读取到确定性的全 0 字节，避免未初始化内存读取。
    #[inline]
    pub fn zero_pad_tail(&mut self) {
        let total = self.allocated_bytes();
        let used = self.used_bytes();
        if total > used {
            unsafe {
                let tail_ptr = (self.ptr.as_ptr() as *mut u8).add(used);
                std::ptr::write_bytes(tail_ptr, 0, total - used);
            }
        }
    }

    /// 从连续切片构造 16KB 硬件页对齐且尾页零填充的缓冲区
    pub fn from_slice_zero_padded(slice: &[T]) -> Self
    where
        T: Clone,
    {
        let mut buf = Self::with_capacity(slice.len());
        for item in slice {
            buf.push(item.clone());
        }
        buf.zero_pad_tail();
        buf
    }

    /// 导出专供 Apple Metal `newBufferWithBytesNoCopy` 使用的原始指针与页整倍数字节长度
    ///
    /// # 契约保证
    /// 1. 返回的指针必定满足 `ptr as usize % 16384 == 0`；
    /// 2. 返回的字节长度必定满足 `allocated_bytes % 16384 == 0`。
    #[inline]
    pub fn as_metal_raw_parts(&self) -> (*mut u8, usize) {
        let bytes = self.allocated_bytes();
        let ptr = self.ptr.as_ptr() as *mut u8;
        if bytes > 0 {
            debug_assert_eq!(
                ptr as usize % PAGE_SIZE,
                0,
                "Pointer must be 16KB page-aligned for Metal"
            );
            debug_assert_eq!(
                bytes % PAGE_SIZE,
                0,
                "Length must be an integer multiple of 16KB for Metal"
            );
        }
        (ptr, bytes)
    }
}

impl<T> Deref for PageAlignedBuffer<T> {
    type Target = [T];
    #[inline]
    fn deref(&self) -> &Self::Target {
        self.as_slice()
    }
}

impl<T> DerefMut for PageAlignedBuffer<T> {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.as_mut_slice()
    }
}

impl<T> Drop for PageAlignedBuffer<T> {
    fn drop(&mut self) {
        if self.capacity > 0 && std::mem::size_of::<T>() > 0 {
            unsafe {
                // 1. 原位析构所有存活元素
                std::ptr::drop_in_place(self.as_mut_slice());
                // 2. 释放 16KB 对齐的整块物理内存
                let page_bytes = self.allocated_bytes();
                let layout = Layout::from_size_align(page_bytes, PAGE_SIZE).unwrap();
                dealloc(self.ptr.as_ptr() as *mut u8, layout);
            }
        }
    }
}

impl<T: Clone> Clone for PageAlignedBuffer<T> {
    fn clone(&self) -> Self {
        let mut new_buf = Self::with_capacity(self.len);
        for item in self.as_slice() {
            new_buf.push(item.clone());
        }
        new_buf
    }
}

impl<T: PartialEq> PartialEq for PageAlignedBuffer<T> {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.as_slice() == other.as_slice()
    }
}

impl<T: Eq> Eq for PageAlignedBuffer<T> {}

impl<T: fmt::Debug> fmt::Debug for PageAlignedBuffer<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_list().entries(self.as_slice()).finish()
    }
}

impl<T> Default for PageAlignedBuffer<T> {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

// 线程安全性：只要元素支持跨线程传输与共享，PageAlignedBuffer 即可支持
unsafe impl<T: Send> Send for PageAlignedBuffer<T> {}
unsafe impl<T: Sync> Sync for PageAlignedBuffer<T> {}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    #[derive(Debug, Clone, Copy, PartialEq)]
    struct Position {
        x: f64,
        y: f64,
    }

    #[test]
    fn test_alignment_and_metal_parts() {
        // 创建装载 10,000 个实体的缓冲区
        let mut buf = PageAlignedBuffer::<Position>::with_capacity(10_000);
        assert!(buf.capacity() >= 10_000);

        let (raw_ptr, raw_bytes) = buf.as_metal_raw_parts();
        // 验证硬性契约 1：首地址必须是 16384 的整数倍
        assert_eq!(raw_ptr as usize % PAGE_SIZE, 0, "Base pointer must be 16KB aligned");
        // 验证硬性契约 2：字节总长必须是 16384 的整数倍
        assert_eq!(raw_bytes % PAGE_SIZE, 0, "Allocated bytes must be 16KB multiple");

        // 验证每个元素仍然紧密排列（16 字节），没有元素空洞
        for i in 0..100 {
            buf.push(Position { x: i as f64, y: (i * 2) as f64 });
        }
        assert_eq!(buf.len(), 100);
        assert_eq!(buf[0], Position { x: 0.0, y: 0.0 });
        assert_eq!(buf[99], Position { x: 99.0, y: 198.0 });

        // 验证切片内部偏移：第 1 个元素与第 0 个元素相距恰好 16 字节
        let p0 = &buf[0] as *const Position as usize;
        let p1 = &buf[1] as *const Position as usize;
        assert_eq!(p1 - p0, std::mem::size_of::<Position>());
    }

    #[test]
    fn test_push_pop_swap_remove() {
        let mut buf = PageAlignedBuffer::new();
        buf.push(10);
        buf.push(20);
        buf.push(30);
        buf.push(40);
        assert_eq!(buf.len(), 4);

        // 测试 swap_remove
        let removed = buf.swap_remove(1); // 移除 20，将末尾 40 移入下标 1
        assert_eq!(removed, Some(20));
        assert_eq!(buf.len(), 3);
        assert_eq!(buf[0], 10);
        assert_eq!(buf[1], 40);
        assert_eq!(buf[2], 30);

        // 测试 pop
        assert_eq!(buf.pop(), Some(30));
        assert_eq!(buf.len(), 2);
    }

    #[test]
    fn test_reserve_and_realloc_integrity() {
        let mut buf = PageAlignedBuffer::<u64>::new();
        // 连续写入超过一个 16KB 页面容量的数据，触发多次自动扩容
        let count = 5000; // 5000 * 8 = 40,000 bytes > 16384 bytes
        for i in 0..count {
            buf.push(i as u64);
        }
        assert_eq!(buf.len(), count);

        // 验证扩容后所有元素数据无损
        for i in 0..count {
            assert_eq!(buf[i], i as u64);
        }

        // 验证扩容后的首地址依然严格保持 16KB 对齐
        let (raw_ptr, raw_bytes) = buf.as_metal_raw_parts();
        assert_eq!(raw_ptr as usize % PAGE_SIZE, 0);
        assert_eq!(raw_bytes % PAGE_SIZE, 0);
    }

    #[test]
    fn test_drop_cleanliness() {
        struct DropTracker {
            counter: Arc<AtomicUsize>,
        }
        impl Drop for DropTracker {
            fn drop(&mut self) {
                self.counter.fetch_add(1, Ordering::SeqCst);
            }
        }

        let drop_count = Arc::new(AtomicUsize::new(0));
        {
            let mut buf = PageAlignedBuffer::with_capacity(10);
            for _ in 0..5 {
                buf.push(DropTracker { counter: Arc::clone(&drop_count) });
            }
            assert_eq!(drop_count.load(Ordering::SeqCst), 0);
        }
        // 离开作用域，验证恰好触发 5 次析构，零内存泄漏
        assert_eq!(drop_count.load(Ordering::SeqCst), 5);
    }

    #[test]
    fn test_zst() {
        #[derive(Debug, PartialEq, Eq)]
        struct EmptyTag;

        let mut buf = PageAlignedBuffer::<EmptyTag>::new();
        buf.push(EmptyTag);
        buf.push(EmptyTag);
        assert_eq!(buf.len(), 2);
        assert_eq!(buf.allocated_bytes(), 0);
        assert_eq!(buf.pop(), Some(EmptyTag));
        assert_eq!(buf.len(), 1);
    }

    #[test]
    fn test_clone_independence() {
        let mut buf1 = PageAlignedBuffer::<String>::new();
        buf1.push("alpha".to_string());
        buf1.push("beta".to_string());

        let mut buf2 = buf1.clone();
        assert_eq!(buf1, buf2);

        buf2.push("gamma".to_string());
        assert_eq!(buf1.len(), 2);
        assert_eq!(buf2.len(), 3);

        let (p1, _) = buf1.as_metal_raw_parts();
        let (p2, _) = buf2.as_metal_raw_parts();
        assert_ne!(p1, p2, "Cloned buffer must have its own allocation");
        assert_eq!(p1 as usize % PAGE_SIZE, 0);
        assert_eq!(p2 as usize % PAGE_SIZE, 0);
    }
}
