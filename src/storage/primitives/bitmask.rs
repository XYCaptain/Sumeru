//! 连续列式状态存储的行级位掩码与字级跳跃/合并扫描原语 (RowBitMask & Mask Run Coalescing)
//!
//! # 性能设计与文献支撑
//!
//! 1. **Roaring Bitmaps & 硬件字跳跃 (Lemire et al. 2016)**:
//!    位图内部采用 64 位机器字（`u64`）紧凑连续布局，以 64 行为一个物理字（Word）。
//!    若某 64 行为全 0（`word == 0`），CPU 单条测试指令即可跳过 64 行实体的列切片与反序列化，
//!    实现稀疏谓词的 $O(1)$ 块级快速跳过。
//!
//! 2. **硬件 `ctz` / `trailing_zeros` 连续置位合并 (Hardware Run Coalescing)**:
//!    对于 `ChangedSince<T>` 这类以 64 行物理块全置位（`u64::MAX`）为特征的掩码，
//!    [`for_each_contiguous_mask_run`] 利用单周期硬件指令直接合并连续全 1 的机器字，
//!    将多个 64 行合并为单个连续切片区间 `[start, end)` 分发给内核，消除细粒度循环开销。

use std::ops::Range;

/// 原型表内行级紧凑位掩码 (`RowBitMask`)
///
/// 采用小向量内联存储优化：对于 $\le 256$ 行的小表，4 个 `u64` 机器字完全驻留栈上，零堆分配。
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct RowBitMask {
    words: smallvec::SmallVec<[u64; 4]>,
    row_len: usize,
}

impl RowBitMask {
    /// 创建长度为 `row_len`、所有位均置 0 的空位图
    #[inline]
    pub fn new_empty(row_len: usize) -> Self {
        let num_words = (row_len + 63) >> 6;
        Self {
            words: smallvec::smallvec![0u64; num_words],
            row_len,
        }
    }

    /// 创建长度为 `row_len`、所有有效行位均置 1 的满位图
    #[inline]
    pub fn new_all_set(row_len: usize) -> Self {
        let num_words = (row_len + 63) >> 6;
        let mut words = smallvec::smallvec![!0u64; num_words];
        let rem = row_len & 63;
        if rem != 0 {
            if let Some(last) = words.last_mut() {
                *last = (1u64 << rem) - 1;
            }
        }
        Self { words, row_len }
    }

    /// 从已有机器字切片构造位图，并自动裁剪超出 `row_len` 的尾部高位
    #[inline]
    pub fn from_words(words_slice: &[u64], row_len: usize) -> Self {
        let num_words = (row_len + 63) >> 6;
        let mut words = smallvec::SmallVec::<[u64; 4]>::with_capacity(num_words);
        for i in 0..num_words {
            words.push(words_slice.get(i).copied().unwrap_or(0));
        }
        let rem = row_len & 63;
        if rem != 0 {
            if let Some(last) = words.last_mut() {
                *last &= (1u64 << rem) - 1;
            }
        }
        Self { words, row_len }
    }

    /// 当前掩码对应的有效行数
    #[inline(always)]
    pub fn row_len(&self) -> usize {
        self.row_len
    }

    /// 底层 64 位机器字切片引用
    #[inline(always)]
    pub fn words(&self) -> &[u64] {
        &self.words
    }

    /// 底层 64 位机器字可变切片引用
    #[inline(always)]
    pub fn words_mut(&mut self) -> &mut [u64] {
        &mut self.words
    }

    /// 将第 `row` 行的标记位置 1
    #[inline(always)]
    pub fn set(&mut self, row: usize) {
        if row < self.row_len {
            self.words[row >> 6] |= 1u64 << (row & 63);
        }
    }

    /// 将第 `row` 行的标记位清除置 0
    #[inline(always)]
    pub fn clear_bit(&mut self, row: usize) {
        if row < self.row_len {
            self.words[row >> 6] &= !(1u64 << (row & 63));
        }
    }

    /// 测试第 `row` 行是否置 1
    #[inline(always)]
    pub fn test(&self, row: usize) -> bool {
        if row >= self.row_len {
            return false;
        }
        (self.words[row >> 6] & (1u64 << (row & 63))) != 0
    }

    /// 判断位图是否全为 0
    #[inline]
    pub fn is_all_zero(&self) -> bool {
        self.words.iter().all(|&w| w == 0)
    }

    /// 统计置 1 的行数量
    #[inline]
    pub fn count_ones(&self) -> usize {
        self.words.iter().map(|w| w.count_ones() as usize).sum()
    }

    /// 与另一组机器字按位求交（`self &= other_words`）
    #[inline]
    pub fn intersect_words(&mut self, other_words: &[u64]) {
        for (i, w) in self.words.iter_mut().enumerate() {
            *w &= other_words.get(i).copied().unwrap_or(0);
        }
    }

    /// 与另一位图按位求交（`self &= other`）
    #[inline]
    pub fn intersect_with(&mut self, other: &RowBitMask) {
        self.intersect_words(&other.words);
    }

    /// 与另一组机器字按位求差（`self &= !other_words`）
    #[inline]
    pub fn difference_words(&mut self, other_words: &[u64]) {
        let rem = self.row_len & 63;
        let num_words = self.words.len();
        for (i, w) in self.words.iter_mut().enumerate() {
            let ow = other_words.get(i).copied().unwrap_or(0);
            *w &= !ow;
            if i + 1 == num_words && rem != 0 {
                *w &= (1u64 << rem) - 1;
            }
        }
    }

    /// 与另一位图按位求差（`self &= !other`）
    #[inline]
    pub fn difference_with(&mut self, other: &RowBitMask) {
        self.difference_words(&other.words);
    }

    /// 与另一位图按位求并（`self |= other`）
    #[inline]
    pub fn union_with(&mut self, other: &RowBitMask) {
        let rem = self.row_len & 63;
        let num_words = self.words.len();
        for (i, w) in self.words.iter_mut().enumerate() {
            let ow = other.words.get(i).copied().unwrap_or(0);
            *w |= ow;
            if i + 1 == num_words && rem != 0 {
                *w &= (1u64 << rem) - 1;
            }
        }
    }

    /// 利用单周期硬件 `trailing_zeros` 指令跳跃遍历所有置 1 的行索引
    #[inline]
    pub fn for_each_set_row<F: FnMut(usize)>(&self, mut f: F) {
        for (word_idx, &word) in self.words.iter().enumerate() {
            let mut w = word;
            let base = word_idx << 6;
            while w != 0 {
                let bit = w.trailing_zeros() as usize;
                let row = base + bit;
                if row < self.row_len {
                    f(row);
                }
                w &= w - 1;
            }
        }
    }

    /// 检查在指定行区间 `[start, end)` 内是否存在任意置 1 的行
    #[inline]
    pub fn has_any_in_range(&self, start: usize, end: usize) -> bool {
        mask_has_any_in_range(self, start, end)
    }

    /// 在指定行区间 `[start, end)` 内扫描所有连续置 1 的区间段 `[run_start, run_end)`
    #[inline]
    pub fn for_each_contiguous_run(&self, start: usize, end: usize, f: impl FnMut(Range<usize>)) {
        for_each_contiguous_mask_run(self, start, end, f);
    }
}

/// 检查位掩码在指定行区间 `[start, end)` 内是否存在任意置 1 的行
#[inline]
pub fn mask_has_any_in_range(mask: &RowBitMask, start: usize, end: usize) -> bool {
    if start >= end {
        return false;
    }
    let words = mask.words();
    let start_word = start >> 6;
    let end_word = (end + 63) >> 6;
    for w_idx in start_word..end_word.min(words.len()) {
        let mut w = words[w_idx];
        if w_idx == start_word && (start & 63) != 0 {
            w &= !((1u64 << (start & 63)) - 1);
        }
        if (w_idx + 1) * 64 > end && (end & 63) != 0 {
            w &= (1u64 << (end & 63)) - 1;
        }
        if w != 0 {
            return true;
        }
    }
    false
}

/// 在 `[start, end)` 行区间内扫描 `RowBitMask` 中所有连续置位（`1`）的行区间 `[run_start, run_end)`。
///
/// 利用硬件 `trailing_zeros` 与全 1 字合并跳跃：对全 1 机器字快速累进，消弭逐位判断开销。
#[inline]
pub fn for_each_contiguous_mask_run(
    mask: &RowBitMask,
    start: usize,
    end: usize,
    mut f: impl FnMut(Range<usize>),
) {
    let words = mask.words();
    let limit = end.min(mask.row_len());
    let mut cur = start;
    while cur < limit {
        let w_idx = cur >> 6;
        if w_idx >= words.len() {
            break;
        }
        let bit = cur & 63;
        let w = words[w_idx] >> bit;
        if w == 0 {
            cur = (w_idx + 1) << 6;
            continue;
        }
        let tz = w.trailing_zeros() as usize;
        let run_start = cur + tz;
        if run_start >= limit {
            break;
        }
        let start_bit = bit + tz;
        let shifted = words[w_idx] >> start_bit;
        if shifted == (u64::MAX >> start_bit) {
            let mut next_w = w_idx + 1;
            while next_w < words.len() && (next_w << 6) < limit && words[next_w] == u64::MAX {
                next_w += 1;
            }
            let tail_ones = if next_w < words.len() && (next_w << 6) < limit {
                (!words[next_w]).trailing_zeros() as usize
            } else {
                0
            };
            let run_end = ((next_w << 6) + tail_ones).min(limit);
            f(run_start..run_end);
            cur = run_end + 1;
        } else {
            let ones = (!shifted).trailing_zeros() as usize;
            let run_end = (run_start + ones).min(limit);
            f(run_start..run_end);
            cur = run_end + 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_row_bit_mask_basic() {
        let mut mask = RowBitMask::new_empty(150);
        assert_eq!(mask.row_len(), 150);
        assert!(mask.is_all_zero());
        assert_eq!(mask.count_ones(), 0);

        mask.set(0);
        mask.set(63);
        mask.set(64);
        mask.set(149);
        assert_eq!(mask.count_ones(), 4);
        assert!(mask.test(0));
        assert!(mask.test(63));
        assert!(mask.test(64));
        assert!(mask.test(149));
        assert!(!mask.test(1));
        assert!(!mask.test(65));
        assert!(!mask.test(150));

        mask.clear_bit(63);
        assert!(!mask.test(63));
        assert_eq!(mask.count_ones(), 3);
    }

    #[test]
    fn test_row_bit_mask_all_set_and_ops() {
        let mask_all = RowBitMask::new_all_set(130);
        assert_eq!(mask_all.count_ones(), 130);
        assert!(!mask_all.test(130));

        let mut mask = RowBitMask::new_empty(130);
        mask.set(10);
        mask.set(70);

        let mut inter = mask_all.clone();
        inter.intersect_with(&mask);
        assert_eq!(inter.count_ones(), 2);
        assert!(inter.test(10));
        assert!(inter.test(70));

        let mut diff = mask_all.clone();
        diff.difference_with(&mask);
        assert_eq!(diff.count_ones(), 128);
        assert!(!diff.test(10));
        assert!(!diff.test(70));
        assert!(diff.test(0));
    }

    #[test]
    fn test_mask_has_any_in_range() {
        let mut mask = RowBitMask::new_empty(200);
        mask.set(100);

        assert!(!mask_has_any_in_range(&mask, 0, 100));
        assert!(mask_has_any_in_range(&mask, 0, 101));
        assert!(mask_has_any_in_range(&mask, 100, 101));
        assert!(mask_has_any_in_range(&mask, 90, 150));
        assert!(!mask_has_any_in_range(&mask, 101, 200));
    }

    #[test]
    fn test_for_each_contiguous_mask_run() {
        let mut mask = RowBitMask::new_empty(300);
        for i in 10..20 {
            mask.set(i);
        }
        for i in 64..192 {
            mask.set(i);
        }
        mask.set(250);

        let mut runs = Vec::new();
        mask.for_each_contiguous_run(0, 300, |r| runs.push(r));

        assert_eq!(runs.len(), 3);
        assert_eq!(runs[0], 10..20);
        assert_eq!(runs[1], 64..192);
        assert_eq!(runs[2], 250..251);
    }
}
