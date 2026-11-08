//! 原型集合倒排位图索引原语 (`ArchetypeBitSet`)
//!
//! # 性能设计与文献支撑
//!
//! 在实体组件系统（ECS）与列式数据仓库中，查询规划器需要对成百上千个原型表进行快速谓词与签名匹配。
//! `ArchetypeBitSet` 采用 64 位机器字对原型 ID（`ArchetypeId` / `u32`）进行位图索引。
//!
//! 1. **常数时间增量求交 (Flecs-style Inverted Index)**:
//!    `Query` 启动或匹配时，通过对查询所需的各个组件倒排位图执行 `words[i] &= other[i]` 位运算，
//!    单个 CPU 周期即可完成 64 个原型的存在性裁决，杜绝了全量线性遍历。
//! 2. **硬件 `trailing_zeros` 极速展开**:
//!    在得到匹配的原型位图后，单周期硬件指令直接跳出匹配的原型 ID，执行开销与未命中的 Archetype 数量完全无关。

/// 原型 ID 倒排位图索引集合
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct ArchetypeBitSet {
    words: smallvec::SmallVec<[u64; 2]>,
}

impl ArchetypeBitSet {
    /// 创建空的 ArchetypeBitSet
    #[inline]
    pub fn new() -> Self {
        Self {
            words: smallvec::SmallVec::new(),
        }
    }

    /// 向集合中插入一个原型 ID
    #[inline]
    pub fn insert(&mut self, arch_id: u32) {
        let idx = (arch_id as usize) >> 6;
        let bit = (arch_id as usize) & 63;
        if idx >= self.words.len() {
            self.words.resize(idx + 1, 0);
        }
        self.words[idx] |= 1u64 << bit;
    }

    /// 检查集合中是否包含指定的原型 ID
    #[inline]
    pub fn contains(&self, arch_id: u32) -> bool {
        let idx = (arch_id as usize) >> 6;
        let bit = (arch_id as usize) & 63;
        if idx >= self.words.len() {
            return false;
        }
        (self.words[idx] & (1u64 << bit)) != 0
    }

    /// 从集合中移除指定的原型 ID
    #[inline]
    pub fn remove(&mut self, arch_id: u32) {
        let idx = (arch_id as usize) >> 6;
        let bit = (arch_id as usize) & 63;
        if idx < self.words.len() {
            self.words[idx] &= !(1u64 << bit);
        }
    }

    /// 底层机器字切片引用
    #[inline(always)]
    pub fn words(&self) -> &[u64] {
        &self.words
    }

    /// 底层机器字可变切片引用
    #[inline(always)]
    pub fn words_mut(&mut self) -> &mut [u64] {
        &mut self.words
    }

    /// 检查位图是否全为 0
    #[inline]
    pub fn is_all_zero(&self) -> bool {
        self.words.iter().all(|&w| w == 0)
    }

    /// 统计置 1 的原型数量
    #[inline]
    pub fn count_ones(&self) -> usize {
        self.words.iter().map(|w| w.count_ones() as usize).sum()
    }

    /// 清空集合
    #[inline]
    pub fn clear(&mut self) {
        self.words.clear();
    }

    /// 与另一位图按位求交（`self &= other`）
    #[inline]
    pub fn intersect_with(&mut self, other: &Self) {
        let min_len = self.words.len().min(other.words.len());
        for i in 0..min_len {
            self.words[i] &= other.words[i];
        }
        if self.words.len() > min_len {
            self.words.truncate(min_len);
        }
    }

    /// 与另一位图按位求差（`self &= !other`）
    #[inline]
    pub fn difference_with(&mut self, other: &Self) {
        let min_len = self.words.len().min(other.words.len());
        for i in 0..min_len {
            self.words[i] &= !other.words[i];
        }
    }

    /// 与另一位图按位求并（`self |= other`）
    #[inline]
    pub fn union_with(&mut self, other: &Self) {
        if other.words.len() > self.words.len() {
            self.words.resize(other.words.len(), 0);
        }
        for (i, &w) in other.words.iter().enumerate() {
            self.words[i] |= w;
        }
    }

    /// 利用单周期硬件 `trailing_zeros` 指令遍历集合中所有置 1 的原型 ID
    #[inline]
    pub fn for_each_set_bit<F: FnMut(u32)>(&self, mut f: F) {
        for (word_idx, &word) in self.words.iter().enumerate() {
            let mut w = word;
            let base = (word_idx as u32) << 6;
            while w != 0 {
                let bit = w.trailing_zeros();
                f(base + bit);
                w &= w - 1;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_archetype_bit_set_basic() {
        let mut set = ArchetypeBitSet::new();
        assert!(set.is_all_zero());
        assert_eq!(set.count_ones(), 0);

        set.insert(0);
        set.insert(63);
        set.insert(64);
        set.insert(130);

        assert!(set.contains(0));
        assert!(set.contains(63));
        assert!(set.contains(64));
        assert!(set.contains(130));
        assert!(!set.contains(1));
        assert!(!set.contains(65));
        assert_eq!(set.count_ones(), 4);

        set.remove(63);
        assert!(!set.contains(63));
        assert_eq!(set.count_ones(), 3);

        let mut collected = Vec::new();
        set.for_each_set_bit(|id| collected.push(id));
        assert_eq!(collected, vec![0, 64, 130]);
    }

    #[test]
    fn test_archetype_bit_set_intersect_difference() {
        let mut a = ArchetypeBitSet::new();
        a.insert(1);
        a.insert(2);
        a.insert(100);

        let mut b = ArchetypeBitSet::new();
        b.insert(2);
        b.insert(100);
        b.insert(200);

        let mut inter = a.clone();
        inter.intersect_with(&b);
        assert_eq!(inter.count_ones(), 2);
        assert!(inter.contains(2));
        assert!(inter.contains(100));
        assert!(!inter.contains(1));

        let mut diff = a.clone();
        diff.difference_with(&b);
        assert_eq!(diff.count_ones(), 1);
        assert!(diff.contains(1));
        assert!(!diff.contains(2));
    }
}
