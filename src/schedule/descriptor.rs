//! 系统组件读写权限描述符与冲突判定原语 (`AccessDescriptor`)
//!
//! # 理论支撑与文献证明
//!
//! 1. **ECS 静态读写分离安全理论 (ACM SAC '26 / Zhao et al. 2026, arXiv:2606.14919)**:
//!    Zhao 等人建立了 ECS 系统级并发的形式化类型系统：当调度器证明系统中任意两个并发执行者
//!    $S_i, S_j$ 满足：
//!    $$(Writes_i \cap (Writes_j \cup Reads_j) = \emptyset) \land (Writes_j \cap (Writes_i \cup Reads_i) = \emptyset)$$
//!    且均不持有独占全局世界变更权时，两个系统的并发执行具有**静态无数据竞争 (Data-Race Free)**
//!    与**执行结果确定性 (Determinism)**。
//!
//! 2. **调度期并发重定位 (Core ECS / Redmond & Kuper 2025, arXiv:2508.15264)**:
//!    将并发安全检查从微观执行循环（如原子借用检查、运行时锁、`RefCell`）上移至宏观调度图构建期。
//!    一旦调度图在构建期证明无冲突，执行期即可完全抹除锁与原子同步开销。

use std::collections::BTreeSet;
use crate::id::ComponentTypeId;

/// 调度系统标识符
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SystemId(pub u32);

/// 系统对组件与全局世界的访问权限描述符
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct AccessDescriptor {
    /// 只读组件集合
    pub reads: BTreeSet<ComponentTypeId>,
    /// 可变写入组件集合
    pub writes: BTreeSet<ComponentTypeId>,
    /// 是否需要对 Store 的独占结构访问（例如创建/销毁实体、添加/删除组件、全局清空）
    pub exclusive: bool,
}

impl AccessDescriptor {
    /// 创建空的访问描述符
    #[inline]
    pub fn new() -> Self {
        Self::default()
    }

    /// 创建独占全局世界的访问描述符
    #[inline]
    pub fn new_exclusive() -> Self {
        Self {
            reads: BTreeSet::new(),
            writes: BTreeSet::new(),
            exclusive: true,
        }
    }

    /// 注册一个只读组件依赖
    #[inline]
    pub fn with_read(mut self, type_id: ComponentTypeId) -> Self {
        self.reads.insert(type_id);
        self
    }

    /// 注册一个可变写组件依赖
    #[inline]
    pub fn with_write(mut self, type_id: ComponentTypeId) -> Self {
        self.writes.insert(type_id);
        self
    }

    /// 标记为需要独占访问
    #[inline]
    pub fn with_exclusive(mut self, exclusive: bool) -> Self {
        self.exclusive = exclusive;
        self
    }

    /// 合并另一个描述符的权限需求
    pub fn merge(&mut self, other: &AccessDescriptor) {
        self.exclusive |= other.exclusive;
        self.reads.extend(other.reads.iter().copied());
        self.writes.extend(other.writes.iter().copied());
    }

    /// 判定本描述符与另一个描述符是否存在读写或写写冲突
    ///
    /// 冲突条件（Zhao et al. 2026 定理 1）：
    /// 1. 任一方需要全局独占访问（`exclusive`）；
    /// 2. $Writes_1 \cap Writes_2 \neq \emptyset$（写写冲突）；
    /// 3. $Writes_1 \cap Reads_2 \neq \emptyset$（写读冲突）；
    /// 4. $Reads_1 \cap Writes_2 \neq \emptyset$（读写冲突）。
    #[inline]
    pub fn conflicts_with(&self, other: &AccessDescriptor) -> bool {
        if self.exclusive || other.exclusive {
            return true;
        }

        // 1. 写写冲突
        if self.writes.iter().any(|w| other.writes.contains(w)) {
            return true;
        }

        // 2. 写读冲突 (self 写，other 读)
        if self.writes.iter().any(|w| other.reads.contains(w)) {
            return true;
        }

        // 3. 读写冲突 (self 读，other 写)
        if self.reads.iter().any(|r| other.writes.contains(r)) {
            return true;
        }

        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_access_descriptor_conflict() {
        let cid_a = ComponentTypeId::from_name("ComponentA");
        let cid_b = ComponentTypeId::from_name("ComponentB");
        let cid_c = ComponentTypeId::from_name("ComponentC");

        // 系统 1: 写 A，读 B
        let sys1 = AccessDescriptor::new().with_write(cid_a).with_read(cid_b);

        // 系统 2: 读 B，读 C (只读共享，无冲突)
        let sys2 = AccessDescriptor::new().with_read(cid_b).with_read(cid_c);
        assert!(!sys1.conflicts_with(&sys2));
        assert!(!sys2.conflicts_with(&sys1));

        // 系统 3: 写 B (与 sys1 读 B 冲突)
        let sys3 = AccessDescriptor::new().with_write(cid_b);
        assert!(sys1.conflicts_with(&sys3));
        assert!(sys3.conflicts_with(&sys1));

        // 系统 4: 写 A (与 sys1 写 A 写写冲突)
        let sys4 = AccessDescriptor::new().with_write(cid_a);
        assert!(sys1.conflicts_with(&sys4));

        // 系统 5: 独占世界
        let sys5 = AccessDescriptor::new_exclusive();
        assert!(sys5.conflicts_with(&sys2));
        assert!(sys2.conflicts_with(&sys5));
    }
}
