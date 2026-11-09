//! 结构变更延迟命令缓冲 (`Commands`) 与并行 Stage 同步点 (Sync Point)
//!
//! # 理论支撑
//!
//! 1. **事件暂存 → 有序批量同步 (Zhao et al., SAC '26, arXiv:2606.14919)**:
//!    archetype ECS 的三条核心不变量之一：系统执行期间的结构修改（创建/销毁实体、增删组件）
//!    必须暂存为事件，在系统执行结束后的同步点按确定顺序批量应用，以维持 SoA 表的稳定布局。
//! 2. **Flecs deferred command queue / Bevy `Commands` + `apply_deferred`**:
//!    并行 Stage 内的系统只能向各自的命令缓冲追加命令；Stage 结束时由主线程独占应用。
//!
//! # 确定性
//!
//! - 并行查询系统的每个子块任务持有独立缓冲，任务内按行序追加；
//! - 子块缓冲按**任务序**（原型序 × 行序）拼接，系统缓冲按**系统注册序**应用；
//! - 因此最终应用顺序与线程调度无关（调度无关确定性，Redmond & Kuper, OOPSLA '25）。

use crate::error::StateError;
use crate::storage::core::store::Store;
use crate::EntityId;

type DeferredCommand = Box<dyn FnOnce(&mut Store) -> Result<(), StateError> + Send>;

/// 延迟结构变更命令缓冲
#[derive(Default)]
pub struct Commands {
    ops: Vec<DeferredCommand>,
}

impl std::fmt::Debug for Commands {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Commands").field("len", &self.ops.len()).finish()
    }
}

impl Commands {
    /// 创建空命令缓冲
    #[inline]
    pub fn new() -> Self {
        Self::default()
    }

    /// 追加任意延迟执行的 Store 变更闭包（在 Stage 同步点由主线程独占执行）
    #[inline]
    pub fn push(
        &mut self,
        op: impl FnOnce(&mut Store) -> Result<(), StateError> + Send + 'static,
    ) {
        self.ops.push(Box::new(op));
    }

    /// 延迟销毁实体
    #[inline]
    pub fn despawn(&mut self, entity: EntityId) {
        self.push(move |store| store.despawn(entity));
    }

    /// 已缓冲的命令数量
    #[inline]
    pub fn len(&self) -> usize {
        self.ops.len()
    }

    /// 是否为空
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.ops.is_empty()
    }

    /// 将另一缓冲的命令按序追加到末尾
    #[inline]
    pub fn append(&mut self, other: &mut Commands) {
        self.ops.append(&mut other.ops);
    }

    /// 在同步点按追加顺序独占应用全部命令；遇到首个错误立即返回（其余命令被丢弃）
    pub fn apply(self, store: &mut Store) -> Result<usize, StateError> {
        let n = self.ops.len();
        for op in self.ops {
            op(store)?;
        }
        Ok(n)
    }
}
