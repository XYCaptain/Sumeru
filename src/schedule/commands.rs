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
use crate::storage::batch::EntityBundle;
use crate::storage::core::store::Store;
use crate::storage::entity::IntoEntity;
use crate::storage::stage::commands::{CommandBuffer, EntityCommands};
use crate::EntityId;

type DeferredCommand = Box<dyn FnOnce(&mut Store) -> Result<(), StateError> + Send>;

/// 延迟结构变更命令缓冲（内嵌两阶段零分配 [`CommandBuffer`] + 扩展闭包通道）
pub struct Commands {
    /// 当前活跃的尾部 `CommandBuffer`
    buffer: CommandBuffer,
    /// 跨子块合并时不可简单折叠的前序 `CommandBuffer` 片段（按任务序排列）
    segments: Vec<CommandBuffer>,
    /// 自定义延迟 Store 操作（如 `set_flag` 或自定义闭包）
    ops: Vec<DeferredCommand>,
}

impl Default for Commands {
    #[inline]
    fn default() -> Self {
        Self {
            buffer: CommandBuffer::new_unbound(),
            segments: Vec::new(),
            ops: Vec::new(),
        }
    }
}

impl std::fmt::Debug for Commands {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Commands")
            .field("len", &self.len())
            .field("segments", &self.segments.len())
            .field("ops", &self.ops.len())
            .finish()
    }
}

impl Commands {
    /// 创建空命令缓冲
    #[inline]
    pub fn new() -> Self {
        Self::default()
    }

    /// 获取当前内嵌 [`CommandBuffer`] 的可变引用（直接调用全部 `CommandBuffer` 原生流式 API）
    #[inline]
    pub fn buffer_mut(&mut self) -> &mut CommandBuffer {
        &mut self.buffer
    }

    /// 获取针对特定实体的流式命令构建器 [`EntityCommands`]
    ///
    /// 支持 `.add` / `.insert` / `.replace` / `.upsert` / `.set` / `.remove` /
    /// `.set_pair` / `.remove_pair` / `.add_tag` / `.remove_tag` / `.despawn`。
    #[inline]
    pub fn entity(&mut self, id: EntityId) -> EntityCommands<'_> {
        self.buffer.entity(id)
    }

    /// 排队创建实体并赋予初始组件束（支持 `Entity` 或 `(EntityId, (T1, ..., Tn))` 多元组）
    #[inline]
    pub fn insert(&mut self, input: impl IntoEntity) -> &mut Self {
        self.buffer.insert(input);
        self
    }

    /// 批量排队创建实体
    #[inline]
    pub fn insert_batch<I, B>(&mut self, batch: I) -> &mut Self
    where
        I: IntoIterator<Item = B>,
        B: IntoEntity,
    {
        self.buffer.insert_batch(batch);
        self
    }

    /// 排队创建一个空实体（后续可通过 `cmds.entity(id).set(component)` 追加组件）
    #[inline]
    pub fn create_empty(&mut self, id: EntityId) -> &mut Self {
        self.buffer.create_empty(id);
        self
    }

    /// 排队创建实体并赋予初始组件束
    #[inline]
    pub fn create_entity(&mut self, id: EntityId, bundle: EntityBundle) -> &mut Self {
        self.buffer.create_entity(id, bundle);
        self
    }

    /// 延迟销毁实体（直接写入内嵌 `CommandBuffer`，零 `Box<dyn FnOnce>` 堆分配）
    #[inline]
    pub fn despawn(&mut self, entity: EntityId) -> &mut Self {
        self.buffer.despawn(entity);
        self
    }

    /// 延迟删除实体（`despawn` 的兼容别名）
    #[inline]
    pub fn delete_entity(&mut self, entity: EntityId) -> &mut Self {
        self.buffer.delete_entity(entity);
        self
    }

    /// 延迟设置实体的非迁移位图状态标记 `FLAG` (`0..8`)
    #[inline]
    pub fn set_flag<const FLAG: u8>(&mut self, entity: EntityId, enabled: bool) -> &mut Self {
        const { assert!(FLAG < 8, "RowFlag index must be in 0..8") };
        self.push(move |store| store.set_flag::<FLAG>(entity, enabled).map(|_| ()));
        self
    }

    /// 延迟设置实体的非迁移位图状态标记 `FLAG` (`0..8`)（`set_flag` 的语义别名）
    #[inline]
    pub fn set_row_flag<const FLAG: u8>(&mut self, entity: EntityId, enabled: bool) -> &mut Self {
        self.set_flag::<FLAG>(entity, enabled)
    }

    /// 追加任意延迟执行的 Store 变更闭包（在 Stage 同步点由主线程独占执行）
    #[inline]
    pub fn push(
        &mut self,
        op: impl FnOnce(&mut Store) -> Result<(), StateError> + Send + 'static,
    ) {
        self.ops.push(Box::new(op));
    }

    /// 已缓冲的命令与实体变更数量
    #[inline]
    pub fn len(&self) -> usize {
        let seg_len: usize = self
            .segments
            .iter()
            .map(|s| s.len() + usize::from(s.has_deferred_error()))
            .sum();
        seg_len
            + self.buffer.len()
            + usize::from(self.buffer.has_deferred_error())
            + self.ops.len()
    }

    /// 是否为空
    #[inline]
    pub fn is_empty(&self) -> bool {
        !self.buffer.has_recorded_work()
            && self.segments.iter().all(|s| !s.has_recorded_work())
            && self.ops.is_empty()
    }

    #[inline]
    fn append_command_buffer(&mut self, mut buf: CommandBuffer) {
        if !buf.has_recorded_work() {
            return;
        }
        if !self.buffer.has_recorded_work() {
            self.buffer = buf;
        } else if self.buffer.can_merge_simple_entries() && buf.can_merge_simple_entries() {
            self.buffer.merge_simple_from(&mut buf);
        } else {
            let prev = std::mem::replace(&mut self.buffer, buf);
            self.segments.push(prev);
        }
    }

    /// 将另一缓冲的命令按任务序追加到末尾
    pub fn append(&mut self, other: &mut Commands) {
        for seg in other.segments.drain(..) {
            self.append_command_buffer(seg);
        }
        if other.buffer.has_recorded_work() {
            let tail = std::mem::replace(&mut other.buffer, CommandBuffer::new_unbound());
            self.append_command_buffer(tail);
        }
        self.ops.append(&mut other.ops);
    }

    /// 在同步点按追加顺序独占应用全部命令；遇到首个错误立即返回（其余命令被丢弃）
    pub fn apply(self, store: &mut Store) -> Result<usize, StateError> {
        let total = self.len();
        for mut seg in self.segments {
            if seg.has_recorded_work() {
                seg.rebind_to_store(store);
                seg.apply_discard(store)?;
            }
        }
        if self.buffer.has_recorded_work() {
            let mut tail = self.buffer;
            tail.rebind_to_store(store);
            tail.apply_discard(store)?;
        }
        for op in self.ops {
            op(store)?;
        }
        Ok(total)
    }
}
