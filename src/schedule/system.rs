//! 调度系统定义与自动类型推导原语 (`System` & `IntoSystem`)
//!
//! # 理论支撑与文献证明
//!
//! 1. **零注解权限推导 (Compile-Time Access Deduction)**:
//!    基于 Sumeru 的 `QueryMutShape`，系统所需查询的元组类型 `Q`（如 `(&mut Position, &Velocity)`）
//!    在编译期即可通过 `Q::access_descriptors()` 自动展开为精确的只读/可变组件类型集合，
//!    完全无需用户手动书写任何宏或在运行时声明依赖。
//!
//! 2. **三级缓存与拓扑增量协同 (Flecs Level-2 Index Caching)**:
//!    每个系统内联持有独立的 `Query<Q, F>` 句柄。当 `Store` 的原型集合未发生结构变化
//!    （`archetype_set_generation` 保持一致）时，系统直接复用已预绑定的物理列槽位索引，
//!    执行期开销降至绝对最低。
//!
//! # 系统签名分层（健全性契约）
//!
//! | 类型 | 访问 | 能否进入并行 Stage | 结构修改 |
//! |------|------|--------------------|----------|
//! | [`ExclusiveFunctionSystem`] | `&mut Store` | 否（恒独占） | 直接修改 |
//! | [`QuerySystem`] | `&mut Query` + `&mut Store` | 否（恒独占，读写集仅用于诊断） | 直接修改 |
//! | [`ParQuerySystem`] | 仅 `Q::Item` 与 [`Commands`] | 是 | 经 `Commands` 在同步点应用 |
//! | [`ParChunkQuerySystem`] | 仅 `Q::Chunk` 连续切片与 [`Commands`] | 是 | 经 `Commands` 在同步点应用 |
//!
//! 并行系统只能通过 [`System::prepare_parallel`] 进入并行 Stage：主线程用 [`PrepareContext`]
//! 完成全部 `Store` 元数据访问（缓存刷新、COW 解耦、纪元标记），返回的 [`ParallelJob`]
//! 只持有列缓冲区物理指针，**不持有 `Store`**。`PrepareContext` 的 `Store` 访问器为
//! `pub(crate)`，外部实现无法把 `Store` 捕获进并行作业，从而由类型系统强制
//! 读写集声明与实际访问一致（Zhao et al., SAC '26, arXiv:2606.14919）。

use std::marker::PhantomData;
use crate::error::StateError;
use crate::storage::core::store::Store;
use crate::storage::query::handle::{Query, QueryBuilder};
use crate::storage::query::QueryMutShape;
use crate::storage::query::spec::{All, QueryFilter};
use crate::schedule::commands::Commands;
use crate::schedule::descriptor::AccessDescriptor;

/// 并行 Stage 中由主线程预备完毕、可在工作线程独立执行的系统作业。
///
/// 作业不得访问 `Store`；其产生的结构变更通过返回的 [`Commands`] 在 Stage 同步点应用。
pub type ParallelJob<'w> = Box<dyn FnOnce() -> Result<Commands, StateError> + Send + 'w>;

/// 并行 Stage 准备阶段上下文（仅在主线程、Stage 内顺序构造）。
///
/// 同一 Stage 的所有系统共享同一个 `stage_epoch`：Stage 开始前（若存在写入）`Store` 纪元只推进一次，
/// 并行作业期间不再有任何线程写 `Store` 级元数据（消除 P0-3 共享元数据竞争）。
pub struct PrepareContext<'w> {
    store: &'w mut Store,
    stage_epoch: u64,
}

impl<'w> PrepareContext<'w> {
    #[inline]
    pub(crate) fn new(store: &'w mut Store, stage_epoch: u64) -> Self {
        Self { store, stage_epoch }
    }

    /// 本并行 Stage 的统一写纪元
    #[inline]
    pub fn stage_epoch(&self) -> u64 {
        self.stage_epoch
    }

    /// 只读访问 Store（crate 内部）
    #[inline]
    #[allow(dead_code)]
    pub(crate) fn store(&self) -> &Store {
        self.store
    }

    /// 消耗上下文取得 `'w` 生命周期的独占 Store（crate 内部，仅用于主线程准备阶段）
    #[inline]
    pub(crate) fn into_store(self) -> &'w mut Store {
        self.store
    }
}

/// 调度系统核心接口
pub trait System: Send + 'static {
    /// 系统的诊断名称
    fn name(&self) -> &'static str;

    /// 系统的组件访问权限描述符
    fn access(&self) -> &AccessDescriptor;

    /// 系统初始化与元数据预绑定（在 `Schedule::build` 时执行，错误会向上传播）
    fn initialize(&mut self, _store: &Store) -> Result<(), StateError> {
        Ok(())
    }

    /// 独占执行（仅当 `access().exclusive == true` 时由调度器调用）
    fn run(&mut self, _store: &mut Store) -> Result<(), StateError> {
        Err(StateError::AccessDenied(format!(
            "system `{}` is not exclusive and cannot run with &mut Store",
            self.name()
        )))
    }

    /// 并行 Stage 准备阶段（仅当 `access().exclusive == false` 时由调度器在主线程调用）
    fn prepare_parallel<'w>(
        &'w mut self,
        _ctx: PrepareContext<'w>,
    ) -> Result<ParallelJob<'w>, StateError> {
        Err(StateError::AccessDenied(format!(
            "system `{}` does not support parallel execution",
            self.name()
        )))
    }
}

/// 支持转换为 [`System`] 的类型特征
pub trait IntoSystem<Marker>: Sized {
    type System: System;

    fn into_system(self) -> Self::System;
}

// ---------------------------------------------------------------------------
// 1. 原生 `&mut Store` 闭包/函数系统（恒为独占全局访问）
// ---------------------------------------------------------------------------

pub struct ExclusiveFunctionSystem<F> {
    name: &'static str,
    f: F,
    access: AccessDescriptor,
}

impl<F> ExclusiveFunctionSystem<F>
where
    F: FnMut(&mut Store) + Send + 'static,
{
    /// 创建独占系统。持有 `&mut Store` 的系统无法被静态证明只访问声明的列，
    /// 因此恒为独占，不提供降级为局部访问的入口。
    pub fn new(name: &'static str, f: F) -> Self {
        Self {
            name,
            f,
            access: AccessDescriptor::new_exclusive(),
        }
    }
}

impl<F> System for ExclusiveFunctionSystem<F>
where
    F: FnMut(&mut Store) + Send + 'static,
{
    #[inline(always)]
    fn name(&self) -> &'static str {
        self.name
    }

    #[inline(always)]
    fn access(&self) -> &AccessDescriptor {
        &self.access
    }

    #[inline(always)]
    fn run(&mut self, store: &mut Store) -> Result<(), StateError> {
        store.check_poisoned()?;
        (self.f)(store);
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// 2. 基于单 `Query<Q, F>` + `&mut Store` 的独占查询系统
// ---------------------------------------------------------------------------

/// 单查询系统：闭包同时拿到 `&mut Query` 与 `&mut Store`，因此**恒为独占**。
///
/// - `access()` 仍携带由 `Q` 与过滤器推导出的读写集、原型谓词与行标志位（`exclusive == true`），供诊断与歧义报告使用；
/// - 自动维护 `last_run_epoch` 游标，使 [`ChangedSince`](crate::storage::query::spec::ChangedSince) 跨帧/跨 Tick 自动只匹配上次运行后的增量变更；
/// - 需要并行执行时请使用 [`ParQuerySystem`] 或 [`ParChunkQuerySystem`]。
pub struct QuerySystem<Q, F, Func>
where
    Q: QueryMutShape,
    F: QueryFilter + Clone,
{
    name: &'static str,
    filter: F,
    query: Option<Query<Q, F>>,
    func: Func,
    access: AccessDescriptor,
    last_run_epoch: Option<u64>,
}

/// 由查询形态与过滤器推导访问描述符：
/// 1. 合并 `Q` 与行级过滤器的组件读写集（修复 P0-4）；
/// 2. 提取 `Q` 必选组件与 `With<T>` / `Without<T>` 过滤器谓词至 `with_components` / `without_components`，
///    启用互不相交原型并发剪枝；
/// 3. 提取 `WithFlag<FLAG>` / `WithoutFlag<FLAG>` 读取的行级标志位掩码至 `reads_flags`。
pub(crate) fn deduce_access<Q: QueryMutShape, F: QueryFilter>(filter: &F) -> AccessDescriptor {
    let mut access = AccessDescriptor::new();
    Q::for_each_param_descriptor(|cid, _name, _write, required| {
        if required {
            access.with_components.insert(cid);
        }
    });
    for (cid, _, is_write) in crate::storage::query::spec::merged_access_descriptors::<Q, F>(filter) {
        if is_write {
            access.writes.insert(cid);
        } else {
            access.reads.insert(cid);
        }
    }
    let mut reqs = Vec::new();
    filter.archetype_requirements(&mut reqs);
    for (cid, must_have) in reqs {
        if must_have {
            access.with_components.insert(cid);
        } else {
            access.without_components.insert(cid);
        }
    }
    access.reads_flags = filter.filter_read_flags();
    access
}

/// 构建或增量刷新系统内联查询（L2 缓存按 `archetype_set_generation` 失效）
fn ensure_query<Q, F>(
    slot: &mut Option<Query<Q, F>>,
    filter: &F,
    store: &Store,
) -> Result<(), StateError>
where
    Q: QueryMutShape,
    F: QueryFilter + Clone,
{
    match slot {
        Some(query) => query.ensure_cache_current(store),
        None => {
            *slot = Some(QueryBuilder::<Q, F>::with_filter(filter.clone()).build(store)?);
            Ok(())
        }
    }
}

impl<Q, F, Func> QuerySystem<Q, F, Func>
where
    Q: QueryMutShape,
    F: QueryFilter + Clone,
    Func: FnMut(&mut Query<Q, F>, &mut Store) + Send + 'static,
{
    pub fn new(name: &'static str, filter: F, func: Func) -> Self {
        let access = deduce_access::<Q, F>(&filter).with_exclusive(true);
        Self {
            name,
            filter,
            query: None,
            func,
            access,
            last_run_epoch: None,
        }
    }

    /// 声明该系统写入的行级标志位掩码（用于 `WithFlag` 冲突与歧义检测）
    #[inline]
    pub fn with_write_flag(mut self, flag_mask: u8) -> Self {
        self.access.writes_flags |= flag_mask;
        self
    }

    /// 可变访问描述符
    #[inline]
    pub fn access_mut(&mut self) -> &mut AccessDescriptor {
        &mut self.access
    }
}

impl<Q, F, Func> System for QuerySystem<Q, F, Func>
where
    Q: QueryMutShape,
    F: QueryFilter + Clone,
    Func: FnMut(&mut Query<Q, F>, &mut Store) + Send + 'static,
{
    #[inline(always)]
    fn name(&self) -> &'static str {
        self.name
    }

    #[inline(always)]
    fn access(&self) -> &AccessDescriptor {
        &self.access
    }

    fn initialize(&mut self, store: &Store) -> Result<(), StateError> {
        ensure_query(&mut self.query, &self.filter, store)
    }

    #[inline]
    fn run(&mut self, store: &mut Store) -> Result<(), StateError> {
        store.check_poisoned()?;
        if let Some(ep) = self.last_run_epoch {
            self.filter.update_since_epoch(ep);
            if let Some(q) = &mut self.query {
                q.set_filter_since_epoch(ep);
            }
        }
        // 每次运行前按拓扑代号增量重验：前序 Stage 新建的原型在本 Stage 可见（P1-4）
        ensure_query(&mut self.query, &self.filter, store)?;
        if let Some(query) = &mut self.query {
            (self.func)(query, store);
        }
        self.last_run_epoch = Some(store.epoch());
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// 3. 可进入并行 Stage 的逐行查询系统 (`ParQuerySystem`)
// ---------------------------------------------------------------------------

/// 并行逐行查询系统：闭包只能看到 `Q::Item` 与本子块的 [`Commands`]，看不到 `Store`。
///
/// - 读写集、原型 `With`/`Without` 谓词及行标志位由 `Q` 与过滤器自动推导，受类型系统强制；
/// - 自动维护 `last_run_epoch` 游标，使 [`ChangedSince`](crate::storage::query::spec::ChangedSince) 跨帧自动推进；
/// - 准备阶段（主线程）完成缓存刷新、写列 COW 解耦与 Stage 纪元标记；
/// - 执行阶段按 `SubchunkPlanner` 子块并行，每个子块持有独立命令缓冲，按任务序拼接。
pub struct ParQuerySystem<Q, F, Func>
where
    Q: QueryMutShape,
    F: QueryFilter + Clone,
{
    name: &'static str,
    filter: F,
    query: Option<Query<Q, F>>,
    func: Func,
    access: AccessDescriptor,
    last_run_epoch: Option<u64>,
}

impl<Q, F, Func> ParQuerySystem<Q, F, Func>
where
    Q: QueryMutShape,
    F: QueryFilter + Clone + Send + Sync + 'static,
    Func: for<'a> Fn(Q::Item<'a>, &mut Commands) + Send + Sync + 'static,
{
    pub fn new(name: &'static str, filter: F, func: Func) -> Self {
        let access = deduce_access::<Q, F>(&filter);
        Self {
            name,
            filter,
            query: None,
            func,
            access,
            last_run_epoch: None,
        }
    }

    /// 声明该系统（例如通过 `Commands::set_flag`）写入的行级标志位掩码
    #[inline]
    pub fn with_write_flag(mut self, flag_mask: u8) -> Self {
        self.access.writes_flags |= flag_mask;
        self
    }

    /// 可变访问描述符
    #[inline]
    pub fn access_mut(&mut self) -> &mut AccessDescriptor {
        &mut self.access
    }
}

impl<Q, F, Func> System for ParQuerySystem<Q, F, Func>
where
    Q: QueryMutShape,
    F: QueryFilter + Clone + Send + Sync + 'static,
    Func: for<'a> Fn(Q::Item<'a>, &mut Commands) + Send + Sync + 'static,
{
    #[inline(always)]
    fn name(&self) -> &'static str {
        self.name
    }

    #[inline(always)]
    fn access(&self) -> &AccessDescriptor {
        &self.access
    }

    fn initialize(&mut self, store: &Store) -> Result<(), StateError> {
        ensure_query(&mut self.query, &self.filter, store)
    }

    fn prepare_parallel<'w>(
        &'w mut self,
        ctx: PrepareContext<'w>,
    ) -> Result<ParallelJob<'w>, StateError> {
        let stage_epoch = ctx.stage_epoch();
        let store = ctx.into_store();
        Q::validate_shape()?;
        if let Some(ep) = self.last_run_epoch {
            self.filter.update_since_epoch(ep);
            if let Some(q) = &mut self.query {
                q.set_filter_since_epoch(ep);
            }
        }
        ensure_query(&mut self.query, &self.filter, store)?;
        self.last_run_epoch = Some(stage_epoch);
        let this: &'w Self = self;
        let Some(query) = this.query.as_ref() else {
            return Ok(Box::new(|| Ok(Commands::new())));
        };
        let bindings = query.resolved_bindings_arc(store)?;
        let tasks = crate::storage::query::spec::iter::prepare_par_matched_mut_tasks::<Q, F>(
            store,
            &bindings,
            &this.filter,
            stage_epoch,
        );
        let func = &this.func;
        Ok(Box::new(move || {
            let mut merged = Commands::new();
            for mut chunk in tasks.execute_with_state::<Commands>(&|item, cmds| func(item, cmds)) {
                merged.append(&mut chunk);
            }
            Ok(merged)
        }))
    }
}

// ---------------------------------------------------------------------------
// 4. 可进入并行 Stage 的连续切片查询系统 (`ParChunkQuerySystem`)
// ---------------------------------------------------------------------------

/// 并行连续切片查询系统：闭包接收 `Q::Chunk<'a>`（例如 `(&[EntityId], (&mut [Pos], &[Vel]))`）与本子块的 [`Commands`]。
///
/// - 专为 SIMD 向量化与批处理热循环设计，消除逐行函数调用开销；
/// - 若配置了行级过滤器（如 `ChangedSince<T>`、`WithFlag<FLAG>`），底层自动将掩码连续 `1` 区间
///   （Contiguous Mask Runs）切分为紧致连续切片喂给闭包，兼顾行级过滤与连续内存访问。
pub struct ParChunkQuerySystem<Q, F, Func>
where
    Q: QueryMutShape,
    F: QueryFilter + Clone,
{
    name: &'static str,
    filter: F,
    query: Option<Query<Q, F>>,
    func: Func,
    access: AccessDescriptor,
    last_run_epoch: Option<u64>,
}

impl<Q, F, Func> ParChunkQuerySystem<Q, F, Func>
where
    Q: QueryMutShape,
    F: QueryFilter + Clone + Send + Sync + 'static,
    Func: for<'a> Fn(Q::Chunk<'a>, &mut Commands) + Send + Sync + 'static,
{
    pub fn new(name: &'static str, filter: F, func: Func) -> Self {
        let access = deduce_access::<Q, F>(&filter);
        Self {
            name,
            filter,
            query: None,
            func,
            access,
            last_run_epoch: None,
        }
    }

    /// 声明该系统写入的行级标志位掩码
    #[inline]
    pub fn with_write_flag(mut self, flag_mask: u8) -> Self {
        self.access.writes_flags |= flag_mask;
        self
    }

    /// 可变访问描述符
    #[inline]
    pub fn access_mut(&mut self) -> &mut AccessDescriptor {
        &mut self.access
    }
}

impl<Q, F, Func> System for ParChunkQuerySystem<Q, F, Func>
where
    Q: QueryMutShape,
    F: QueryFilter + Clone + Send + Sync + 'static,
    Func: for<'a> Fn(Q::Chunk<'a>, &mut Commands) + Send + Sync + 'static,
{
    #[inline(always)]
    fn name(&self) -> &'static str {
        self.name
    }

    #[inline(always)]
    fn access(&self) -> &AccessDescriptor {
        &self.access
    }

    fn initialize(&mut self, store: &Store) -> Result<(), StateError> {
        ensure_query(&mut self.query, &self.filter, store)
    }

    fn prepare_parallel<'w>(
        &'w mut self,
        ctx: PrepareContext<'w>,
    ) -> Result<ParallelJob<'w>, StateError> {
        let stage_epoch = ctx.stage_epoch();
        let store = ctx.into_store();
        Q::validate_shape()?;
        if let Some(ep) = self.last_run_epoch {
            self.filter.update_since_epoch(ep);
            if let Some(q) = &mut self.query {
                q.set_filter_since_epoch(ep);
            }
        }
        ensure_query(&mut self.query, &self.filter, store)?;
        self.last_run_epoch = Some(stage_epoch);
        let this: &'w Self = self;
        let Some(query) = this.query.as_ref() else {
            return Ok(Box::new(|| Ok(Commands::new())));
        };
        let bindings = query.resolved_bindings_arc(store)?;
        let tasks = crate::storage::query::spec::iter::prepare_par_matched_mut_tasks::<Q, F>(
            store,
            &bindings,
            &this.filter,
            stage_epoch,
        );
        let func = &this.func;
        Ok(Box::new(move || {
            let mut merged = Commands::new();
            for mut chunk in tasks.execute_chunks_with_state::<Commands>(&|slice_chunk, cmds| {
                func(slice_chunk, cmds)
            }) {
                merged.append(&mut chunk);
            }
            Ok(merged)
        }))
    }
}

// ---------------------------------------------------------------------------
// 5. IntoSystem 自动实现与构造函数
// ---------------------------------------------------------------------------

pub struct ExclusiveMarker;
pub struct QueryMarker<Q, F>(PhantomData<(Q, F)>);
pub struct SystemMarker;

impl<S: System> IntoSystem<SystemMarker> for S {
    type System = S;

    #[inline(always)]
    fn into_system(self) -> Self::System {
        self
    }
}

impl<Func> IntoSystem<ExclusiveMarker> for Func
where
    Func: FnMut(&mut Store) + Send + 'static,
{
    type System = ExclusiveFunctionSystem<Func>;

    fn into_system(self) -> Self::System {
        ExclusiveFunctionSystem::new(std::any::type_name::<Func>(), self)
    }
}

/// 辅助函数：快速创建具名独占系统
#[inline]
pub fn exclusive_system(
    name: &'static str,
    f: impl FnMut(&mut Store) + Send + 'static,
) -> ExclusiveFunctionSystem<impl FnMut(&mut Store) + Send + 'static> {
    ExclusiveFunctionSystem::new(name, f)
}

/// 创建带自动类型推导的单 Query 独占系统
pub fn query_system<Q, F, Func>(
    name: &'static str,
    filter: F,
    func: Func,
) -> QuerySystem<Q, F, Func>
where
    Q: QueryMutShape,
    F: QueryFilter + Clone,
    Func: FnMut(&mut Query<Q, F>, &mut Store) + Send + 'static,
{
    QuerySystem::new(name, filter, func)
}

/// 创建无过滤条件的通用 Query 独占系统
pub fn simple_query_system<Q, Func>(
    name: &'static str,
    func: Func,
) -> QuerySystem<Q, All, Func>
where
    Q: QueryMutShape,
    Func: FnMut(&mut Query<Q, All>, &mut Store) + Send + 'static,
{
    QuerySystem::new(name, All, func)
}

/// 创建带过滤条件、可进入并行 Stage 的逐行查询系统
pub fn par_query_system<Q, F, Func>(
    name: &'static str,
    filter: F,
    func: Func,
) -> ParQuerySystem<Q, F, Func>
where
    Q: QueryMutShape,
    F: QueryFilter + Clone + Send + Sync + 'static,
    Func: for<'a> Fn(Q::Item<'a>, &mut Commands) + Send + Sync + 'static,
{
    ParQuerySystem::new(name, filter, func)
}

/// 创建无过滤条件、可进入并行 Stage 的逐行查询系统
pub fn simple_par_query_system<Q, Func>(
    name: &'static str,
    func: Func,
) -> ParQuerySystem<Q, All, Func>
where
    Q: QueryMutShape,
    Func: for<'a> Fn(Q::Item<'a>, &mut Commands) + Send + Sync + 'static,
{
    ParQuerySystem::new(name, All, func)
}

/// 创建带过滤条件、可进入并行 Stage 的连续切片查询系统（SIMD / 向量化友好）
pub fn par_chunk_query_system<Q, F, Func>(
    name: &'static str,
    filter: F,
    func: Func,
) -> ParChunkQuerySystem<Q, F, Func>
where
    Q: QueryMutShape,
    F: QueryFilter + Clone + Send + Sync + 'static,
    Func: for<'a> Fn(Q::Chunk<'a>, &mut Commands) + Send + Sync + 'static,
{
    ParChunkQuerySystem::new(name, filter, func)
}

/// 创建无过滤条件、可进入并行 Stage 的连续切片查询系统（SIMD / 向量化友好）
pub fn simple_par_chunk_query_system<Q, Func>(
    name: &'static str,
    func: Func,
) -> ParChunkQuerySystem<Q, All, Func>
where
    Q: QueryMutShape,
    Func: for<'a> Fn(Q::Chunk<'a>, &mut Commands) + Send + Sync + 'static,
{
    ParChunkQuerySystem::new(name, All, func)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::id::ComponentTypeId;
    use crate::storage::query::spec::{With, WithFlag, Without};

    #[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
    struct Pos {
        x: f32,
        y: f32,
    }

    #[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
    struct Vel {
        dx: f32,
        dy: f32,
    }

    #[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
    struct Frozen(bool);

    #[test]
    fn test_query_system_access_deduction() {
        let sys = simple_query_system::<(&mut Pos, &Vel), _>("movement", |q, store| {
            q.for_each_mut(store, |(_e, (p, v))| {
                p.x += v.dx;
                p.y += v.dy;
            })
            .unwrap();
        });

        assert_eq!(sys.name(), "movement");
        // 闭包持有 &mut Store，无法静态约束其访问范围 → 恒为独占
        assert!(sys.access().exclusive);
        assert_eq!(sys.access().writes.len(), 1); // Pos
        assert_eq!(sys.access().reads.len(), 1);  // Vel
        assert!(sys.access().writes.contains(&ComponentTypeId::of::<Pos>()));
        assert!(sys.access().reads.contains(&ComponentTypeId::of::<Vel>()));
        assert!(sys.access().with_components.contains(&ComponentTypeId::of::<Pos>()));
        assert!(sys.access().with_components.contains(&ComponentTypeId::of::<Vel>()));
    }

    #[test]
    fn test_par_query_system_access_includes_filter_reads_and_predicates() {
        let filter = (
            crate::storage::query::spec::ChangedSince::<Frozen>::new(All, 0),
            Without::<Vel>::new(All),
            WithFlag::<2>::new(All),
        );
        let sys = par_query_system::<&mut Pos, _, _>("movement", filter, |(_e, p), _cmds| {
            p.x += 1.0;
        });

        assert!(!sys.access().exclusive);
        assert!(sys.access().writes.contains(&ComponentTypeId::of::<Pos>()));
        // P0-4：行级过滤器读取的列必须登记进读集
        assert!(sys.access().reads.contains(&ComponentTypeId::of::<Frozen>()));
        // 原型必选/排除集合与行标志位掩码自动推导
        assert!(sys.access().with_components.contains(&ComponentTypeId::of::<Pos>()));
        assert!(sys.access().without_components.contains(&ComponentTypeId::of::<Vel>()));
        assert_eq!(sys.access().reads_flags, 1u8 << 2);

        let sys_with_vel = par_query_system::<&mut Pos, _, _>(
            "vel_movement",
            With::<Vel>::new(All),
            |(_e, p), _cmds| {
                p.x += 1.0;
            },
        );
        assert!(sys.access().is_archetype_disjoint(sys_with_vel.access()));
        assert!(!sys.access().conflicts_with(sys_with_vel.access()));
    }

    #[test]
    fn test_exclusive_function_system_always_exclusive() {
        let sys = exclusive_system("spawn", |_s: &mut Store| {});
        assert!(sys.access().exclusive);
    }

    #[test]
    fn test_optional_query_system_access_deduction() {
        let sys = simple_par_query_system::<(&mut Pos, Option<&mut Vel>), _>(
            "optional_movement",
            |(_e, (p, v)), _cmds| {
                p.x += 1.0;
                if let Some(vel) = v {
                    vel.dx += 1.0;
                }
            },
        );
        assert!(!sys.access().exclusive);
        assert!(sys.access().writes.contains(&ComponentTypeId::of::<Pos>()));
        assert!(sys.access().writes.contains(&ComponentTypeId::of::<Vel>()));
        // Pos is required, Vel is optional -> Vel must NOT be in with_components!
        assert!(sys.access().with_components.contains(&ComponentTypeId::of::<Pos>()));
        assert!(!sys.access().with_components.contains(&ComponentTypeId::of::<Vel>()));
    }
}
