//! Layer 2 自动流水线 DAG 调度器 (`Schedule`)
//!
//! # 理论支撑与三级缓存设计
//!
//! 1. **三级执行计划缓存体系 (Three-Level Execution Plan Cache)**:
//!    - **Level 1 (冲突矩阵与 Stage 分层)**：在 `Schedule::build()` 时完成 $O(K^2)$ 位图分析与
//!      Kahn 拓扑分层，热循环中开销为 **0 ns**。
//!    - **Level 2 (原型表槽位物理预绑定)**：各系统内联缓存匹配原型的 `flat_col_indices`。
//!      每个系统在**自身运行/准备前**通过 $O(1)$ 的 `topology_epoch` 标量比对核验有效性，
//!      因此前序 Stage（独占系统或命令同步点）新建的原型对后续 Stage 立即可见。
//!    - **Level 3 (硬件直通与自适应切块)**：并行 Stage 的执行阶段只持有不相交物理列切片的
//!      裸指针（不持有 `Store`），结合 `SubchunkPlanner` 严格驻留 L1/L2 缓存。
//!
//! 2. **两阶段并行 Stage（健全性证明, ACM SAC '26 Theorem 1）**:
//!    - **阶段 A（主线程，顺序）**：校验中毒状态，Stage 纪元只推进**一次**；
//!      对 Stage 内每个系统调用 [`System::prepare_parallel`]，完成全部 `Store` 元数据访问
//!      （L2 缓存刷新、写列 COW 解耦、Stage 纪元标记、行掩码求值）；
//!    - **阶段 B（Rayon，并行）**：各系统作业只访问其声明读写集对应的列缓冲区，
//!      读写集经 [`ConflictMatrix`] 静态证明两两不相交，零数据竞争、零全局锁；
//!    - **同步点（主线程）**：按系统注册序依次应用各系统的 [`Commands`]，
//!      结构修改结果与线程调度无关（确定性）。
//!
//! 独占系统（`access().exclusive`）恒单独占用一个 Stage，直接以 `&mut Store` 运行。

use rayon::prelude::*;
use crate::error::StateError;
use crate::storage::core::store::Store;
use crate::schedule::ambiguity::{AmbiguityDetector, AmbiguityPair, ConflictMatrix};
use crate::schedule::commands::Commands;
use crate::schedule::dag::{DependencyConfig, ScheduleGraph, SystemStage};
use crate::schedule::descriptor::{AccessDescriptor, SystemId};
use crate::schedule::system::{IntoSystem, ParallelJob, PrepareContext, System};

/// 内部系统条目
struct SystemEntry {
    name: &'static str,
    system: Box<dyn System>,
    access: AccessDescriptor,
    dependency: DependencyConfig,
}

/// 流水线调度配置构建句柄
pub struct SystemConfigHandle<'a> {
    schedule: &'a mut Schedule,
    id: SystemId,
}

impl<'a> SystemConfigHandle<'a> {
    /// 声明该系统必须在 `other` 之前执行
    pub fn before(self, other: SystemId) -> Self {
        if (self.id.0 as usize) < self.schedule.entries.len() {
            self.schedule.entries[self.id.0 as usize].dependency.before.insert(other);
        }
        self
    }

    /// 声明该系统必须在指定名称的系统之前执行
    pub fn before_named(self, other_name: &'static str) -> Self {
        if let Some(target) = self.schedule.find_system(other_name) {
            return self.before(target);
        }
        self
    }

    /// 声明该系统必须在 `other` 之后执行
    pub fn after(self, other: SystemId) -> Self {
        if (self.id.0 as usize) < self.schedule.entries.len() {
            self.schedule.entries[self.id.0 as usize].dependency.after.insert(other);
        }
        self
    }

    /// 声明该系统必须在指定名称的系统之后执行
    pub fn after_named(self, other_name: &'static str) -> Self {
        if let Some(target) = self.schedule.find_system(other_name) {
            return self.after(target);
        }
        self
    }

    /// 获取系统 ID
    #[inline(always)]
    pub fn id(&self) -> SystemId {
        self.id
    }
}

/// Layer 2 自动流水线 DAG 调度器
pub struct Schedule {
    entries: Vec<SystemEntry>,
    stages: Vec<SystemStage>,
    conflict_matrix: Option<ConflictMatrix>,
    ambiguities: Vec<AmbiguityPair>,
    initialized: bool,
}

impl Default for Schedule {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

impl Schedule {
    /// 创建空的调度流水线
    #[inline]
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
            stages: Vec::new(),
            conflict_matrix: None,
            ambiguities: Vec::new(),
            initialized: false,
        }
    }

    /// 向调度器注册一个系统
    pub fn add_system<M>(&mut self, system: impl IntoSystem<M>) -> SystemConfigHandle<'_> {
        let sys = system.into_system();
        let name = sys.name();
        let access = sys.access().clone();
        let id = SystemId(self.entries.len() as u32);

        self.entries.push(SystemEntry {
            name,
            system: Box::new(sys),
            access,
            dependency: DependencyConfig::default(),
        });
        self.initialized = false;

        SystemConfigHandle { schedule: self, id }
    }

    /// 根据名称查找系统 ID
    pub fn find_system(&self, name: &'static str) -> Option<SystemId> {
        self.entries
            .iter()
            .position(|e| e.name == name)
            .map(|idx| SystemId(idx as u32))
    }

    /// 系统总数
    #[inline(always)]
    pub fn system_count(&self) -> usize {
        self.entries.len()
    }

    /// 编译调度 DAG 与构建 Stages（若未显式调用，在第一次 `run` 时自动触发）
    pub fn build(&mut self, store: &Store) -> Result<(), StateError> {
        let n = self.entries.len();
        if n == 0 {
            self.stages.clear();
            self.initialized = true;
            return Ok(());
        }

        // 1. Level 1 缓存：构建两两冲突矩阵
        let descs: Vec<AccessDescriptor> = self.entries.iter().map(|e| e.access.clone()).collect();
        let names: Vec<&'static str> = self.entries.iter().map(|e| e.name).collect();
        let matrix = ConflictMatrix::build(&descs);

        // 2. 构建 DAG 并检测拓扑环
        let deps: Vec<DependencyConfig> = self.entries.iter().map(|e| e.dependency.clone()).collect();
        let graph = ScheduleGraph::build(n, &deps)?;

        // 3. 歧义检测 (Ambiguity Detection)
        let reachable = graph.compute_reachability();
        self.ambiguities = AmbiguityDetector::find_ambiguities(&matrix, &descs, &names, &reachable);

        // 4. 将系统拓扑聚类为互不冲突的 Stages
        self.stages = graph.partition_into_stages(&matrix);

        // 5. Level 2 缓存：预初始化所有系统的列槽位与查询计划（错误向上传播）
        for entry in &mut self.entries {
            entry.system.initialize(store)?;
        }

        self.conflict_matrix = Some(matrix);
        self.initialized = true;
        Ok(())
    }

    /// 执行一轮调度流水线 (Tick)
    ///
    /// 任一系统或命令同步点返回错误时立即中止本 Tick 并返回该错误。
    pub fn run(&mut self, store: &mut Store) -> Result<(), StateError> {
        if !self.initialized {
            self.build(store)?;
        }

        // 按 Stages 顺序推进
        for stage_idx in 0..self.stages.len() {
            let systems = &self.stages[stage_idx].systems;
            match systems.as_slice() {
                [] => {}
                [only] if self.entries[only.0 as usize].access.exclusive => {
                    // 独占 Stage：主线程以 &mut Store 直接执行
                    self.entries[only.0 as usize].system.run(store)?;
                }
                _ => {
                    let mut in_stage = vec![false; self.entries.len()];
                    for id in systems {
                        let idx = id.0 as usize;
                        if self.entries[idx].access.exclusive {
                            return Err(StateError::AccessDenied(format!(
                                "exclusive system `{}` cannot share a parallel stage",
                                self.entries[idx].name
                            )));
                        }
                        in_stage[idx] = true;
                    }
                    Self::run_parallel_stage(&mut self.entries, &in_stage, store)?;
                }
            }
        }
        Ok(())
    }

    /// 两阶段并行 Stage 执行（见模块文档）
    fn run_parallel_stage(
        entries: &mut [SystemEntry],
        in_stage: &[bool],
        store: &mut Store,
    ) -> Result<(), StateError> {
        // ---- 阶段 A：主线程顺序准备 ----
        store.check_poisoned()?;
        store.next_epoch()?;
        store.bump_epoch();
        let stage_epoch = store.epoch();

        let store_ptr: *mut Store = store;
        let mut jobs: Vec<ParallelJob<'_>> = Vec::new();
        for (idx, entry) in entries.iter_mut().enumerate() {
            if !in_stage[idx] {
                continue;
            }
            // SAFETY:
            // 1. 本循环在主线程顺序执行，每次 `prepare_parallel` 返回前独占使用该 `&mut Store`，
            //    任一时刻不存在两个同时活跃使用的 `&mut Store`；
            // 2. 返回的 `ParallelJob` 不持有 `Store`/`Archetype` 引用（`PrepareContext` 的 Store
            //    访问器为 crate 私有，作业只持有列缓冲区与实体切片的物理指针）；
            // 3. 后续系统的准备阶段只解耦/标记**其自身写集**的列，Stage 内读写集经冲突矩阵
            //    证明两两不相交，且不做任何结构修改，因此不会移动或改写先前作业持有的缓冲区；
            // 4. 所有作业在本函数返回前执行完毕并被消费，之后才重新使用 `store`。
            let ctx = PrepareContext::new(unsafe { &mut *store_ptr }, stage_epoch);
            jobs.push(entry.system.prepare_parallel(ctx)?);
        }

        // ---- 阶段 B：作业并行执行（结果保持系统注册序）----
        let results: Vec<Result<Commands, StateError>> = if jobs.len() <= 1 {
            jobs.into_iter().map(|job| job()).collect()
        } else {
            jobs.into_par_iter().map(|job| job()).collect()
        };

        // ---- 同步点：按系统注册序在主线程应用结构变更命令 ----
        let mut commands = Vec::with_capacity(results.len());
        for result in results {
            commands.push(result?);
        }
        for cmds in commands {
            if !cmds.is_empty() {
                cmds.apply(store)?;
            }
        }
        Ok(())
    }

    /// 获取编译生成的 Stages 列表
    pub fn stages(&self) -> &[SystemStage] {
        &self.stages
    }

    /// 获取发现的未定义顺序冲突歧义列表
    pub fn ambiguities(&self) -> &[AmbiguityPair] {
        &self.ambiguities
    }

    /// 获取冲突矩阵引用
    pub fn conflict_matrix(&self) -> Option<&ConflictMatrix> {
        self.conflict_matrix.as_ref()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schedule::system::{
        exclusive_system, par_query_system, simple_par_query_system, simple_query_system,
    };
    use crate::storage::query::spec::{All, ChangedSince};
    use crate::{EntityBundle, EntityId};
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    #[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
    struct Pos {
        x: i32,
    }

    #[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
    struct Vel {
        dx: i32,
    }

    #[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
    struct Health {
        hp: i32,
    }

    #[test]
    fn test_schedule_parallel_stages_and_execution() {
        let mut store = Store::new();

        // 创建测试实体
        for i in 1..=100 {
            let e = EntityId::from_u128(i as u128);
            let bundle = EntityBundle::new()
                .with(Pos { x: 10 })
                .unwrap()
                .with(Vel { dx: 2 })
                .unwrap()
                .with(Health { hp: 100 })
                .unwrap();
            store.create_entity(e, bundle).unwrap();
        }

        let mut schedule = Schedule::new();

        // 系统 1: 物理移动 (写 Pos, 读 Vel)
        let s1 = schedule
            .add_system(simple_par_query_system::<(&mut Pos, &Vel), _>(
                "movement",
                |(_e, (p, v)), _cmds| {
                    p.x += v.dx;
                },
            ))
            .id();

        // 系统 2: 生命回复 (写 Health) -> 与 系统 1 完全无重叠，应自动聚类到同一 Stage 并发执行！
        let s2 = schedule
            .add_system(simple_par_query_system::<&mut Health, _>(
                "regeneration",
                |(_e, h), _cmds| {
                    h.hp += 5;
                },
            ))
            .id();

        // 系统 3: 生命检查 (读 Health) -> 依赖 Health，应在系统 2 之后的 Stage 执行
        let s3 = schedule
            .add_system(simple_query_system::<&Health, _>(
                "health_check",
                |q, s| {
                    q.for_each(s, |(_e, h)| {
                        assert_eq!(h.hp, 105);
                    })
                    .unwrap();
                },
            ))
            .after(s2)
            .id();

        // 编译流水线
        schedule.build(&store).unwrap();

        // 验证拓扑分层：
        // Stage 0 应该包含 s1 (movement) 和 s2 (regeneration)，它们互不冲突；
        // Stage 1 应该包含 s3 (health_check，独占)。
        assert_eq!(schedule.stages().len(), 2);
        assert!(schedule.stages()[0].systems.contains(&s1));
        assert!(schedule.stages()[0].systems.contains(&s2));
        assert_eq!(schedule.stages()[1].systems, vec![s3]);
        // s3 声明了读集（Health）且有 after(s2) 约束；与 s1 组件不相交 → 无歧义
        assert!(schedule.ambiguities().is_empty(), "{:?}", schedule.ambiguities());

        // 执行 1 帧
        schedule.run(&mut store).unwrap();

        // 验证结果：所有实体的 Pos.x 从 10 变为 12，Health.hp 从 100 变为 105
        let q_verify = store.query::<(&Pos, &Health)>();
        let mut count = 0;
        for (_e, (p, h)) in q_verify.iter() {
            assert_eq!(p.x, 12);
            assert_eq!(h.hp, 105);
            count += 1;
        }
        assert_eq!(count, 100);
    }

    #[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
    struct TrackA(i32);
    #[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
    struct TrackB(i32);
    #[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
    struct TrackC(i32);
    #[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
    struct TrackD(i32);

    /// Miri 下缩小实体规模（解释执行很慢），仍保证多系统并行作业路径被覆盖
    const fn scaled(n: u128) -> u128 {
        if cfg!(miri) { if n > 64 { 64 } else { n } } else { n }
    }

    fn spawn_tracks(store: &mut Store, n: u128) {
        for i in 1..=n {
            let e = EntityId::from_u128(i);
            let b = EntityBundle::new()
                .with(TrackA(0))
                .unwrap()
                .with(TrackB(10))
                .unwrap()
                .with(TrackC(100))
                .unwrap()
                .with(TrackD(1000))
                .unwrap();
            store.create_entity(e, b).unwrap();
        }
    }

    #[test]
    fn test_schedule_disjoint_parallel_scalability_and_topology_invalidation() {
        let mut store = Store::new();

        // 插入 5,000 个实体
        let n = scaled(5000);
        spawn_tracks(&mut store, n);

        let mut schedule = Schedule::new();

        // 4 个互不相交的读写系统：
        // sys_a 读写 TrackA
        // sys_b 读写 TrackB
        // sys_c 读写 TrackC
        // sys_d 读写 TrackD
        let sa = schedule
            .add_system(simple_par_query_system::<&mut TrackA, _>("sys_a", |(_e, a), _c| a.0 += 1))
            .id();
        let sb = schedule
            .add_system(simple_par_query_system::<&mut TrackB, _>("sys_b", |(_e, b), _c| b.0 += 2))
            .id();
        let sc = schedule
            .add_system(simple_par_query_system::<&mut TrackC, _>("sys_c", |(_e, c), _c| c.0 += 3))
            .id();
        let sd = schedule
            .add_system(simple_par_query_system::<&mut TrackD, _>("sys_d", |(_e, d), _c| d.0 += 4))
            .id();

        // 编译流水线
        schedule.build(&store).unwrap();

        // 4 个系统互不冲突，拓扑分析器必须将它们合并在同一个并发 Stage (Stage 0)
        assert_eq!(schedule.stages().len(), 1);
        assert_eq!(schedule.stages()[0].systems.len(), 4);
        assert!(schedule.stages()[0].systems.contains(&sa));
        assert!(schedule.stages()[0].systems.contains(&sb));
        assert!(schedule.stages()[0].systems.contains(&sc));
        assert!(schedule.stages()[0].systems.contains(&sd));

        // 第一次 Tick
        schedule.run(&mut store).unwrap();

        // 验证 5,000 个实体全部被准确并发更新
        let q_check = store.query::<(&TrackA, &TrackB, &TrackC, &TrackD)>();
        let mut count = 0;
        for (_e, (a, b, c, d)) in q_check.iter() {
            assert_eq!(a.0, 1);
            assert_eq!(b.0, 12);
            assert_eq!(c.0, 103);
            assert_eq!(d.0, 1004);
            count += 1;
        }
        assert_eq!(count, n as usize);

        // 插入具有新原型（带额外组件）的实体，验证 topology_epoch 自适应缓存失效与增量重新绑定
        let e_new = EntityId::from_u128(99999);
        let b_new = EntityBundle::new()
            .with(TrackA(50))
            .unwrap()
            .with(TrackB(50))
            .unwrap()
            .with(TrackC(50))
            .unwrap()
            .with(TrackD(50))
            .unwrap()
            .with(Pos { x: 999 })
            .unwrap();
        store.create_entity(e_new, b_new).unwrap();

        // 第二次 Tick：各系统准备阶段自动检测到 archetype_set_generation 变更并刷新查询缓存
        schedule.run(&mut store).unwrap();

        let q_new_entity = store.query::<(&TrackA, &TrackB, &TrackC, &TrackD)>();
        let mut new_seen = false;
        for (e, (a, b, c, d)) in q_new_entity.iter() {
            if e == e_new {
                assert_eq!(a.0, 51);
                assert_eq!(b.0, 52);
                assert_eq!(c.0, 53);
                assert_eq!(d.0, 54);
                new_seen = true;
            }
        }
        assert!(new_seen, "Newly added archetype entity must be matched after cache update");
    }

    #[test]
    fn test_commands_applied_at_sync_point_deterministically() {
        let mut store = Store::new();
        let n = scaled(10_000);
        spawn_tracks(&mut store, n);

        let mut schedule = Schedule::new();
        // 并行系统中通过 Commands 延迟销毁 TrackA 为偶数编号的实体
        schedule.add_system(simple_par_query_system::<&TrackA, _>("cull", |(e, _a), cmds| {
            if e.to_u128() % 2 == 0 {
                cmds.despawn(e);
            }
        }));
        // 同 Stage 的另一个并行系统（写 TrackB），不受命令影响
        schedule.add_system(simple_par_query_system::<&mut TrackB, _>("bump_b", |(_e, b), _c| b.0 += 1));
        schedule.build(&store).unwrap();
        assert_eq!(schedule.stages().len(), 1);

        schedule.run(&mut store).unwrap();

        // 同步点之后：偶数实体已销毁，奇数实体的 TrackB 被更新
        assert_eq!(store.active_entities(), (n / 2) as usize);
        let mut seen = 0;
        for (e, b) in store.query::<&TrackB>().iter() {
            assert_eq!(e.to_u128() % 2, 1);
            assert_eq!(b.0, 11);
            seen += 1;
        }
        assert_eq!(seen, (n / 2) as usize);
    }

    #[test]
    fn test_parallel_stage_shares_single_epoch_and_changed_since() {
        let mut store = Store::new();
        let n = scaled(2_000);
        spawn_tracks(&mut store, n);
        let before = store.epoch();

        let mut schedule = Schedule::new();
        schedule.add_system(simple_par_query_system::<&mut TrackA, _>("a", |(_e, a), _c| a.0 += 1));
        schedule.add_system(simple_par_query_system::<&mut TrackB, _>("b", |(_e, b), _c| b.0 += 1));
        schedule.build(&store).unwrap();
        assert_eq!(schedule.stages().len(), 1);
        schedule.run(&mut store).unwrap();

        // 一个并行 Stage 只推进一次 Store 纪元
        assert_eq!(store.epoch(), before + 1);

        // 两个系统写入的列都可被 ChangedSince(before) 观察到
        let q_a = crate::storage::query::handle::QueryBuilder::<&TrackA, _>::with_filter(
            ChangedSince::<TrackA>::new(All, before),
        )
        .build(&store)
        .unwrap();
        let mut changed_a = 0;
        q_a.for_each(&store, |_| changed_a += 1).unwrap();
        assert_eq!(changed_a, n as usize);

        let q_b = crate::storage::query::handle::QueryBuilder::<&TrackB, _>::with_filter(
            ChangedSince::<TrackB>::new(All, before),
        )
        .build(&store)
        .unwrap();
        let mut changed_b = 0;
        q_b.for_each(&store, |_| changed_b += 1).unwrap();
        assert_eq!(changed_b, n as usize);

        // TrackC 未被写入 → ChangedSince(before) 为空
        let q_c = crate::storage::query::handle::QueryBuilder::<&TrackC, _>::with_filter(
            ChangedSince::<TrackC>::new(All, before),
        )
        .build(&store)
        .unwrap();
        let mut changed_c = 0;
        q_c.for_each(&store, |_| changed_c += 1).unwrap();
        assert_eq!(changed_c, 0);
    }

    #[test]
    fn test_exclusive_spawn_visible_to_next_stage_in_same_tick() {
        let mut store = Store::new();
        spawn_tracks(&mut store, 10);

        let mut schedule = Schedule::new();
        let spawner = schedule
            .add_system(exclusive_system("spawner", |s: &mut Store| {
                // 新原型（额外带 Pos）——本 Tick 内后续 Stage 必须可见
                let b = EntityBundle::new()
                    .with(TrackA(100))
                    .unwrap()
                    .with(Pos { x: 0 })
                    .unwrap();
                s.create_entity(EntityId::from_u128(777), b).unwrap();
            }))
            .id();
        let counter = Arc::new(AtomicUsize::new(0));
        let counter_in = Arc::clone(&counter);
        schedule
            .add_system(simple_par_query_system::<&mut TrackA, _>("bump_a", move |(_e, a), _c| {
                a.0 += 1;
                counter_in.fetch_add(1, Ordering::Relaxed);
            }))
            .after(spawner);
        schedule.build(&store).unwrap();
        assert_eq!(schedule.stages().len(), 2);

        schedule.run(&mut store).unwrap();
        assert_eq!(counter.load(Ordering::Relaxed), 11);
        let mut found = false;
        for (e, a) in store.query::<&TrackA>().iter() {
            if e == EntityId::from_u128(777) {
                assert_eq!(a.0, 101);
                found = true;
            }
        }
        assert!(found);
    }

    #[test]
    fn test_filter_reads_force_separate_stages() {
        let mut store = Store::new();
        spawn_tracks(&mut store, 10);

        let mut schedule = Schedule::new();
        // 系统 0：写 TrackB
        schedule.add_system(simple_par_query_system::<&mut TrackB, _>("write_b", |(_e, b), _c| b.0 += 1));
        // 系统 1：查询只写 TrackA，但行级过滤器读取 TrackB → 必须与系统 0 冲突（P0-4）
        schedule.add_system(par_query_system::<&mut TrackA, _, _>(
            "a_if_b_changed",
            ChangedSince::<TrackB>::new(All, 0),
            |(_e, a), _c| a.0 += 1,
        ));
        schedule.build(&store).unwrap();
        assert_eq!(schedule.stages().len(), 2);
        assert_eq!(schedule.ambiguities().len(), 1);
        schedule.run(&mut store).unwrap();
    }

    #[test]
    fn test_exclusive_systems_get_own_stage() {
        let mut store = Store::new();
        spawn_tracks(&mut store, 10);

        let mut schedule = Schedule::new();
        let e1 = schedule.add_system(exclusive_system("ex1", |_s: &mut Store| {})).id();
        let p1 = schedule
            .add_system(simple_par_query_system::<&mut TrackA, _>("p1", |(_e, a), _c| a.0 += 1))
            .id();
        let q1 = schedule
            .add_system(simple_query_system::<&mut TrackB, _>("q1", |q, s| {
                q.for_each_mut(s, |(_e, b)| b.0 += 1).unwrap();
            }))
            .id();
        schedule.build(&store).unwrap();

        for stage in schedule.stages() {
            if stage.systems.contains(&e1) || stage.systems.contains(&q1) {
                assert_eq!(stage.systems.len(), 1);
            }
        }
        assert!(schedule.stages().iter().any(|s| s.systems.contains(&p1)));
        schedule.run(&mut store).unwrap();
        for (_e, (a, b)) in store.query::<(&TrackA, &TrackB)>().iter() {
            assert_eq!(a.0, 1);
            assert_eq!(b.0, 11);
        }
    }

    #[test]
    fn test_command_error_aborts_tick() {
        let mut store = Store::new();
        spawn_tracks(&mut store, 4);

        let mut schedule = Schedule::new();
        schedule.add_system(simple_par_query_system::<&TrackA, _>("fail", |_item, cmds| {
            cmds.push(|_s| Err(StateError::AccessDenied("boom".into())));
        }));
        assert!(schedule.run(&mut store).is_err());
    }
}
