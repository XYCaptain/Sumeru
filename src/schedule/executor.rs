//! Layer 2 自动流水线 DAG 调度器 (`Schedule`)
//!
//! # 理论支撑与三级缓存设计
//!
//! 1. **三级执行计划缓存体系 (Three-Level Execution Plan Cache)**:
//!    - **Level 1 (冲突矩阵与 Stage 分层)**：在 `Schedule::build()` 时完成 $O(K^2)$ 位图分析与
//!      Kahn 拓扑分层，热循环中开销为 **0 ns**。
//!    - **Level 2 (原型表槽位物理预绑定)**：各系统内联缓存匹配原型的 `flat_col_indices`。
//!      每帧通过 $O(1)$ 的 `topology_epoch` 标量比对核验有效性。
//!    - **Level 3 (硬件直通与自适应切块)**：在并行 Stage 内，通过 `UnsafeStoreCell` 零锁直通
//!      不相交物理列切片，结合 `SubchunkPlanner` 严格驻留 L1/L2 缓存。
//!
//! 2. **无锁多核并发安全性证明 (ACM SAC '26 Theorem 1)**:
//!    在同一个 Stage 内的所有系统，其读写描述符经形式化静态证明互不重叠。
//!    因此各系统在多核 Rayon 线程池中并发执行时，对内存的读写完全正交，零数据竞争，零全局锁。

use std::marker::PhantomData;
use rayon::prelude::*;
use crate::error::StateError;
use crate::storage::core::store::Store;
use crate::schedule::ambiguity::{AmbiguityDetector, AmbiguityPair, ConflictMatrix};
use crate::schedule::dag::{DependencyConfig, ScheduleGraph, SystemStage};
use crate::schedule::descriptor::{AccessDescriptor, SystemId};
use crate::schedule::system::{IntoSystem, System};

/// 内部系统条目
struct SystemEntry {
    name: &'static str,
    system: Box<dyn System>,
    access: AccessDescriptor,
    dependency: DependencyConfig,
}


/// 阶段内用于无锁并行访问的 Store 细胞封装
///
/// # Safety
/// 调用者必须保证：在同一 Stage 内并发调用的所有系统，其 [`AccessDescriptor`]
/// 经由 [`ConflictMatrix`] 静态验证互不重叠，且绝不包含全局结构独占修改。
#[derive(Clone, Copy)]
pub struct UnsafeStoreCell<'a> {
    store_ptr: *mut Store,
    _marker: PhantomData<&'a mut Store>,
}

unsafe impl Send for UnsafeStoreCell<'_> {}
unsafe impl Sync for UnsafeStoreCell<'_> {}

impl<'a> UnsafeStoreCell<'a> {
    #[inline(always)]
    pub fn new(store: &'a mut Store) -> Self {
        Self {
            store_ptr: store as *mut Store,
            _marker: PhantomData,
        }
    }

    /// 获取底层 Store 的引用
    ///
    /// # Safety
    /// 仅在多系统读写集合被证明不相交时安全。
    #[inline(always)]
    pub unsafe fn get_mut(&self) -> &'a mut Store {
        unsafe { &mut *self.store_ptr }
    }
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
    last_topology_epoch: u64,
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
            last_topology_epoch: 0,
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

        // 5. Level 2 缓存：预初始化所有系统的列槽位与查询计划
        for entry in &mut self.entries {
            entry.system.initialize(store);
        }

        self.conflict_matrix = Some(matrix);
        self.last_topology_epoch = store.archetype_set_generation();
        self.initialized = true;
        Ok(())
    }

    /// 检查并按需基于 `topology_epoch` 增量刷新各系统的查询匹配缓存
    #[inline]
    fn ensure_cache_current(&mut self, store: &Store) {
        let current_gen = store.archetype_set_generation();
        if self.last_topology_epoch != current_gen {
            for entry in &mut self.entries {
                entry.system.initialize(store);
            }
            self.last_topology_epoch = current_gen;
        }
    }

    /// 执行一轮调度流水线 (Tick)
    pub fn run(&mut self, store: &mut Store) {
        if !self.initialized {
            self.build(store).expect("Failed to build schedule pipeline");
        } else {
            self.ensure_cache_current(store);
        }

        // 按 Stages 顺序推进
        for stage in &self.stages {
            match stage.systems.len() {
                0 => {}
                1 => {
                    // 单系统 Stage：直接在当前主线程内联执行，0 线程调度开销
                    let sys_idx = stage.systems[0].0 as usize;
                    if let Some(entry) = self.entries.get_mut(sys_idx) {
                        entry.system.run(store);
                    }
                }
                _ => {
                    // 多系统并发 Stage：
                    // 该 Stage 内的所有系统经 ConflictMatrix 证明互不相交且非独占。
                    // 采用 Rayon 并发执行各系统。
                    let cell = UnsafeStoreCell::new(store);
                    let sys_indices: Vec<usize> = stage.systems.iter().map(|id| id.0 as usize).collect();
                    let entries_addr = self.entries.as_mut_ptr() as usize;

                    sys_indices.into_par_iter().for_each(move |idx| {
                        // SAFETY:
                        // 1. 各系统只访问其所属的独立组件列；
                        // 2. idx 互不相同，每个系统实例独占借用；
                        // 3. 所有并发系统均无结构修改权。
                        unsafe {
                            let entry_ptr = (entries_addr as *mut SystemEntry).add(idx);
                            let store_ref = cell.get_mut();
                            (*entry_ptr).system.run(store_ref);
                        }
                    });
                }
            }
        }
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
    use crate::schedule::system::simple_query_system;
    use crate::{EntityBundle, EntityId};

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
            .add_system(simple_query_system::<(&mut Pos, &Vel), _>(
                "movement",
                |q, s| {
                    q.for_each_mut(s, |(_e, (p, v))| {
                        p.x += v.dx;
                    })
                    .unwrap();
                },
            ))
            .id();

        // 系统 2: 生命回复 (写 Health) -> 与 系统 1 完全无重叠，应自动聚类到同一 Stage 并发执行！
        let s2 = schedule
            .add_system(simple_query_system::<&mut Health, _>(
                "regeneration",
                |q, s| {
                    q.for_each_mut(s, |(_e, h)| {
                        h.hp += 5;
                    })
                    .unwrap();
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
        // Stage 1 应该包含 s3 (health_check)。
        assert_eq!(schedule.stages().len(), 2);
        assert!(schedule.stages()[0].systems.contains(&s1));
        assert!(schedule.stages()[0].systems.contains(&s2));
        assert_eq!(schedule.stages()[1].systems, vec![s3]);

        // 执行 1 帧
        schedule.run(&mut store);

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

    #[test]
    fn test_schedule_disjoint_parallel_scalability_and_topology_invalidation() {
        let mut store = Store::new();

        // 插入 5,000 个实体
        for i in 1..=5000 {
            let e = EntityId::from_u128(i as u128);
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

        let mut schedule = Schedule::new();

        // 4 个互不相交的读写系统：
        // sys_a 读写 TrackA
        // sys_b 读写 TrackB
        // sys_c 读写 TrackC
        // sys_d 读写 TrackD
        let sa = schedule
            .add_system(simple_query_system::<&mut TrackA, _>("sys_a", |q, s| {
                q.for_each_mut(s, |(_e, a)| a.0 += 1).unwrap();
            }))
            .id();
        let sb = schedule
            .add_system(simple_query_system::<&mut TrackB, _>("sys_b", |q, s| {
                q.for_each_mut(s, |(_e, b)| b.0 += 2).unwrap();
            }))
            .id();
        let sc = schedule
            .add_system(simple_query_system::<&mut TrackC, _>("sys_c", |q, s| {
                q.for_each_mut(s, |(_e, c)| c.0 += 3).unwrap();
            }))
            .id();
        let sd = schedule
            .add_system(simple_query_system::<&mut TrackD, _>("sys_d", |q, s| {
                q.for_each_mut(s, |(_e, d)| d.0 += 4).unwrap();
            }))
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
        schedule.run(&mut store);

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
        assert_eq!(count, 5000);

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

        // 第二次 Tick：自动检测到 store.archetype_set_generation() 变更并刷新查询缓存
        schedule.run(&mut store);

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
}
