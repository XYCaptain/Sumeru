//! Layer 2 自动流水线调度器与 DAG 执行引擎 (Schedule & Pipeline Orchestrator)
//!
//! # 核心能力
//!
//! 1. **全自动读写权限推导 (`AccessDescriptor`)**:
//!    系统从查询形态 `Query<Q, F>` 中零反射、零注解自动推导读写组件集合。
//!
//! 2. **$O(K^2)$ 静态冲突判定与歧义检测 (`AmbiguityDetector`)**:
//!    构建期单次判定，热循环 0 ns 判定开销。
//!
//! 3. **拓扑分层与最大并行度 (`ScheduleGraph` & `SystemStage`)**:
//!    自动将无冲突系统聚类至同一 Stage 进行多核 Rayon 并发执行，有冲突系统自动序列化。
//!    并行 Stage 采用“主线程准备 → 无 `Store` 作业并行 → `Commands` 同步点”两阶段模型，
//!    独占系统（`&mut Store`）恒单独占用一个 Stage。
//!
//! 4. **三级缓存体系**:
//!    - Level 1: 冲突矩阵与拓扑 Stage 缓存；
//!    - Level 2: `topology_epoch` 校验的原型列物理槽位预绑定；
//!    - Level 3: 硬件直通与 `SubchunkPlanner` 缓存驻留。

pub mod ambiguity;
pub mod commands;
pub mod dag;
pub mod descriptor;
pub mod executor;
pub mod system;

pub use self::ambiguity::{AmbiguityDetector, AmbiguityPair, ConflictMatrix};
pub use self::commands::Commands;
pub use self::dag::{DependencyConfig, ScheduleGraph, SystemStage};
pub use self::descriptor::{AccessDescriptor, SystemId};
pub use self::executor::{Schedule, SystemConfigHandle};
pub use self::system::{
    exclusive_system, par_chunk_query_system, par_query_system, query_system,
    simple_par_chunk_query_system, simple_par_query_system, simple_query_system,
    ExclusiveFunctionSystem, ExclusiveMarker, IntoSystem, ParChunkQuerySystem, ParQuerySystem,
    ParallelJob, PrepareContext, QueryMarker, QuerySystem, System,
};
