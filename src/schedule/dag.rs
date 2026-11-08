//! 依赖有向无环图 (DAG) 与 Stages 拓扑分层构建器
//!
//! # 理论支撑与算法设计
//!
//! 1. **拓扑分层与最大并行集 (Kahn's Topological Batching)**:
//!    系统之间的 `.before()` / `.after()` 关系构成有向图。
//!    在传统的顺序调度中，拓扑排序输出的是全序线性序列。
//!    而在现代流水线调度中，算法将图划分为若干个时序离散的阶段（Stages）。
//!
//! 2. **Stage 内冲突自由不变量 (Intra-Stage Conflict-Free Invariant)**:
//!    对于任意一个 Stage $k$，其中包含的系统子集 $\{S_{k,1}, S_{k,2}, \dots, S_{k,m}\}$
//!    必须满足两两无冲突（通过 [`ConflictMatrix`] 静态验证）：
//!    $$\forall i \neq j,\quad \text{has\_conflict}(S_{k,i}, S_{k,j}) = \text{false}$$
//!    若存在两个可执行系统互相冲突，拓扑规划器自动将其切分至后续 Stage，
//!    在彻底杜绝数据竞争的同时最大化同一 Stage 内的多核并发度。

use std::collections::{BTreeSet, VecDeque};
use crate::error::StateError;
use crate::schedule::ambiguity::ConflictMatrix;
use crate::schedule::descriptor::SystemId;

/// 系统在 DAG 中的依赖边配置
#[derive(Debug, Clone, Default)]
pub struct DependencyConfig {
    pub before: BTreeSet<SystemId>,
    pub after: BTreeSet<SystemId>,
}

/// 执行阶段（Stage）：包含一组互不冲突、可安全并发执行的系统
#[derive(Debug, Clone)]
pub struct SystemStage {
    pub systems: Vec<SystemId>,
}

/// 拓扑调度图
pub struct ScheduleGraph {
    num_systems: usize,
    /// 邻接表：`edges[u]` 包含所有必须在 `u` 之后执行的系统 `v`（即 `u -> v`）
    edges: Vec<BTreeSet<usize>>,
    /// 入度表
    in_degrees: Vec<usize>,
}

impl ScheduleGraph {
    /// 构建拓扑图
    pub fn build(
        num_systems: usize,
        dependencies: &[DependencyConfig],
    ) -> Result<Self, StateError> {
        let mut edges = vec![BTreeSet::new(); num_systems];
        let mut in_degrees = vec![0usize; num_systems];

        for (u_idx, dep) in dependencies.iter().enumerate() {
            // u 在 dep.before 之前：u -> v
            for &v_id in &dep.before {
                let v_idx = v_id.0 as usize;
                if v_idx < num_systems && edges[u_idx].insert(v_idx) {
                    in_degrees[v_idx] += 1;
                }
            }
            // u 在 dep.after 之后：v -> u
            for &v_id in &dep.after {
                let v_idx = v_id.0 as usize;
                if v_idx < num_systems && edges[v_idx].insert(u_idx) {
                    in_degrees[u_idx] += 1;
                }
            }
        }

        // 环检测 (Cycle Detection via Kahn's algorithm)
        let mut queue = VecDeque::new();
        let mut temp_in_degrees = in_degrees.clone();
        for (i, &deg) in temp_in_degrees.iter().enumerate() {
            if deg == 0 {
                queue.push_back(i);
            }
        }

        let mut visited_count = 0;
        while let Some(u) = queue.pop_front() {
            visited_count += 1;
            for &v in &edges[u] {
                temp_in_degrees[v] -= 1;
                if temp_in_degrees[v] == 0 {
                    queue.push_back(v);
                }
            }
        }

        if visited_count != num_systems {
            return Err(StateError::BatchConflict(
                "Dependency cycle detected in system schedule graph".to_string(),
            ));
        }

        Ok(Self {
            num_systems,
            edges,
            in_degrees,
        })
    }

    /// 计算可达性传递闭包矩阵（`reachable[u][v]` 为 true 表示从 `u` 可到达 `v`）
    pub fn compute_reachability(&self) -> Vec<Vec<bool>> {
        let n = self.num_systems;
        let mut reachable = vec![vec![false; n]; n];

        for u in 0..n {
            for &v in &self.edges[u] {
                reachable[u][v] = true;
            }
        }

        // Floyd-Warshall 传递闭包
        for k in 0..n {
            for i in 0..n {
                for j in 0..n {
                    if reachable[i][k] && reachable[k][j] {
                        reachable[i][j] = true;
                    }
                }
            }
        }

        reachable
    }

    /// 基于依赖入度与冲突矩阵，将系统拓扑聚类为若干个互不冲突的并发 Stages
    pub fn partition_into_stages(&self, conflict_matrix: &ConflictMatrix) -> Vec<SystemStage> {
        let n = self.num_systems;
        if n == 0 {
            return Vec::new();
        }

        let mut in_degrees = self.in_degrees.clone();
        let mut remaining = n;
        let mut stages = Vec::new();
        let mut scheduled = vec![false; n];

        while remaining > 0 {
            // 候选就绪系统（入度为 0 且尚未被调度）
            let mut ready_candidates = Vec::new();
            for i in 0..n {
                if !scheduled[i] && in_degrees[i] == 0 {
                    ready_candidates.push(i);
                }
            }

            if ready_candidates.is_empty() {
                break;
            }

            // 在当前就绪候选中，贪心提取互不冲突的最大独立子集组成单个 Stage
            let mut current_stage_systems = Vec::new();

            for &cand in &ready_candidates {
                let conflicts_with_current = current_stage_systems.iter().any(|&existing| {
                    conflict_matrix.has_conflict(cand, existing)
                });

                if !conflicts_with_current {
                    current_stage_systems.push(cand);
                    scheduled[cand] = true;
                    remaining -= 1;
                }
            }

            // 当前 Stage 内的系统完成，释放下游系统的入度
            for &u in &current_stage_systems {
                for &v in &self.edges[u] {
                    if in_degrees[v] > 0 {
                        in_degrees[v] -= 1;
                    }
                }
            }

            stages.push(SystemStage {
                systems: current_stage_systems.into_iter().map(|idx| SystemId(idx as u32)).collect(),
            });
        }

        stages
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schedule::descriptor::AccessDescriptor;
    use crate::id::ComponentTypeId;

    #[test]
    fn test_dag_cycle_detection() {
        let mut deps = vec![DependencyConfig::default(); 2];
        // 0 before 1 (0 -> 1)
        deps[0].before.insert(SystemId(1));
        // 1 before 0 (1 -> 0) => cycle!
        deps[1].before.insert(SystemId(0));

        let result = ScheduleGraph::build(2, &deps);
        assert!(result.is_err());
    }

    #[test]
    fn test_dag_partition_into_stages() {
        let cid_a = ComponentTypeId::from_name("A");
        let cid_b = ComponentTypeId::from_name("B");

        // 系统 0: 写 A
        let d0 = AccessDescriptor::new().with_write(cid_a);
        // 系统 1: 读 A (与 0 冲突)
        let d1 = AccessDescriptor::new().with_read(cid_a);
        // 系统 2: 读 B (与 0, 1 均不冲突)
        let d2 = AccessDescriptor::new().with_read(cid_b);

        let descs = vec![d0, d1, d2];
        let matrix = ConflictMatrix::build(&descs);

        // 无显式 DAG 边
        let deps = vec![DependencyConfig::default(); 3];
        let graph = ScheduleGraph::build(3, &deps).unwrap();

        let stages = graph.partition_into_stages(&matrix);
        // 0 和 2 不冲突，可以在 Stage 0 同时运行；
        // 1 与 0 冲突，被自动推迟到 Stage 1！
        assert_eq!(stages.len(), 2);
        assert!(stages[0].systems.contains(&SystemId(0)));
        assert!(stages[0].systems.contains(&SystemId(2)));
        assert_eq!(stages[1].systems, vec![SystemId(1)]);
    }
}
