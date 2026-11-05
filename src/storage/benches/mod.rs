//! 包内底层私有微基准套件 (In-Crate Differential Diagnostic Benchmark Suites)
//!
//! # 定位与分工说明
//! - **本模块 (`src/storage/benches/`)**：专门用于纳秒级测量和差分对比底层**非公开物理字段与未导出寻址路径**
//!   （如 Layer A 全路由哈希检索 vs Layer B 本地槽位直查 vs Layer C 预绑定切片底线硬件极限）。
//! - **外部基准 (`tests/benchmarks/`)**：负责面向所有公开 API（如 `Query`、`Store`、`CommandBuffer`）
//!   进行宏观工作负载与端到端系统级性能采样。

#[cfg(test)]
mod addressing_benchmark;
#[cfg(test)]
mod point_write_benchmark;
