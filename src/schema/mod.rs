//! 统一状态模式与契约定义模块 (Schema & Contracts)
//!
//! 提供物理存储所需的数据布局定义、单写多读访问屏障计划、
//! 领域类型与物理存储类别 (StateClass) 绑定，以及跨域 Checkpoint 校验契约。
//! 纯粹表达数据语义，严格不包含任何 P-DEVS 推进协议、时钟或 Actor 邮箱。

pub mod component;
pub mod domain;
pub mod model;

pub use component::ComponentSchema;
pub use domain::{DomainTypeDefinition, SchemaStoreBinding};
pub use model::{BehaviorAccessPlan, ModelSchema};

/// 稳定 64 位 FNV-1a 哈希算法，用于生成平台无关的 Schema/Codec 摘要
#[inline]
pub(crate) fn stable_digest(value: &str) -> u64 {
    value
        .as_bytes()
        .iter()
        .fold(0xcbf29ce484222325, |hash, byte| {
            (hash ^ u64::from(*byte)).wrapping_mul(0x100000001b3)
        })
}
