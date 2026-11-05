//! 原子组件内存布局与编码契约 (ComponentSchema)

use crate::context::ModelComponent;
use crate::id::{ComponentTypeId, SchemaVersion};
use serde::{Deserialize, Serialize};

/// 状态组件内存契约与编码定义 (ComponentSchema)。
///
/// 描述组件类型标识、版本、编码摘要，以及底层列式存储所必需的内存大小与对齐要求。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ComponentSchema {
    pub type_id: ComponentTypeId,
    pub type_name: String,
    pub version: SchemaVersion,
    pub codec_digest: u64,
    pub size: usize,
    pub alignment: usize,
}

impl ComponentSchema {
    /// 从强类型组件自动推导 ComponentSchema
    pub fn of<T: ModelComponent + Serialize>(version: SchemaVersion) -> Self {
        let type_name = std::any::type_name::<T>();
        Self {
            type_id: ComponentTypeId::from_name(type_name),
            type_name: type_name.to_owned(),
            version,
            codec_digest: super::stable_digest(&format!("{type_name}:{}", version.as_u32())),
            size: std::mem::size_of::<T>(),
            alignment: std::mem::align_of::<T>(),
        }
    }
}
