//! 领域类型定义与 Schema/Store Binding 契约 (DomainTypeDefinition, SchemaStoreBinding)

use crate::binding::StateClass;
use crate::id::{DomainTypeId, PersistedDomainTypeRef, SchemaVersion};
use serde::{Deserialize, Serialize};

/// 状态分类与底层存储/模式绑定描述 (SchemaStoreBinding)。
///
/// 每个状态组件声明且仅声明一个 StateClass 权威存储类别，
/// 支持复合领域模型绑定不同的 Store Shard。
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct SchemaStoreBinding {
    pub state_class: StateClass,
    pub schema_digest: u64,
}

impl SchemaStoreBinding {
    #[inline]
    pub const fn new(state_class: StateClass, schema_digest: u64) -> Self {
        Self {
            state_class,
            schema_digest,
        }
    }
}

/// 领域类型全量定义契约 (DomainTypeDefinition)。
///
/// 支持复合领域模型 (Composite Domain Model)：
/// 单个稳定领域类型可同时拥有 `DurableService` (业务服务持久状态)、
/// `RollbackableModel` (仿真域可回滚投机状态) 与 `CommittedProjection` (GVT 已提交全局只读投影) 等
/// 多种 Store/Schema Binding。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DomainTypeDefinition {
    pub id: DomainTypeId,
    pub version: SchemaVersion,
    pub bindings: Vec<SchemaStoreBinding>,
    pub schema_digest: u64,
    pub code_digest: u64,
}

impl DomainTypeDefinition {
    /// 兼容单状态分类构造函数
    pub fn new(
        id: impl Into<DomainTypeId>,
        version: SchemaVersion,
        state_class: StateClass,
        schema_digest: u64,
        code_digest: u64,
    ) -> Self {
        Self {
            id: id.into(),
            version,
            bindings: vec![SchemaStoreBinding::new(state_class, schema_digest)],
            schema_digest,
            code_digest,
        }
    }

    /// 复合领域模型全量构造函数
    pub fn new_composite(
        id: impl Into<DomainTypeId>,
        version: SchemaVersion,
        bindings: Vec<SchemaStoreBinding>,
        schema_digest: u64,
        code_digest: u64,
    ) -> Self {
        Self {
            id: id.into(),
            version,
            bindings,
            schema_digest,
            code_digest,
        }
    }

    /// 链式添加 Store/Schema 绑定
    pub fn with_binding(mut self, state_class: StateClass, schema_digest: u64) -> Self {
        if !self.has_state_class(state_class) {
            self.bindings
                .push(SchemaStoreBinding::new(state_class, schema_digest));
        }
        self
    }

    /// 检查是否包含指定的 StateClass 绑定
    #[inline]
    pub fn has_state_class(&self, state_class: StateClass) -> bool {
        self.bindings.iter().any(|b| b.state_class == state_class)
    }

    /// 获取特定 StateClass 的 Store/Schema 绑定
    #[inline]
    pub fn binding_for(&self, state_class: StateClass) -> Option<&SchemaStoreBinding> {
        self.bindings.iter().find(|b| b.state_class == state_class)
    }

    /// 获取首个状态分类（兼容单分类场景）
    #[inline]
    pub fn state_class(&self) -> Option<StateClass> {
        self.bindings.first().map(|b| b.state_class)
    }

    /// 转换为持久化引用
    #[inline]
    pub fn to_persisted_ref(&self) -> PersistedDomainTypeRef {
        PersistedDomainTypeRef::new(
            self.id.clone(),
            self.version,
            self.schema_digest,
            self.code_digest,
        )
    }
}
