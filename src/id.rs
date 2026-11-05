//! 稳定 ID、版本与代际键定义
//!
//! `sumer_state` 零内部依赖核心标识定义：包含 128 位稳定实体标识 [`EntityId`]、
//! 64 位局部物理槽位标识 [`LocalSlotId`]、16 位连续列标识 [`ColumnId`] 以及组件/领域类型标识。

use serde::{Deserialize, Serialize};
use std::fmt::{Display, Formatter};
use uuid::Uuid;

/// Stable, opaque 128-bit entity identity.
///
/// Service and simulation facets share this identity, but own independent
/// storage locations. No bits encode a partition, slot, or model generation.
/// Constructing an ID does not register it or prove that it can be routed.
/// The nil default is only a placeholder; registration must reject it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[repr(transparent)]
#[serde(transparent)]
pub struct EntityId(Uuid);

impl std::hash::Hash for EntityId {
    #[inline(always)]
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        state.write_u128(u128::from_ne_bytes(*self.0.as_bytes()));
    }
}

impl PartialOrd for EntityId {
    #[inline(always)]
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for EntityId {
    #[inline(always)]
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0.as_u128().cmp(&other.0.as_u128())
    }
}

/// 内部私有使用的 64 位局部物理槽位标识 (`LocalSlotId`)。
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize, Default,
)]
#[repr(C)]
pub struct LocalSlotId {
    pub slot: u32,
    pub generation: u32,
}

impl LocalSlotId {
    #[inline(always)]
    pub const fn new(slot: u32, generation: u32) -> Self {
        Self { slot, generation }
    }
}

impl EntityId {
    /// Wrap a stable UUID without deriving any storage or routing address.
    #[inline]
    pub const fn from_uuid(uuid: Uuid) -> Self {
        Self(uuid)
    }

    #[inline]
    pub const fn to_uuid(self) -> Uuid {
        self.0
    }

    #[inline]
    pub const fn from_u128(value: u128) -> Self {
        Self(Uuid::from_u128(value))
    }

    #[inline]
    pub const fn to_u128(self) -> u128 {
        self.0.as_u128()
    }

    #[inline]
    pub const fn is_nil(self) -> bool {
        self.0.is_nil()
    }
}

impl Display for EntityId {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        Display::fmt(&self.0, f)
    }
}

/// Explicit address of one simulation model incarnation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize)]
pub struct ModelAddress {
    entity_id: EntityId,
    partition_id: u32,
    model_generation: u64,
}

impl ModelAddress {
    pub fn new(
        entity_id: EntityId,
        partition_id: u32,
        model_generation: u64,
    ) -> Result<Self, String> {
        if entity_id.is_nil() {
            return Err("ModelAddress entity_id must not be nil".into());
        }
        if model_generation == 0 {
            return Err("ModelAddress model_generation must be nonzero".into());
        }
        Ok(Self {
            entity_id,
            partition_id,
            model_generation,
        })
    }

    #[inline]
    pub const fn entity_id(self) -> EntityId {
        self.entity_id
    }

    #[inline]
    pub const fn partition_id(self) -> u32 {
        self.partition_id
    }

    #[inline]
    pub const fn model_generation(self) -> u64 {
        self.model_generation
    }
}

impl<'de> Deserialize<'de> for ModelAddress {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        #[derive(Deserialize)]
        #[serde(deny_unknown_fields)]
        struct Wire {
            entity_id: EntityId,
            partition_id: u32,
            model_generation: u64,
        }

        let wire = Wire::deserialize(deserializer)?;
        Self::new(wire.entity_id, wire.partition_id, wire.model_generation)
            .map_err(serde::de::Error::custom)
    }
}

impl Display for ModelAddress {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}@P{}:ModelGen{}",
            self.entity_id, self.partition_id, self.model_generation
        )
    }
}

/// 连续列式存储中强类型的 16 位列标识 (`ColumnId`)。
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize, Default,
)]
pub struct ColumnId(pub u16);

impl From<u16> for ColumnId {
    #[inline(always)]
    fn from(val: u16) -> Self {
        Self(val)
    }
}

impl Display for ColumnId {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "Col({})", self.0)
    }
}

/// 组件类型唯一哈希标识 (`ComponentTypeId`)。
///
/// 采用 64 位 FNV-1a 哈希或明确分配的静态 ID，具备零成本比对与稳定序列化特性。
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize, Default,
)]
pub struct ComponentTypeId(pub u64);

impl ComponentTypeId {
    #[inline]
    pub const fn new(raw: u64) -> Self {
        Self(raw)
    }

    /// 使用 const FNV-1a 64-bit 算法从类型名称生成稳定哈希
    #[inline(always)]
    pub const fn from_name(name: &str) -> Self {
        let bytes = name.as_bytes();
        let mut hash: u64 = 0xcbf29ce484222325;
        let prime: u64 = 0x100000001b3;
        let mut i = 0;
        let len = bytes.len();
        while i + 8 <= len {
            hash = (hash ^ (bytes[i] as u64)).wrapping_mul(prime);
            hash = (hash ^ (bytes[i + 1] as u64)).wrapping_mul(prime);
            hash = (hash ^ (bytes[i + 2] as u64)).wrapping_mul(prime);
            hash = (hash ^ (bytes[i + 3] as u64)).wrapping_mul(prime);
            hash = (hash ^ (bytes[i + 4] as u64)).wrapping_mul(prime);
            hash = (hash ^ (bytes[i + 5] as u64)).wrapping_mul(prime);
            hash = (hash ^ (bytes[i + 6] as u64)).wrapping_mul(prime);
            hash = (hash ^ (bytes[i + 7] as u64)).wrapping_mul(prime);
            i += 8;
        }
        while i < len {
            hash = (hash ^ (bytes[i] as u64)).wrapping_mul(prime);
            i += 1;
        }
        Self(hash)
    }

    /// 从类型 T 生成唯一稳定类型标识
    #[inline(always)]
    pub fn of<T: 'static>() -> Self {
        Self::from_name(std::any::type_name::<T>())
    }

    #[inline]
    pub const fn as_u64(&self) -> u64 {
        self.0
    }
}

impl Display for ComponentTypeId {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "ComponentTypeId({:#018x})", self.0)
    }
}

/// 语义架构版本号 (`SchemaVersion`)
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize, Default,
)]
pub struct SchemaVersion(pub u32);

impl SchemaVersion {
    pub const V1: SchemaVersion = SchemaVersion(1);

    #[inline]
    pub const fn new(version: u32) -> Self {
        Self(version)
    }

    #[inline]
    pub const fn as_u32(&self) -> u32 {
        self.0
    }
}

impl Display for SchemaVersion {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "v{}", self.0)
    }
}

/// 逻辑稳定领域类型标识 (`DomainTypeId`)，如 `"acc.vehicle"`
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct DomainTypeId(pub String);

impl DomainTypeId {
    #[inline]
    pub fn new(id: impl Into<String>) -> Self {
        Self(id.into())
    }

    #[inline]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl Display for DomainTypeId {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl From<&str> for DomainTypeId {
    fn from(s: &str) -> Self {
        Self(s.to_string())
    }
}

impl From<String> for DomainTypeId {
    fn from(s: String) -> Self {
        Self(s)
    }
}

/// 进程内单态化快速索引代际键 (`DomainTypeKey`)。
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize, Default,
)]
pub struct DomainTypeKey {
    pub index: u32,
    pub generation: u32,
}

impl DomainTypeKey {
    #[inline]
    pub const fn new(index: u32, generation: u32) -> Self {
        Self { index, generation }
    }
}

/// 强类型端口标识 (`PortId`)
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize, Default,
)]
pub struct PortId(pub u32);

impl PortId {
    #[inline]
    pub const fn new(id: u32) -> Self {
        Self(id)
    }

    #[inline]
    pub const fn as_u32(&self) -> u32 {
        self.0
    }
}

impl Display for PortId {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "Port({})", self.0)
    }
}

/// 持久化/跨进程边界使用的权威类型引用 (`PersistedDomainTypeRef`)。
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct PersistedDomainTypeRef {
    pub id: DomainTypeId,
    pub version: SchemaVersion,
    pub schema_digest: u64,
    pub code_digest: u64,
}

impl PersistedDomainTypeRef {
    pub fn new(
        id: impl Into<DomainTypeId>,
        version: SchemaVersion,
        schema_digest: u64,
        code_digest: u64,
    ) -> Self {
        Self {
            id: id.into(),
            version,
            schema_digest,
            code_digest,
        }
    }

    /// 校验是否与本地运行时的注册信息严格匹配
    #[inline]
    pub fn matches(&self, expected_schema: u64, expected_code: u64) -> bool {
        self.schema_digest == expected_schema && self.code_digest == expected_code
    }
}

/// Archetype 唯一标识
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct ArchetypeId(pub u32);

/// 历史链头标识
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct HistoryHead(pub u64);
