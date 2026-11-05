//! 统一存储与模式错误定义 (StateError)

use crate::{binding::EntityId, id::ComponentTypeId};
use thiserror::Error;

/// Why an otherwise valid access cannot proceed now. Waiting policy belongs to the caller.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AccessBusyReason {
    Metadata,
    ActiveLeases,
    StructuralPreparation,
    /// An explicit whole-directory handoff owns admission until cancellation/open.
    DirectorySwitch,
}

/// A Pending's original target is no longer usable as declared.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TargetChangeReason {
    Boundary,
    EntityMissing,
    ComponentMissing,
    Topology,
}

/// Recovery category, not an automatic retry or scheduling policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RecoveryAction {
    Retry,
    Recompute,
    Reselect,
    FixInput,
    RelieveCapacity,
    Restore,
    /// General batch conflicts are ambiguous; do not blindly retry them.
    InspectConflict,
}

/// 永久单调标识序列耗尽种类（不可逆单调序列溢出，释放数据内存无法自愈）
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExhaustionKind {
    LeaseId,
    HistoryHead,
    TrackingRevision,
    TrackingDomain,
    CoreEpoch,
    ArchetypeGeneration,
    IdentityStamp,
    PhysicalVersion,
}

/// StateFabric 统一存储、类型注册与模式契约访问错误
#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum StateError {
    #[error("Access temporarily busy: {reason:?}")]
    AccessBusy { reason: AccessBusyReason },

    #[error("Input baseline changed for entity {entity}, component {component:?}")]
    InputBaselineChanged {
        entity: EntityId,
        component: ComponentTypeId,
    },

    #[error("Input target changed: {reason:?}, entity {entity:?}")]
    InputTargetChanged {
        reason: TargetChangeReason,
        entity: Option<EntityId>,
    },

    #[error("Range targets must be contiguous and ordered in one archetype")]
    RangeNotContiguous,

    #[error("Component '{0}' not found in archetype")]
    ComponentNotFound(String),

    #[error("Domain type '{0}' not registered")]
    TypeNotRegistered(String),

    #[error("Domain type '{0}' already registered with different schema/digest")]
    TypeAlreadyRegistered(String),

    #[error("Digest mismatch for domain type '{id}': expected {expected:#x}, got {actual:#x}")]
    DigestMismatch {
        id: String,
        expected: u64,
        actual: u64,
    },

    #[error("Capability access denied: {0}")]
    AccessDenied(String),

    #[error("Capacity exhausted: {0}")]
    CapacityExhausted(String),

    /// 单调标识序列永久耗尽（释放内存无法产生新编号）
    #[error("Identifier sequence permanently exhausted: {kind:?}")]
    IdentifierExhausted { kind: ExhaustionKind },

    #[error("Entity generation overflow (exceeded u32::MAX)")]
    GenerationOverflow,

    #[error("Serialization / Deserialization error: {0}")]
    CodecError(String),

    #[error("Entity {0} does not exist or is inactive")]
    EntityNotFound(EntityId),

    #[error("Entity {0} already exists")]
    EntityAlreadyExists(EntityId),

    #[error("Invalid entity ID: {0}")]
    InvalidEntityId(String),

    #[error("Component '{type_name}' already exists on entity {entity}")]
    ComponentAlreadyExists { entity: EntityId, type_name: String },

    #[error("Component '{type_name}' not found on entity {entity}")]
    ComponentNotFoundInEntity { entity: EntityId, type_name: String },

    #[error("Component '{type_name}' duplicate operation in patch for entity {entity}")]
    ComponentDuplicateInPatch { entity: EntityId, type_name: String },

    #[error("Type mismatch: expected '{expected}', found '{actual}'")]
    TypeMismatch { expected: String, actual: String },

    #[error("Batch conflict: {0}")]
    BatchConflict(String),

    #[error("Stale batch: expected epoch {expected_epoch}, store is at epoch {actual_epoch}")]
    StaleBatch {
        expected_epoch: u64,
        actual_epoch: u64,
    },

    #[error(
        "Store mismatch: batch was prepared for store {expected_store}, target is {actual_store}"
    )]
    StoreMismatch {
        expected_store: u64,
        actual_store: u64,
    },

    #[error("Borrow conflict on {component}: {existing_access} held, {requested_access} requested")]
    BorrowConflict {
        component: String,
        existing_access: String,
        requested_access: String,
    },

    /// 请求内部包含互相冲突的借用区间或参数（确定性请求错误，重试无用）
    #[error("Self-conflict in borrow request for component '{component}': {detail}")]
    SelfBorrowConflict {
        component: String,
        detail: String,
    },

    #[error("Archetype set generation overflow (exceeded u64::MAX)")]
    ArchetypeGenerationOverflow,

    #[error("Storage access poisoned by a prior application-stage failure")]
    SessionPoisoned,

    #[error("Entity {0} has already been committed in this staging session")]
    AlreadyCommitted(EntityId),

    #[error("Invalid commit mode transition: {0}")]
    InvalidCommitMode(String),

    #[error("Unsupported operation: {0}")]
    UnsupportedOperation(String),

    #[error("History is empty")]
    HistoryEmpty,
}

impl StateError {
    /// Classify without parsing messages. A retry still needs a healthy Store;
    /// this does not promise input recovery after application-stage failure.
    pub fn recovery_action(&self) -> RecoveryAction {
        match self {
            Self::AccessBusy { .. } | Self::BorrowConflict { .. } => RecoveryAction::Retry,
            Self::SelfBorrowConflict { .. } => RecoveryAction::FixInput,
            Self::InputBaselineChanged { .. } | Self::StaleBatch { .. } => {
                RecoveryAction::Recompute
            }
            Self::InputTargetChanged { .. } | Self::RangeNotContiguous => RecoveryAction::Reselect,
            Self::CapacityExhausted(_) => RecoveryAction::RelieveCapacity,
            Self::IdentifierExhausted { .. }
            | Self::SessionPoisoned
            | Self::GenerationOverflow
            | Self::ArchetypeGenerationOverflow => RecoveryAction::Restore,
            Self::BatchConflict(_) => RecoveryAction::InspectConflict,
            _ => RecoveryAction::FixInput,
        }
    }
}
