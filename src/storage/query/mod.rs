//! Layer 3: 业务查询与迭代视图 (Query & View Facade)
//!
//! 负责原型匹配算法、行级谓词过滤、强类型/弱类型迭代视图与实体门面。

pub(crate) mod entity_mut;
pub(crate) mod entity_ref;
pub(crate) mod handle;
pub(crate) mod spec;

pub use spec::iter;

pub use entity_mut::EntityMut;
pub use entity_ref::{ComponentRef, EntityData, EntityRef};
pub use handle::{MatchedTableBinding, Query, QueryBuilder, QueryEntityRequest};
pub use spec::{
    merged_access_descriptors, All, BusinessFilter, ChangedSince, ClosureOpMarker,
    ColumnChunkProbe, ComponentField, FilterEq, KernelOpMarker, QueryChunks, QueryChunksIter,
    QueryData, QueryFilter, QueryForEachOp, QueryIter, QueryMutBuilder, QueryMutData,
    QueryMutParamShape, QueryMutShape, QueryParForEachOp, QueryParMutRunner, QueryRowItem,
    SelectBuilder, SelectParRunner, With, WithFlag, WithPairTarget, Without, WithoutFlag,
};
#[cfg(feature = "gpu")]
pub use spec::QueryGpuRunner;
pub use crate::storage::archetype::RowBitMask;
