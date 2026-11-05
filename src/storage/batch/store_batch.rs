use super::ops::*;
use crate::binding::EntityId;
use crate::context::ModelComponent;
use crate::error::StateError;
use serde::de::DeserializeOwned;
use serde::Serialize;
use std::collections::BTreeMap;

/// 批次命令按操作类别的精准计数
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct BatchOpCounts {
    pub creates: usize,
    pub deletes: usize,
    pub in_place_patches: usize,
    pub structural_patches: usize,
    pub conditional_patches: usize,
    pub empty_patches: usize,
}

/// 统一存储批次操作容器 (StoreBatch)。
///
/// 严格约束：同批次对同一实体只允许出现一个操作项，重复则拒绝，杜绝时序歧义。
#[derive(Debug, Default)]
pub struct StoreBatch {
    pub(crate) ops: BTreeMap<EntityId, BatchOp>,
    creates: usize,
    deletes: usize,
    in_place_patches: usize,
    structural_patches: usize,
    conditional_patches: usize,
    empty_patches: usize,
}

impl StoreBatch {
    pub fn new() -> Self {
        Self {
            ops: BTreeMap::new(),
            creates: 0,
            deletes: 0,
            in_place_patches: 0,
            structural_patches: 0,
            conditional_patches: 0,
            empty_patches: 0,
        }
    }

    pub fn create_entity(
        &mut self,
        id: EntityId,
        bundle: EntityBundle,
    ) -> Result<&mut Self, StateError> {
        match self.ops.entry(id) {
            std::collections::btree_map::Entry::Occupied(_) => {
                Err(StateError::BatchConflict(format!(
                    "Entity {id} already has a pending operation in this batch"
                )))
            }
            std::collections::btree_map::Entry::Vacant(vac) => {
                vac.insert(BatchOp::CreateEntity { id, bundle });
                self.creates += 1;
                Ok(self)
            }
        }
    }

    pub fn patch_entity(
        &mut self,
        id: EntityId,
        patch: EntityPatch,
    ) -> Result<&mut Self, StateError> {
        match self.ops.entry(id) {
            std::collections::btree_map::Entry::Occupied(_) => {
                Err(StateError::BatchConflict(format!(
                    "Entity {id} already has a pending operation in this batch"
                )))
            }
            std::collections::btree_map::Entry::Vacant(vac) => {
                match patch.semantics() {
                    PatchSemantics::Empty => self.empty_patches += 1,
                    PatchSemantics::InPlace => self.in_place_patches += 1,
                    PatchSemantics::Structural => self.structural_patches += 1,
                    PatchSemantics::Conditional => self.conditional_patches += 1,
                }
                vac.insert(BatchOp::PatchEntity { id, patch });
                Ok(self)
            }
        }
    }

    pub fn delete_entity(&mut self, id: EntityId) -> Result<&mut Self, StateError> {
        match self.ops.entry(id) {
            std::collections::btree_map::Entry::Occupied(_) => {
                Err(StateError::BatchConflict(format!(
                    "Entity {id} already has a pending operation in this batch"
                )))
            }
            std::collections::btree_map::Entry::Vacant(vac) => {
                vac.insert(BatchOp::DeleteEntity { id });
                self.deletes += 1;
                Ok(self)
            }
        }
    }

    /// 更新已有实体的单个组件值（快捷原位替换更新）
    #[inline]
    pub fn update<T: ModelComponent + Serialize + DeserializeOwned + 'static>(
        &mut self,
        id: EntityId,
        component: T,
    ) -> &mut Self {
        let mut patch = EntityPatch::new();
        if let Err(e) = patch.replace(component) {
            panic!("StoreBatch::update failed to construct patch: {e}");
        }
        if let Err(e) = self.patch_entity(id, patch) {
            panic!("StoreBatch::update failed to insert patch for entity {id}: {e}");
        }
        self
    }

    /// 尝试更新已有实体的单个组件值（快捷原位替换更新）
    #[inline]
    pub fn try_update<T: ModelComponent + Serialize + DeserializeOwned + 'static>(
        &mut self,
        id: EntityId,
        component: T,
    ) -> Result<&mut Self, StateError> {
        let mut patch = EntityPatch::new();
        patch.replace(component)?;
        self.patch_entity(id, patch)
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.ops.is_empty()
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.ops.len()
    }

    #[inline]
    pub fn ops(&self) -> &BTreeMap<EntityId, BatchOp> {
        &self.ops
    }

    #[inline]
    pub(crate) fn counts(&self) -> BatchOpCounts {
        BatchOpCounts {
            creates: self.creates,
            deletes: self.deletes,
            in_place_patches: self.in_place_patches,
            structural_patches: self.structural_patches,
            conditional_patches: self.conditional_patches,
            empty_patches: self.empty_patches,
        }
    }

    #[inline]
    pub fn into_ops(self) -> BTreeMap<EntityId, BatchOp> {
        self.ops
    }

    pub(crate) fn from_ops(ops: BTreeMap<EntityId, BatchOp>) -> Self {
        let mut creates = 0;
        let mut deletes = 0;
        let mut in_place_patches = 0;
        let mut structural_patches = 0;
        let mut conditional_patches = 0;
        let mut empty_patches = 0;
        for op in ops.values() {
            match op {
                BatchOp::CreateEntity { .. } => creates += 1,
                BatchOp::DeleteEntity { .. } => deletes += 1,
                BatchOp::PatchEntity { patch, .. } => match patch.semantics() {
                    PatchSemantics::Empty => empty_patches += 1,
                    PatchSemantics::InPlace => in_place_patches += 1,
                    PatchSemantics::Structural => structural_patches += 1,
                    PatchSemantics::Conditional => conditional_patches += 1,
                },
            }
        }
        Self {
            ops,
            creates,
            deletes,
            in_place_patches,
            structural_patches,
            conditional_patches,
            empty_patches,
        }
    }
}
