//! Core 物理存储引擎内部单元测试套件

pub(crate) use super::store::*;
pub(crate) use super::*;
pub(crate) use crate::error::StateError;
pub(crate) use crate::id::EntityId;
pub(crate) use crate::storage::batch::{EntityBundle, EntityPatch, StoreBatch};
pub(crate) use serde::{Deserialize, Serialize};

mod capacity_planning_tests;
mod component_registry_tests;
mod erased_component_value_test;
mod poison_guard_tests;
mod registration_facade_internal_tests;
mod revision_tests;
mod storage_tests;
