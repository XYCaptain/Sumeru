//! 组件定义与空列原型注册表 (ComponentRegistry)
//!
//! 统一管理强类型组件的静态元数据、模式版本、列编解码器与空列工厂原型。
//! 严格保障边界一致性与冲突检测，并为新 Archetype 建列与批次规划提供 O(log K) 高效原型复用。

#![allow(dead_code)]

use crate::storage::column::{AnyColumn, ColumnCodecId};
use crate::storage::ErasedComponentValue;
use crate::error::StateError;
use crate::id::ComponentTypeId;
use std::collections::BTreeMap;
use std::sync::Arc;

/// 集中管理的组件元数据与空列原型工厂
pub(crate) struct ComponentDefinition {
    type_id: ComponentTypeId,
    type_name: &'static str,
    rust_type_id: Option<std::any::TypeId>,
    codec: ColumnCodecId,
    schema_version: u32,
    prototype: Box<dyn AnyColumn>,
}

impl ComponentDefinition {
    /// 从空列原型构造组件定义
    pub fn new(prototype: Box<dyn AnyColumn>) -> Self {
        Self {
            type_id: prototype.type_id(),
            type_name: prototype.type_name(),
            rust_type_id: prototype.rust_type_id(),
            codec: prototype.codec(),
            schema_version: prototype.schema_version(),
            prototype,
        }
    }

    #[inline]
    pub fn type_id(&self) -> ComponentTypeId {
        self.type_id
    }

    #[inline]
    pub fn type_name(&self) -> &'static str {
        self.type_name
    }

    #[inline]
    pub fn rust_type_id(&self) -> Option<std::any::TypeId> {
        self.rust_type_id
    }

    #[inline]
    pub fn codec(&self) -> ColumnCodecId {
        self.codec
    }

    #[inline]
    pub fn schema_version(&self) -> u32 {
        self.schema_version
    }

    #[inline]
    pub fn prototype(&self) -> &dyn AnyColumn {
        self.prototype.as_ref()
    }

    #[inline]
    pub fn has_archive_capability(&self) -> bool {
        self.prototype.has_archive_capability()
    }

    #[inline]
    pub fn is_page_aligned(&self) -> bool {
        self.prototype.is_page_aligned()
    }

    #[inline]
    pub fn promote_to_page_aligned(&mut self) {
        self.prototype.promote_to_page_aligned();
    }

    /// 克隆一份用于新 Archetype 的空列 (零组件数据拷贝)
    #[inline]
    pub fn create_empty_column(&self) -> Box<dyn AnyColumn> {
        self.prototype.empty_clone()
    }

    /// 严格核验外部原型与现有注册定义是否冲突 (类型名、Rust TypeId、编解码器、模式版本)
    pub fn check_compatibility(&self, other: &dyn AnyColumn) -> Result<(), StateError> {
        if self.rust_type_id != other.rust_type_id() || self.type_name != other.type_name() {
            return Err(StateError::AccessDenied(format!(
                "Component prototype {} conflicts with existing registered type",
                self.type_id
            )));
        }
        if self.codec != other.codec()
            && !(self.codec == ColumnCodecId::None && other.has_archive_capability())
        {
            return Err(StateError::AccessDenied(format!(
                "Component prototype {} codec {:?} conflicts with existing registered codec {:?}",
                self.type_id,
                other.codec(),
                self.codec,
            )));
        }
        if self.codec != ColumnCodecId::None && self.schema_version != other.schema_version() {
            return Err(StateError::AccessDenied(format!(
                "Component prototype {} schema version {} conflicts with existing registered version {}",
                self.type_id,
                other.schema_version(),
                self.schema_version,
            )));
        }
        Ok(())
    }

    /// 核验分区合并时外部原型与当前注册定义是否兼容（允许纯内存分区自动升级至已归档定义）
    pub fn check_merge_compatibility(&self, other: &dyn AnyColumn) -> Result<(), StateError> {
        if self.rust_type_id != other.rust_type_id() || self.type_name != other.type_name() {
            return Err(StateError::AccessDenied(format!(
                "Component prototype {} conflicts with existing registered type",
                self.type_id
            )));
        }
        if self.codec != other.codec()
            && !(self.codec == ColumnCodecId::None && other.has_archive_capability())
            && !(other.codec() == ColumnCodecId::None && self.has_archive_capability())
        {
            return Err(StateError::AccessDenied(format!(
                "Component prototype {} codec {:?} conflicts with existing registered codec {:?}",
                self.type_id,
                other.codec(),
                self.codec,
            )));
        }
        if self.codec != ColumnCodecId::None
            && other.codec() != ColumnCodecId::None
            && self.schema_version != other.schema_version()
        {
            return Err(StateError::AccessDenied(format!(
                "Component prototype {} schema version {} conflicts with existing registered version {}",
                self.type_id,
                other.schema_version(),
                self.schema_version,
            )));
        }
        Ok(())
    }

    /// 升级已有纯内存定义的归档能力或 16KB 页对齐能力
    pub fn upgrade_archive_capability(&mut self, new_prototype: &dyn AnyColumn) -> Result<(), StateError> {
        if self.rust_type_id != new_prototype.rust_type_id() || self.type_name != new_prototype.type_name() {
            return Err(StateError::AccessDenied(format!(
                "Cannot upgrade archive capability for {} due to type mismatch",
                self.type_id
            )));
        }
        let can_upgrade_archive = !self.has_archive_capability() && new_prototype.has_archive_capability();
        let can_upgrade_paged = !self.is_page_aligned() && new_prototype.is_page_aligned();
        if can_upgrade_archive || can_upgrade_paged {
            let was_paged = self.is_page_aligned() || new_prototype.is_page_aligned();
            if can_upgrade_archive {
                self.codec = new_prototype.codec();
                self.schema_version = new_prototype.schema_version();
                self.prototype = new_prototype.empty_clone();
            }
            if was_paged {
                self.prototype.promote_to_page_aligned();
            }
            Ok(())
        } else {
            Err(StateError::AccessDenied(format!(
                "Component prototype {} already has archive capability or source lacks it",
                self.type_id
            )))
        }
    }

    /// 普通写值类型核验：核验擦除值的原生类型与定义是否相符 (零堆分配，不调用 empty_column)
    #[inline]
    pub fn validate_value_type(&self, value: &ErasedComponentValue) -> Result<(), StateError> {
        if self.rust_type_id.is_some_and(|expected| expected != value.rust_type_id())
            || value.type_name() != self.type_name
        {
            return Err(StateError::AccessDenied(format!(
                "Native component {} ({:?}) does not match registered column {} ({:?})",
                value.type_name(),
                value.rust_type_id(),
                self.type_name,
                self.rust_type_id,
            )));
        }
        Ok(())
    }
}

impl Clone for ComponentDefinition {
    fn clone(&self) -> Self {
        Self {
            type_id: self.type_id,
            type_name: self.type_name,
            rust_type_id: self.rust_type_id,
            codec: self.codec,
            schema_version: self.schema_version,
            prototype: self.prototype.empty_clone(),
        }
    }
}

impl std::fmt::Debug for ComponentDefinition {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ComponentDefinition")
            .field("type_id", &self.type_id)
            .field("type_name", &self.type_name)
            .field("rust_type_id", &self.rust_type_id)
            .field("codec", &self.codec)
            .field("schema_version", &self.schema_version)
            .finish()
    }
}

/// 集中组件定义注册表
#[derive(Default, Clone, Debug)]
pub(crate) struct ComponentRegistry {
    definitions: Arc<BTreeMap<ComponentTypeId, ComponentDefinition>>,
}

impl ComponentRegistry {
    pub fn new() -> Self {
        Self {
            definitions: Arc::new(BTreeMap::new()),
        }
    }

    #[inline]
    pub fn get(&self, type_id: &ComponentTypeId) -> Option<&ComponentDefinition> {
        self.definitions.get(type_id)
    }

    /// 轻量直查组件的原生 TypeId 与类型名称，零复制、零堆分配。
    #[inline]
    pub fn get_type_info(&self, type_id: &ComponentTypeId) -> Option<(Option<std::any::TypeId>, &'static str)> {
        self.definitions.get(type_id).map(|d| (d.rust_type_id, d.type_name))
    }

    #[inline]
    pub fn contains(&self, type_id: &ComponentTypeId) -> bool {
        self.definitions.contains_key(type_id)
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.definitions.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.definitions.is_empty()
    }

    #[inline]
    pub fn keys(&self) -> impl Iterator<Item = &ComponentTypeId> {
        self.definitions.keys()
    }

    #[inline]
    pub fn iter(&self) -> impl Iterator<Item = (&ComponentTypeId, &ComponentDefinition)> {
        self.definitions.iter()
    }

    /// 快速克隆指定组件类型的空列 (O(log K) 查找，零 Archetype 线性扫描)
    #[inline]
    pub fn spawn_column(&self, type_id: ComponentTypeId) -> Option<Box<dyn AnyColumn>> {
        self.definitions.get(&type_id).map(|d| d.create_empty_column())
    }

    /// 将当前注册表中所有原型提升为 16KB 页对齐模式
    pub fn promote_all_to_page_aligned(&mut self) {
        if self.definitions.values().all(|d| d.is_page_aligned()) {
            return;
        }
        let defs = Arc::make_mut(&mut self.definitions);
        for def in defs.values_mut() {
            def.promote_to_page_aligned();
        }
    }

    /// 暂存并校验组件原型批次 (All-or-Nothing 只读暂存)
    ///
    /// 完整继承现有安全校验防线，绝不直接变更内部注册状态：
    /// 1. 原型列必须为空；
    /// 2. 必须具备原生 Rust 类型 (拒绝未定型 RawColumn)；
    /// 3. 组件 ID 与名称必须完全一致；
    /// 4. 批次内部不得重复定义；
    /// 5. 与正式注册表已有定义比对冲突 (类型名、TypeId、编解码器、模式版本)。
    pub fn stage_prototypes(
        &self,
        prototypes: Vec<Box<dyn AnyColumn>>,
    ) -> Result<BTreeMap<ComponentTypeId, ComponentDefinition>, StateError> {
        let mut staging: BTreeMap<ComponentTypeId, ComponentDefinition> = BTreeMap::new();
        for column in prototypes {
            let type_id = column.as_ref().type_id();
            // 1. 空原型防卫
            if !column.is_empty() {
                return Err(StateError::AccessDenied(format!(
                    "Component prototype {type_id} is nonempty"
                )));
            }
            // 2. 原生类型防卫 (拒绝 RawColumn 等非原生定义)
            if column.rust_type_id().is_none() {
                return Err(StateError::AccessDenied(format!(
                    "Component prototype {type_id} missing native Rust type"
                )));
            }
            // 3. 规范命名与 ID 对应
            if type_id != ComponentTypeId::from_name(column.type_name()) {
                return Err(StateError::AccessDenied(format!(
                    "Component prototype {type_id} has noncanonical name {}",
                    column.type_name()
                )));
            }
            // 4. 单批次内防重与一致性比对
            if let Some(staged) = staging.get_mut(&type_id) {
                staged.check_compatibility(column.as_ref())?;
                let was_paged = staged.is_page_aligned() || column.is_page_aligned();
                if !staged.has_archive_capability() && column.has_archive_capability() {
                    *staged = ComponentDefinition::new(column);
                }
                if was_paged {
                    staged.promote_to_page_aligned();
                }
                continue;
            }
            // 5. 与正式注册表中已有定义比对冲突
            if let Some(existing) = self.definitions.get(&type_id) {
                existing.check_compatibility(column.as_ref())?;
                let upgrade_archive = !existing.has_archive_capability() && column.has_archive_capability();
                let upgrade_paged = !existing.is_page_aligned() && column.is_page_aligned();
                if upgrade_archive || upgrade_paged {
                    let mut def = if upgrade_archive {
                        ComponentDefinition::new(column)
                    } else {
                        existing.clone()
                    };
                    if existing.is_page_aligned() || upgrade_paged {
                        def.promote_to_page_aligned();
                    }
                    staging.insert(type_id, def);
                }
                continue;
            }

            staging.insert(type_id, ComponentDefinition::new(column));
        }

        Ok(staging)
    }

    /// 原子批注册组件原型 (All-or-Nothing)
    pub fn register_prototypes(
        &mut self,
        prototypes: Vec<Box<dyn AnyColumn>>,
    ) -> Result<(), StateError> {
        let staging = self.stage_prototypes(prototypes)?;
        self.extend(staging);
        Ok(())
    }

    /// 登记单组件原型
    pub fn register_single_prototype(
        &mut self,
        prototype: Box<dyn AnyColumn>,
    ) -> Result<(), StateError> {
        self.register_prototypes(vec![prototype])
    }

    /// 原子合并外部已验证定义 (如批次提交 PreparedBatch.new_definitions 或新注册能力升级)
    #[inline]
    pub fn extend(&mut self, new_defs: BTreeMap<ComponentTypeId, ComponentDefinition>) {
        if new_defs.is_empty() {
            return;
        }
        let needs_mutation = new_defs.iter().any(|(type_id, def)| {
            match self.definitions.get(type_id) {
                Some(existing) => {
                    (!existing.has_archive_capability() && def.has_archive_capability())
                        || (!existing.is_page_aligned() && def.is_page_aligned())
                }
                None => true,
            }
        });
        if !needs_mutation {
            return;
        }
        let defs = Arc::make_mut(&mut self.definitions);
        for (type_id, def) in new_defs {
            if let Some(existing) = defs.get_mut(&type_id) {
                if (!existing.has_archive_capability() && def.has_archive_capability())
                    || (!existing.is_page_aligned() && def.is_page_aligned())
                {
                    existing
                        .upgrade_archive_capability(def.prototype())
                        .expect("pre-validated staging definitions must upgrade successfully");
                }
            } else {
                defs.insert(type_id, def);
            }
        }
    }

    /// 移除指定组件定义 (用于注册事务回滚)
    #[inline]
    pub fn remove(&mut self, type_id: &ComponentTypeId) -> Option<ComponentDefinition> {
        if !self.definitions.contains_key(type_id) {
            return None;
        }
        Arc::make_mut(&mut self.definitions).remove(type_id)
    }
}
