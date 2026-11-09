//! 组件注册与模式管理 (Component Schema Registration)

use super::Store;
use crate::error::StateError;
use crate::id::ComponentTypeId;
use crate::storage::column::{AnyColumn, ColumnCodecId};

impl Store {
    pub fn register_components(
        &mut self,
        registrations: impl IntoIterator<Item = crate::storage::column::ComponentRegistration>,
    ) -> Result<(), StateError> {
        self.register_component_prototypes(registrations.into_iter().map(|r| r.column).collect())
    }

    pub(crate) fn register_component_prototypes(
        &mut self,
        mut prototypes: Vec<Box<dyn AnyColumn>>,
    ) -> Result<(), StateError> {
        self.check_poisoned()?;
        if self.layout_optimization == super::LayoutOptimization::PageAlignedGpu {
            for proto in &mut prototypes {
                proto.promote_to_page_aligned();
            }
        }
        // 1. 全局版本容量与代际预检 (提前检查，任何溢出零副作用、零注册表污染)
        self.check_archetype_set_generation_capacity()?;
        self.next_epoch()?;

        // 2. 原型结构与已有表布局完整性校验
        for index in 0..self.archetypes.len() {
            self.validate_archetype_layout(index)?;
        }

        // 3. 针对 ComponentRegistry 执行只读暂存与冲突比对 (All-or-Nothing)
        let staged = self.component_registry.stage_prototypes(prototypes)?;

        // 4. 物理列升级兼容性只读预检：确保若需要升级，所有现有物理列均与升级原型类型兼容
        for (type_id, def) in &staged {
            if def.has_archive_capability() || def.is_page_aligned() {
                for archetype in &self.archetypes {
                    if let Some(col) = archetype.columns().get(type_id) {
                        if (!col.has_archive_capability() || !col.is_page_aligned())
                            && (col.rust_type_id() != def.prototype().rust_type_id()
                                || col.type_name() != def.prototype().type_name())
                        {
                            return Err(StateError::AccessDenied(format!(
                                "Physical column {} in Archetype {} is incompatible with upgrading prototype",
                                type_id,
                                archetype.id().0
                            )));
                        }
                    }
                }
            }
        }

        // 5. 全部预检 100% 通过后，升级已有 Archetype 物理列能力并原子合并入正式注册表
        for (type_id, def) in &staged {
            if def.has_archive_capability() || def.is_page_aligned() {
                for archetype in &mut self.archetypes {
                    let mut changed = false;
                    if let Some(col) = archetype.get_erased_column_mut(*type_id) {
                        if (!col.has_archive_capability() && def.has_archive_capability())
                            || (!col.is_page_aligned() && def.is_page_aligned())
                        {
                            let upgraded = col.upgrade_archive_capability_from(def.prototype());
                            assert!(
                                upgraded,
                                "pre-validated physical column for {type_id} must successfully upgrade capability"
                            );
                            changed = true;
                        }
                    }
                    if changed {
                        archetype.rebuild_flat_columns();
                    }
                }
            }
        }
        self.component_registry.extend(staged);
        self.bump_epoch();
        self.bump_archetype_set_generation()?;
        Ok(())
    }

    /// 返回当前正式注册的独立组件类型定义总数
    #[inline]
    pub fn registered_components_count(&self) -> usize {
        self.component_registry.len()
    }

    /// 检查指定组件类型是否已在正式注册表中登记
    #[inline]
    pub fn is_component_registered(&self, type_id: ComponentTypeId) -> bool {
        self.component_registry.contains(&type_id)
    }

    /// 检查指定组件类型是否具备快照归档能力
    #[inline]
    pub fn has_archive_capability(&self, type_id: ComponentTypeId) -> bool {
        self.component_registry
            .get(&type_id)
            .map_or(false, |d| d.has_archive_capability())
    }

    /// 获取指定组件类型的快照归档编解码标识
    #[inline]
    pub fn component_codec(&self, type_id: ComponentTypeId) -> Option<ColumnCodecId> {
        self.component_registry.get(&type_id).map(|d| d.codec())
    }

    /// Build a detached, empty store retaining registered column schemas.
    pub fn register_component<T: crate::storage::column::ArchiveComponent>(&mut self) -> Result<(), StateError> {
        self.register_components([crate::storage::column::ComponentRegistration::of::<T>()])
    }

    /// 注册组件原型（类型擦除）
    #[allow(dead_code)]
    pub(crate) fn register_component_erased(
        &mut self,
        _type_name: &'static str,
        prototype: Box<dyn AnyColumn>,
    ) -> Result<(), StateError> {
        self.register_component_prototypes(vec![prototype])
    }
}
