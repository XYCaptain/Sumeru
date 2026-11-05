//! 统一实体只读引用视图 (EntityRef)

use crate::storage::allocator::EntityLocation;
use crate::storage::archetype::Archetype;
use crate::storage::column::TypedColumn;
use crate::storage::store::Store;
use crate::binding::EntityId;
use crate::context::ModelComponent;
use crate::error::StateError;
use crate::id::ComponentTypeId;
use serde::Serialize;
use serde::de::DeserializeOwned;

/// 实体只读借用视图 (EntityRef)。
///
/// 生命周期绑定到底层 Store，提供零堆分配、零拷贝的属性与组件访问能力。
///
/// A borrowed component cannot outlive an intervening structural write:
/// ```compile_fail
/// use sumeru::{EntityBundle, EntityId, Store};
/// let mut store = Store::new();
/// let id = EntityId::from_u128(1);
/// store.create_entity(id, EntityBundle::new().with(1u64).unwrap()).unwrap();
/// let view = store.entity(id).unwrap();
/// let value = view.get::<u64>().unwrap();
/// store.delete_entity(id).unwrap();
/// println!("{value}");
/// ```
pub struct EntityRef<'a> {
    store: &'a Store,
    entity: EntityId,
    location: EntityLocation,
}

impl<'a> EntityRef<'a> {
    pub(crate) fn new(
        store: &'a Store,
        entity: EntityId,
        location: EntityLocation,
    ) -> Self {
        Self {
            store,
            entity,
            location,
        }
    }

    #[inline]
    pub fn id(&self) -> EntityId {
        self.entity
    }

    #[inline]
    fn archetype(&self) -> &'a Archetype {
        &self.store.archetypes[self.location.archetype_id as usize]
    }

    /// 检查实体是否拥有特定组件类型
    #[inline]
    pub fn has<T: ModelComponent + 'static>(&self) -> bool {
        let type_id = ComponentTypeId::of::<T>();
        self.has_type(type_id)
    }

    /// 检查实体是否拥有特定组件类型标识
    #[inline]
    pub fn has_type(&self, type_id: ComponentTypeId) -> bool {
        self.archetype().component_types_slice().contains(&type_id)
    }

    /// 检查实体在当前原型行上的非迁移状态位（`0..=255`）是否已置位
    #[inline]
    pub fn has_flag<const FLAG: u8>(&self) -> bool {
        self.archetype()
            .get_row_flag(self.location.row_index as usize, FLAG)
    }

    /// 获取实体所有组件类型列表（零分配借用内部连续切片）
    #[inline]
    pub fn component_types(&self) -> &[ComponentTypeId] {
        self.archetype().component_types_slice()
    }

    /// 获取强类型组件不可变引用（缺失或类型不匹配时返回 `None`，零堆分配、零拷贝）
    #[inline]
    pub fn get<T: ModelComponent + Serialize + DeserializeOwned + 'static>(&self) -> Option<&'a T> {
        self.archetype()
            .get_column::<T>()?
            .get(self.location.row_index as usize)
    }

    /// 获取强类型组件可选不可变引用（`get::<T>()` 的兼容别名）
    #[inline]
    pub fn get_opt<T: ModelComponent + Serialize + DeserializeOwned + 'static>(&self) -> Option<&'a T> {
        self.get::<T>()
    }

    /// 获取强类型组件不可变引用，缺失或类型不匹配时返回详细的 `StateError`
    pub fn try_get<T: ModelComponent + Serialize + DeserializeOwned + 'static>(
        &self,
    ) -> Result<&'a T, StateError> {
        let arch = self.archetype();
        if let Some(typed) = arch.get_column::<T>() {
            return typed.get(self.location.row_index as usize).ok_or_else(|| {
                StateError::AccessDenied(format!(
                    "Entity {} row {} out of bounds in archetype {}",
                    self.entity, self.location.row_index, self.location.archetype_id
                ))
            });
        }
        let type_id = ComponentTypeId::of::<T>();
        let col = arch.columns().get(&type_id).ok_or_else(|| {
            StateError::ComponentNotFoundInEntity {
                entity: self.entity,
                type_name: std::any::type_name::<T>().to_string(),
            }
        })?;

        let typed = col
            .as_any()
            .downcast_ref::<TypedColumn<T>>()
            .ok_or_else(|| StateError::TypeMismatch {
                expected: std::any::type_name::<T>().to_string(),
                actual: col.type_name().to_string(),
            })?;

        typed.get(self.location.row_index as usize).ok_or_else(|| {
            StateError::AccessDenied(format!(
                "Entity {} row {} out of bounds in archetype {}",
                self.entity, self.location.row_index, self.location.archetype_id
            ))
        })
    }

    /// 查询当前实体上关系 `R` 指向的目标实体 ID（若目标实体已销毁则返回 `None`）
    #[inline]
    pub fn pair_target<R: ModelComponent + Serialize + DeserializeOwned + 'static>(
        &self,
    ) -> Option<EntityId> {
        let target = self.get::<crate::al::Pair<R>>()?.target;
        if self.store.contains_entity(target) {
            Some(target)
        } else {
            None
        }
    }

    /// Borrow a dynamically selected component without cloning or encoding it.
    pub fn get_value(&self, type_id: ComponentTypeId) -> Option<ComponentRef<'a>> {
        let col = self.archetype().columns().get(&type_id)?;
        Some(ComponentRef {
            type_id,
            type_name: col.type_name(),
            value: col.value_ref(self.location.row_index as usize)?,
        })
    }

    /// 克隆组件值（明确需要独立副本时使用）
    pub fn clone_value<T: ModelComponent + Clone + Serialize + DeserializeOwned + 'static>(
        &self,
    ) -> Result<T, StateError> {
        self.try_get::<T>().cloned()
    }

    /// 克隆指定类型 ID 的类型擦除组件值
    pub fn clone_erased(
        &self,
        type_id: ComponentTypeId,
    ) -> Result<crate::storage::ErasedComponentValue, StateError> {
        let col = self.archetype().columns().get(&type_id).ok_or_else(|| {
            StateError::ComponentNotFoundInEntity {
                entity: self.entity,
                type_name: type_id.to_string(),
            }
        })?;
        col.clone_erased(self.location.row_index as usize)
    }

    /// 克隆该实体挂载的全部组件为独立的 `EntityBundle`
    pub fn clone_bundle(&self) -> Result<crate::storage::batch::EntityBundle, StateError> {
        let mut bundle = crate::storage::batch::EntityBundle::new();
        for &cid in self.component_types() {
            bundle.insert_erased(self.clone_erased(cid)?)?;
        }
        Ok(bundle)
    }

    /// 联合借用提取多元组组件（零堆分配、零克隆）
    #[inline]
    pub fn extract<E: EntityData<'a>>(&self) -> Result<E::Item, StateError> {
        E::fetch(self.store, self.entity, self.location)
    }

    /// 联合借用提取多元组组件可选引用（任一组件缺失时返回 `None`，零错误对象分配）
    #[inline]
    pub fn get_tuple<E: EntityData<'a>>(&self) -> Option<E::Item> {
        E::fetch(self.store, self.entity, self.location).ok()
    }
}

/// Read-only native component borrow. Concrete storage and row positions stay private.
pub struct ComponentRef<'a> {
    type_id: ComponentTypeId,
    type_name: &'static str,
    value: &'a (dyn std::any::Any + Send + Sync),
}

impl<'a> ComponentRef<'a> {
    pub fn component_type_id(&self) -> ComponentTypeId {
        self.type_id
    }
    pub fn type_name(&self) -> &'static str {
        self.type_name
    }
    pub fn downcast_ref<T: 'static>(&self) -> Option<&'a T> {
        self.value.downcast_ref()
    }
}

/// A supported read-only entity component extraction shape.
///
/// Supported shapes are `&T` and tuples of two to four shared component references:
/// `&T` yields `&T`, and `(&A, &B)` yields `(&A, &B)`.
///
/// This trait is sealed: internal archetype columns and row access remain private.
///
/// An external type cannot implement the sealed EntityData trait:
/// ```compile_fail
/// struct ExternalData;
/// impl<'a> sumeru::storage::EntityData<'a> for ExternalData {
///     type Item = ();
/// }
/// ```
#[allow(private_bounds)]
pub trait EntityData<'a>: sealed::EntityFetch<'a> {
    type Item: 'a;
}

mod sealed {
    use super::*;

    pub(crate) trait EntityFetch<'a> {
        fn fetch(
            store: &'a Store,
            entity: EntityId,
            location: EntityLocation,
        ) -> Result<<Self as EntityData<'a>>::Item, StateError>
        where
            Self: EntityData<'a>;
    }
}

#[inline]
fn fetch_component<'a, T: ModelComponent + Serialize + DeserializeOwned + 'static>(
    store: &'a Store,
    entity: EntityId,
    location: EntityLocation,
) -> Result<&'a T, StateError> {
    let type_id = ComponentTypeId::of::<T>();
    let archetype = &store.archetypes[location.archetype_id as usize];
    let col = archetype.columns().get(&type_id).ok_or_else(|| {
        StateError::ComponentNotFoundInEntity {
            entity,
            type_name: std::any::type_name::<T>().to_string(),
        }
    })?;

    let typed = col
        .as_any()
        .downcast_ref::<TypedColumn<T>>()
        .ok_or_else(|| StateError::TypeMismatch {
            expected: std::any::type_name::<T>().to_string(),
            actual: col.type_name().to_string(),
        })?;

    typed.get(location.row_index as usize).ok_or_else(|| {
        StateError::AccessDenied(format!(
            "Entity {} row {} out of bounds in archetype {}",
            entity, location.row_index, location.archetype_id
        ))
    })
}

impl<'a, T: ModelComponent + Serialize + DeserializeOwned + 'static> EntityData<'a> for &'a T {
    type Item = &'a T;
}

impl<'a, T: ModelComponent + Serialize + DeserializeOwned + 'static> sealed::EntityFetch<'a>
    for &'a T
{
    #[inline]
    fn fetch(
        store: &'a Store,
        entity: EntityId,
        location: EntityLocation,
    ) -> Result<<Self as EntityData<'a>>::Item, StateError> {
        fetch_component::<T>(store, entity, location)
    }
}

macro_rules! tuple_entity_data {
    ($($name:ident),+) => {
        impl<'a, $($name: ModelComponent + Serialize + DeserializeOwned + 'static),+> EntityData<'a>
            for ($(&'a $name,)+)
        {
            type Item = ($(&'a $name,)+);
        }

        impl<'a, $($name: ModelComponent + Serialize + DeserializeOwned + 'static),+> sealed::EntityFetch<'a>
            for ($(&'a $name,)+)
        {
            #[inline]
            fn fetch(
                store: &'a Store,
                entity: EntityId,
                location: EntityLocation,
            ) -> Result<<Self as EntityData<'a>>::Item, StateError> {
                Ok(($(fetch_component::<$name>(store, entity, location)?,)+))
            }
        }
    };
}

tuple_entity_data!(A, B);
tuple_entity_data!(A, B, C);
tuple_entity_data!(A, B, C, D);
tuple_entity_data!(A, B, C, D, E);
tuple_entity_data!(A, B, C, D, E, F);
tuple_entity_data!(A, B, C, D, E, F, G);
tuple_entity_data!(A, B, C, D, E, F, G, H);

impl Store {
    /// 获取实体的只读借用引用视图 (`EntityRef`)。
    /// 完全零堆分配、零组件拷贝。若实体不存在或处于不活跃状态则返回 `None`。
    #[inline]
    pub fn entity(&self, id: EntityId) -> Option<EntityRef<'_>> {
        let loc = self.allocator.get_location(id)?;
        let arch = self.archetypes.get(loc.archetype_id as usize)?;
        if arch.entities().get(loc.row_index as usize) != Some(&id) {
            return None;
        }
        Some(EntityRef::new(self, id, loc))
    }

    /// 联合借用提取实体的多元组组件引用（零堆分配、零克隆，1次实体定位查找）。
    ///
    /// 若实体不存在返回 `StateError::EntityNotFound(id)`，若缺失组件返回 `StateError::ComponentNotFoundInEntity`。
    #[inline]
    pub fn extract<'a, E: EntityData<'a>>(&'a self, id: EntityId) -> Result<E::Item, StateError> {
        self.entity(id)
            .ok_or(StateError::EntityNotFound(id))?
            .extract::<E>()
    }

    /// 联合借用提取实体的多元组组件可选引用（实体不存在或缺失任一组件时返回 `None`，零堆分配）。
    #[inline]
    pub fn get_tuple<'a, E: EntityData<'a>>(&'a self, id: EntityId) -> Option<E::Item> {
        self.entity(id)?.get_tuple::<E>()
    }
}
