//! 实体构建器与类型安全转换特征 (Entity & IntoEntity)

use super::batch::EntityBundle;
use crate::binding::EntityId;
use crate::context::ModelComponent;
use crate::error::StateError;
use serde::Serialize;
use serde::de::DeserializeOwned;

/// 实体构建载体 (Entity)。
///
/// 封装目标实体 ID 与其携带的组件束 (`EntityBundle`)，用于以流式链式调用的方式
/// 在实体创建时装配所有初始组件。
///
/// 实体构建器在装配阶段严格保证组件唯一性，若传入重复类型的组件将记录延迟错误并在提交或提取时返回 `ComponentDuplicateInPatch`。
#[derive(Debug)]
pub struct Entity {
    pub(crate) id: EntityId,
    pub(crate) bundle: EntityBundle,
    pub(crate) deferred_error: Option<StateError>,
}

impl Entity {
    /// 创建指定 ID 的空实体构建器。
    #[inline]
    pub fn new(id: EntityId) -> Self {
        Self {
            id,
            bundle: EntityBundle::new(),
            deferred_error: None,
        }
    }

    /// 链式装配强类型组件。若已有同类型组件，记录延迟错误并在提交/提取时返回 `ComponentDuplicateInPatch`。
    #[inline]
    pub fn with<T: ModelComponent + Serialize + DeserializeOwned + 'static>(
        mut self,
        value: T,
    ) -> Self {
        if self.deferred_error.is_some() {
            return self;
        }
        let bundle = std::mem::take(&mut self.bundle);
        match bundle.with(value) {
            Ok(bundle) => {
                self.bundle = bundle;
            }
            Err(StateError::ComponentDuplicateInPatch { type_name, .. }) => {
                self.deferred_error = Some(StateError::ComponentDuplicateInPatch {
                    entity: self.id,
                    type_name,
                });
            }
            Err(err) => {
                self.deferred_error = Some(err);
            }
        }
        self
    }

    /// 链式装配强类型实体关系对组件 (`Pair<R>`)。
    #[inline]
    pub fn with_pair<R: ModelComponent + Serialize + DeserializeOwned + 'static>(
        self,
        relation: R,
        target: EntityId,
    ) -> Self {
        self.with(crate::al::Pair::new(relation, target))
    }

    /// 链式装配零大小编译期槽位标签组件 (`Tag<N>`)。
    #[inline]
    pub fn with_tag<const N: u8>(self) -> Self {
        self.with(crate::al::Tag::<N>)
    }

    /// 获取实体 ID
    #[inline]
    pub fn id(&self) -> EntityId {
        self.id
    }

    /// 分解实体为实体 ID 与组件束（若链式装配存在冲突则在此返回错误）
    #[inline]
    pub fn into_bundle(self) -> Result<(EntityId, EntityBundle), StateError> {
        if let Some(err) = self.deferred_error {
            return Err(err);
        }
        Ok((self.id, self.bundle))
    }

    /// 分解实体为实体 ID 与组件束（包内受控提取，调用方须已通过 `into_entity` 校验）
    #[inline]
    pub(crate) fn into_parts(self) -> (EntityId, EntityBundle) {
        (self.id, self.bundle)
    }
}

/// 统一实体输入转换特征 (IntoEntity)。
///
/// 统一抽象了 `Entity` 构建器及从 0 到 64 元组件元组 `(EntityId, (T1, ..., Tn))` 的输入形式，
/// 允许直接将多元组或实体构建器传递给 `store.insert(...)` 或 `commands.insert(...)`。
pub trait IntoEntity: Sized {
    fn into_entity(self) -> Result<Entity, StateError>;

    #[doc(hidden)]
    #[inline]
    fn insert_into_store(self, store: &mut super::core::store::Store) -> Result<(), StateError> {
        let entity = self.into_entity()?;
        let (id, bundle) = entity.into_parts();
        store.create_entity(id, bundle)
    }
}

impl IntoEntity for Entity {
    #[inline]
    fn into_entity(self) -> Result<Entity, StateError> {
        if let Some(err) = self.deferred_error {
            return Err(err);
        }
        Ok(self)
    }
}

impl IntoEntity for (EntityId, EntityBundle) {
    #[inline]
    fn into_entity(self) -> Result<Entity, StateError> {
        Ok(Entity {
            id: self.0,
            bundle: self.1,
            deferred_error: None,
        })
    }

    #[inline]
    fn insert_into_store(self, store: &mut super::core::store::Store) -> Result<(), StateError> {
        store.create_entity(self.0, self.1)
    }
}

impl IntoEntity for (EntityId, ()) {
    #[inline]
    fn into_entity(self) -> Result<Entity, StateError> {
        Ok(Entity::new(self.0))
    }

    #[inline]
    fn insert_into_store(self, store: &mut super::core::store::Store) -> Result<(), StateError> {
        store.spawn(self.0).map(|_| ())
    }
}

macro_rules! impl_into_entity_for_tuple {
    ($head:ident $(, $tail:ident)* $(,)?) => {
        impl<$head, $($tail),*> IntoEntity for (EntityId, ($head, $($tail,)*))
        where
            $head: ModelComponent + Serialize + DeserializeOwned + 'static,
            $($tail: ModelComponent + Serialize + DeserializeOwned + 'static,)*
        {
            #[inline]
            fn into_entity(self) -> Result<Entity, StateError> {
                #[allow(non_snake_case)]
                let (id, ($head, $($tail,)*)) = self;
                #[allow(unused_mut)]
                let mut entity = Entity::new(id).with($head);
                $(entity = entity.with($tail);)*
                entity.into_entity()
            }

            #[inline]
            fn insert_into_store(
                self,
                store: &mut super::core::store::Store,
            ) -> Result<(), StateError> {
                let mut cids = [
                    crate::id::ComponentTypeId::of::<$head>(),
                    $(crate::id::ComponentTypeId::of::<$tail>(),)*
                ];
                let n = cids.len();
                cids.sort_unstable();
                let mut has_dup = false;
                let mut i = 1;
                while i < n {
                    if cids[i - 1] == cids[i] {
                        has_dup = true;
                        break;
                    }
                    i += 1;
                }
                if !has_dup {
                    let target_arch_id = if let Some(cached_id) = store.last_created_archetype {
                        if (cached_id as usize) < store.archetypes.len()
                            && store.archetypes[cached_id as usize].component_types_slice() == &cids[..]
                        {
                            Some(cached_id)
                        } else {
                            None
                        }
                    } else {
                        None
                    };
                    let target_arch_id = match target_arch_id {
                        Some(aid) => Some(aid),
                        None if store.archetypes.len() <= 32 => {
                            let found = store
                                .archetypes
                                .iter()
                                .position(|a| a.component_types_slice() == &cids[..])
                                .map(|idx| idx as u32);
                            if let Some(aid) = found {
                                store.last_created_archetype = Some(aid);
                            }
                            found
                        }
                        None => None,
                    };
                    if let Some(target_arch_id) = target_arch_id {
                        let arch = &store.archetypes[target_arch_id as usize];
                        if !arch.may_be_segmented_or_unaligned() && arch.flat_columns().len() == n {
                            let flat = arch.flat_columns();
                            let idx_head = cids
                                .iter()
                                .position(|&c| c == crate::id::ComponentTypeId::of::<$head>())
                                .unwrap();
                            let types_match = unsafe {
                                flat.get_unchecked(idx_head).rust_type == Some(std::any::TypeId::of::<$head>())
                                $(&& {
                                    let idx_t = cids
                                        .iter()
                                        .position(|&c| c == crate::id::ComponentTypeId::of::<$tail>())
                                        .unwrap();
                                    flat.get_unchecked(idx_t).rust_type == Some(std::any::TypeId::of::<$tail>())
                                })*
                            };
                            if types_match {
                                store.check_poisoned()?;
                                let row_usize = arch.len();
                                if row_usize >= u32::MAX as usize {
                                    return Err(StateError::CapacityExhausted(
                                        "Archetype rows exhausted".into(),
                                    ));
                                }
                                let next_epoch = store.next_epoch()?;
                                #[allow(non_snake_case)]
                                let (id, ($head, $($tail,)*)) = self;
                                let local_slot = store.allocator.allocate_unique_with_location(
                                    id,
                                    crate::storage::allocator::EntityLocation::new(
                                        target_arch_id,
                                        row_usize as u32,
                                    ),
                                )?;
                                let mut poison_guard =
                                    super::core::store::CorePoisonGuard::new(&store.poisoned);
                                let target_arch = &mut store.archetypes[target_arch_id as usize];
                                let flat = target_arch.flat_columns();
                                unsafe {
                                    let entry_head = *flat.get_unchecked(idx_head);
                                    let col_head = &mut *(entry_head.col_ptr as *mut ()
                                        as *mut crate::storage::column::TypedColumn<$head>);
                                    col_head.push_direct_owned_unchecked($head);
                                    $(
                                        let idx_t = cids
                                            .iter()
                                            .position(|&c| c == crate::id::ComponentTypeId::of::<$tail>())
                                            .unwrap();
                                        let entry_t = *flat.get_unchecked(idx_t);
                                        let col_t = &mut *(entry_t.col_ptr as *mut ()
                                            as *mut crate::storage::column::TypedColumn<$tail>);
                                        col_t.push_direct_owned_unchecked($tail);
                                    )*
                                }
                                target_arch.bump_columns_version_in_place();
                                let row = target_arch.push_entity_with_slot(id, local_slot) as u32;
                                debug_assert_eq!(row, row_usize as u32);
                                store.epoch = next_epoch;
                                poison_guard.disarm();
                                return Ok(());
                            }
                        }
                    }
                }
                let entity = self.into_entity()?;
                let (id, bundle) = entity.into_parts();
                store.create_entity(id, bundle)
            }
        }

        impl_into_entity_for_tuple!($($tail),*);
    };
    () => {};
}

impl_into_entity_for_tuple!(
    T64, T63, T62, T61, T60, T59, T58, T57, T56, T55, T54, T53, T52, T51, T50, T49,
    T48, T47, T46, T45, T44, T43, T42, T41, T40, T39, T38, T37, T36, T35, T34, T33,
    T32, T31, T30, T29, T28, T27, T26, T25, T24, T23, T22, T21, T20, T19, T18, T17,
    T16, T15, T14, T13, T12, T11, T10, T9, T8, T7, T6, T5, T4, T3, T2, T1
);

/// 强类型组件束转换特征 (ComponentBundle)。
///
/// 支持 `()`、1..=16 元组件元组 `(T1, ..., Tn)` 以及 `EntityBundle`，供 `Store::insert_batch` 与 `StateSpace::insert_batch` 使用。
pub trait ComponentBundle: Sized {
    fn apply_to_entity(self, entity: Entity) -> Entity;

    #[doc(hidden)]
    #[inline]
    fn insert_with_id_into_store(
        self,
        id: EntityId,
        store: &mut super::core::store::Store,
    ) -> Result<(), StateError> {
        self.apply_to_entity(Entity::new(id)).insert_into_store(store)
    }
}

impl ComponentBundle for () {
    #[inline]
    fn apply_to_entity(self, entity: Entity) -> Entity {
        entity
    }

    #[inline]
    fn insert_with_id_into_store(
        self,
        id: EntityId,
        store: &mut super::core::store::Store,
    ) -> Result<(), StateError> {
        (id, ()).insert_into_store(store)
    }
}

impl ComponentBundle for EntityBundle {
    #[inline]
    fn apply_to_entity(self, mut entity: Entity) -> Entity {
        if entity.deferred_error.is_some() {
            return entity;
        }
        entity.bundle = self;
        entity
    }

    #[inline]
    fn insert_with_id_into_store(
        self,
        id: EntityId,
        store: &mut super::core::store::Store,
    ) -> Result<(), StateError> {
        store.create_entity(id, self)
    }
}

/// 多态实体生成载荷特征 (SpawnBundle)。
///
/// 统一支持：
/// - `()`：自动分配实体 ID 并在空原型（Archetype 0）生成；
/// - `(T1, ..., Tn)`（1..=16 元组）：自动分配实体 ID 并一步写入目标原型（0 次中间迁移）；
/// - `EntityBundle`：自动分配实体 ID 并一次性写入组件束；
/// - `Entity`：按构建器内指定的实体 ID 与组件束生成；
/// - `EntityId`：在指定实体 ID 上生成空实体（向后兼容 `store.spawn(id)`）。
pub trait SpawnBundle: Sized {
    fn spawn_in_store(self, store: &mut super::core::store::Store) -> Result<EntityId, StateError>;
}

impl SpawnBundle for EntityId {
    #[inline]
    fn spawn_in_store(self, store: &mut super::core::store::Store) -> Result<EntityId, StateError> {
        store.spawn_with_id(self)
    }
}

impl SpawnBundle for () {
    #[inline]
    fn spawn_in_store(self, store: &mut super::core::store::Store) -> Result<EntityId, StateError> {
        store.spawn_auto()
    }
}

impl SpawnBundle for EntityBundle {
    #[inline]
    fn spawn_in_store(self, store: &mut super::core::store::Store) -> Result<EntityId, StateError> {
        let id = store.next_auto_entity_id()?;
        store.create_entity(id, self)?;
        Ok(id)
    }
}

impl SpawnBundle for Entity {
    #[inline]
    fn spawn_in_store(self, store: &mut super::core::store::Store) -> Result<EntityId, StateError> {
        let entity = self.into_entity()?;
        let id = entity.id();
        entity.insert_into_store(store)?;
        Ok(id)
    }
}

macro_rules! impl_component_bundle_for_tuple {
    ($($name:ident),+) => {
        impl<$($name),+> ComponentBundle for ($($name,)+)
        where
            $($name: ModelComponent + Serialize + DeserializeOwned + 'static,)+
        {
            #[inline]
            fn apply_to_entity(self, mut entity: Entity) -> Entity {
                #[allow(non_snake_case)]
                let ($($name,)+) = self;
                $(entity = entity.with($name);)+
                entity
            }

            #[inline]
            fn insert_with_id_into_store(
                self,
                id: EntityId,
                store: &mut super::core::store::Store,
            ) -> Result<(), StateError> {
                (id, self).insert_into_store(store)
            }
        }

        impl<$($name),+> SpawnBundle for ($($name,)+)
        where
            $($name: ModelComponent + Serialize + DeserializeOwned + 'static,)+
        {
            #[inline]
            fn spawn_in_store(
                self,
                store: &mut super::core::store::Store,
            ) -> Result<EntityId, StateError> {
                let id = store.next_auto_entity_id()?;
                (id, self).insert_into_store(store)?;
                Ok(id)
            }
        }
    };
}

impl_component_bundle_for_tuple!(T1);
impl_component_bundle_for_tuple!(T1, T2);
impl_component_bundle_for_tuple!(T1, T2, T3);
impl_component_bundle_for_tuple!(T1, T2, T3, T4);
impl_component_bundle_for_tuple!(T1, T2, T3, T4, T5);
impl_component_bundle_for_tuple!(T1, T2, T3, T4, T5, T6);
impl_component_bundle_for_tuple!(T1, T2, T3, T4, T5, T6, T7);
impl_component_bundle_for_tuple!(T1, T2, T3, T4, T5, T6, T7, T8);
impl_component_bundle_for_tuple!(T1, T2, T3, T4, T5, T6, T7, T8, T9);
impl_component_bundle_for_tuple!(T1, T2, T3, T4, T5, T6, T7, T8, T9, T10);
impl_component_bundle_for_tuple!(T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11);
impl_component_bundle_for_tuple!(T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12);
impl_component_bundle_for_tuple!(T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13);
impl_component_bundle_for_tuple!(T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14);
impl_component_bundle_for_tuple!(T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15);
impl_component_bundle_for_tuple!(T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15, T16);

impl super::core::store::Store {
    /// 插入新实体及其携带的初始组件束。
    ///
    /// 支持直接传入 `Entity` 或多元组 `(EntityId, (T1, ..., Tn))`。
    /// 若实体已存在，严格返回 `EntityAlreadyExists(id)`。
    #[inline]
    pub fn insert(&mut self, input: impl IntoEntity) -> Result<(), StateError> {
        input.insert_into_store(self)
    }

    /// 批量插入实体及其初始组件束（自动预留分配器容量并走原型缓存极速直推通道）。
    #[inline]
    pub fn insert_batch<B: ComponentBundle>(
        &mut self,
        items: impl IntoIterator<Item = (EntityId, B)>,
    ) -> Result<usize, StateError> {
        let iter = items.into_iter();
        let (lower, _) = iter.size_hint();
        if lower > 0 {
            self.reserve_entities(lower)?;
        }
        let mut count = 0usize;
        for (id, bundle) in iter {
            bundle.insert_with_id_into_store(id, self)?;
            count += 1;
        }
        Ok(count)
    }

    /// 批量销毁实体，返回实际成功销毁的活跃实体数量。
    #[inline]
    pub fn despawn_batch(
        &mut self,
        entities: impl IntoIterator<Item = EntityId>,
    ) -> Result<usize, StateError> {
        let mut count = 0usize;
        for id in entities {
            if self.contains_entity(id) {
                self.delete_entity(id)?;
                count += 1;
            }
        }
        Ok(count)
    }
}
