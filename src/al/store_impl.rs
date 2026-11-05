//! # StateFabric Abstraction Layer (AL) — Zero-Cost Native `Store` Implementation
//!
//! `sumer_state::Store` 直接实现 `StateSpace`（`WorldEngine`），且其 GAT 关联类型直接绑定为原生句柄：
//! - `type EntityRef<'a> = crate::EntityRef<'a>;`
//! - `type EntityMut<'a> = crate::EntityMut<'a>;`
//! - `type CommandBuffer = crate::CommandBuffer;`
//!
//! 对 `Store`、`EntityMut`、`crud` 与 `CommandBuffer` 生产内核保持 **零字段侵入、零旁路钩子、零二次包装**。

use rayon::prelude::*;

use crate::al::traits::{
    CommandBufferAccess, Component, EntityCommandsAccess, EntityMutAccess, EntityRefAccess,
    QueryFilter, StateSpace,
};
use crate::binding::EntityId;
use crate::error::StateError;
use crate::id::ComponentTypeId;
use crate::storage::archetype::Archetype;
use crate::storage::column::TypedColumn;
use crate::storage::core::store::Store;
use crate::storage::query::{EntityMut, EntityRef};
use crate::storage::stage::commands::{CommandBuffer, EntityCommands};

// ============================================================================
// 1. 原生句柄直接实现 GAT 门面契约 (EntityRef / EntityMut / EntityCommands / CommandBuffer)
// ============================================================================

impl<'a> EntityRefAccess<'a> for EntityRef<'a> {
    type Entity = EntityId;
    type Error = StateError;

    #[inline]
    fn id(&self) -> Self::Entity {
        EntityRef::id(self)
    }

    #[inline]
    fn has<T: Component>(&self) -> bool {
        EntityRef::has::<T>(self)
    }

    #[inline]
    fn get<T: Component>(&self) -> Option<&'a T> {
        EntityRef::get::<T>(self)
    }

    #[inline]
    fn try_get<T: Component>(&self) -> Result<&'a T, Self::Error> {
        EntityRef::try_get::<T>(self)
    }
}

impl<'a> EntityMutAccess<'a> for EntityMut<'a> {
    type Entity = EntityId;
    type Error = StateError;

    #[inline]
    fn id(&self) -> Self::Entity {
        EntityMut::id(self)
    }

    #[inline]
    fn has<T: Component>(&self) -> bool {
        EntityMut::has::<T>(self)
    }

    #[inline]
    fn get<T: Component>(&self) -> Option<&T> {
        EntityMut::get::<T>(self)
    }

    #[inline]
    fn try_get<T: Component>(&self) -> Result<&T, Self::Error> {
        EntityMut::try_get::<T>(self)
    }

    #[inline]
    fn get_mut<T: Component>(&mut self) -> Option<&mut T> {
        EntityMut::get_mut::<T>(self)
    }

    #[inline]
    fn set<T: Component>(&mut self, value: T) -> Result<Option<T>, Self::Error> {
        EntityMut::set::<T>(self, value)
    }

    #[inline]
    fn add<T: Component>(&mut self, value: T) -> Result<(), Self::Error> {
        EntityMut::add::<T>(self, value)
    }

    #[inline]
    fn replace<T: Component>(&mut self, value: T) -> Result<T, Self::Error> {
        EntityMut::replace::<T>(self, value)
    }

    #[inline]
    fn remove<T: Component>(&mut self) -> Result<T, Self::Error> {
        EntityMut::remove::<T>(self)
    }

    #[inline]
    fn modify<T: Component, R>(
        &mut self,
        f: impl FnOnce(&mut T) -> R,
    ) -> Result<R, Self::Error> {
        EntityMut::modify::<T, R>(self, f)
    }

    #[inline]
    fn despawn(self) -> Result<(), Self::Error> {
        EntityMut::despawn(self)
    }
}

impl<'a> EntityCommandsAccess<'a> for EntityCommands<'a> {
    type Entity = EntityId;

    #[inline]
    fn id(&self) -> Self::Entity {
        EntityCommands::id(self)
    }

    #[inline]
    fn add<T: Component>(&mut self, value: T) -> &mut Self {
        EntityCommands::add::<T>(self, value)
    }

    #[inline]
    fn replace<T: Component>(&mut self, value: T) -> &mut Self {
        EntityCommands::replace::<T>(self, value)
    }

    #[inline]
    fn set<T: Component>(&mut self, value: T) -> &mut Self {
        EntityCommands::set::<T>(self, value)
    }

    #[inline]
    fn remove<T: Component>(&mut self) -> &mut Self {
        EntityCommands::remove::<T>(self)
    }

    #[inline]
    fn despawn(&mut self) -> &mut Self {
        EntityCommands::despawn(self)
    }
}

impl CommandBufferAccess<Store> for CommandBuffer {
    type Entity = EntityId;
    type Error = StateError;
    type EntityCommands<'a> = EntityCommands<'a>;

    #[inline]
    fn entity(&mut self, id: Self::Entity) -> Self::EntityCommands<'_> {
        CommandBuffer::entity(self, id)
    }

    #[inline]
    fn create_empty(&mut self, id: Self::Entity) -> &mut Self {
        CommandBuffer::create_empty(self, id)
    }

    #[inline]
    fn despawn(&mut self, id: Self::Entity) -> &mut Self {
        CommandBuffer::despawn(self, id)
    }

    #[inline]
    fn is_empty(&self) -> bool {
        CommandBuffer::is_empty(self)
    }

    #[inline]
    fn apply_discard(self, store: &mut Store) -> Result<(), Self::Error> {
        CommandBuffer::apply_discard(self, store)
    }
}

// ============================================================================
// 2. 零拷贝列存连续块切片辅助函数
// ============================================================================

#[inline]
unsafe fn flat_col_slice<'a, T: Component>(arch: &'a Archetype, len: usize) -> Option<&'a [T]> {
    let cid = ComponentTypeId::of::<T>();
    if !arch.may_be_segmented_or_unaligned()
        && arch.flat_columns().len() == arch.component_types_slice().len()
    {
        let pos = arch.component_types_slice().iter().position(|&id| id == cid)?;
        let entry = arch.flat_columns()[pos];
        if entry.rust_type == Some(std::any::TypeId::of::<T>()) {
            let typed = unsafe { &*(entry.col_ptr as *const () as *const TypedColumn<T>) };
            return Some(&typed.as_slice()[..len]);
        }
    }
    arch.get_column::<T>().map(|c| &c.as_slice()[..len])
}

#[inline]
unsafe fn flat_col_slice_mut<'a, T: Component>(
    arch: &'a mut Archetype,
    len: usize,
) -> Option<&'a mut [T]> {
    let cid = ComponentTypeId::of::<T>();
    if !arch.may_be_segmented_or_unaligned()
        && arch.flat_columns().len() == arch.component_types_slice().len()
    {
        let pos = arch.component_types_slice().iter().position(|&id| id == cid)?;
        let entry = arch.flat_columns()[pos];
        if entry.rust_type == Some(std::any::TypeId::of::<T>()) {
            arch.bump_columns_version_in_place();
            let typed = unsafe { &mut *(entry.col_ptr as *mut () as *mut TypedColumn<T>) };
            return Some(&mut typed.as_mut_slice()[..len]);
        }
    }
    arch.get_column_mut::<T>()
        .map(|typed| &mut typed.as_mut_slice()[..len])
}

// ============================================================================
// 3. Layer 1: StateSpace for Store
// ============================================================================

impl StateSpace for Store {
    const ENGINE_NAME: &'static str = "StateFabric (Store)";

    type Entity = EntityId;
    type Error = StateError;
    type EntityRef<'a> = EntityRef<'a>;
    type EntityMut<'a> = EntityMut<'a>;
    type CommandBuffer = CommandBuffer;

    #[inline]
    fn entity_count(&self) -> usize {
        Store::len(self)
    }

    #[inline]
    fn contains_entity(&self, entity: Self::Entity) -> bool {
        Store::contains_entity(self, entity)
    }

    #[inline]
    fn alive_entities(&self) -> Vec<Self::Entity> {
        self.entities().collect()
    }

    #[inline]
    fn clear(&mut self) {
        Store::clear(self);
    }

    #[inline]
    fn spawn(&mut self) -> Result<Self::Entity, Self::Error> {
        Store::spawn_auto(self)
    }

    #[inline]
    fn spawn_at(&mut self, entity: Self::Entity) -> Result<Self::Entity, Self::Error> {
        Store::spawn_at(self, entity)
    }

    #[inline]
    fn despawn(&mut self, entity: Self::Entity) -> Result<(), Self::Error> {
        Store::despawn(self, entity)
    }

    #[inline]
    fn despawn_batch(&mut self, entities: &[Self::Entity]) -> Result<usize, Self::Error> {
        Store::despawn_batch(self, entities.iter().copied())
    }

    #[inline]
    fn entity(&self, id: Self::Entity) -> Option<Self::EntityRef<'_>> {
        Store::entity(self, id)
    }

    #[inline]
    fn entity_mut(&mut self, id: Self::Entity) -> Result<Self::EntityMut<'_>, Self::Error> {
        Store::entity_mut(self, id)
    }

    #[inline]
    fn get<T: Component>(&self, entity: Self::Entity) -> Option<&T> {
        Store::get::<T>(self, entity)
    }

    #[inline]
    fn get_mut<T: Component>(&mut self, entity: Self::Entity) -> Option<&mut T> {
        Store::get_mut::<T>(self, entity)
    }

    #[inline]
    fn has<T: Component>(&self, entity: Self::Entity) -> bool {
        Store::has::<T>(self, entity)
    }

    #[inline]
    fn set<T: Component>(&mut self, entity: Self::Entity, component: T) -> Result<(), Self::Error> {
        Store::set::<T>(self, entity, component).map(|_| ())
    }

    #[inline]
    fn add<T: Component>(&mut self, entity: Self::Entity, component: T) -> Result<(), Self::Error> {
        Store::add::<T>(self, entity, component)
    }

    #[inline]
    fn replace<T: Component>(
        &mut self,
        entity: Self::Entity,
        component: T,
    ) -> Result<T, Self::Error> {
        Store::replace::<T>(self, entity, component)
    }

    #[inline]
    fn remove<T: Component>(&mut self, entity: Self::Entity) -> Result<Option<T>, Self::Error> {
        Store::remove::<T>(self, entity)
    }

    #[inline]
    fn modify<T: Component, R>(
        &mut self,
        entity: Self::Entity,
        f: impl FnOnce(&mut T) -> R,
    ) -> Result<R, Self::Error> {
        Store::modify::<T, R>(self, entity, f)
    }

    #[inline]
    fn commands(&self) -> Self::CommandBuffer {
        Store::commands(self)
    }

    #[inline]
    fn for_each_chunk1<A: Component, F: QueryFilter>(
        &self,
        mut f: impl FnMut(&[Self::Entity], &[A]),
    ) {
        for arch in &self.archetypes {
            let len = arch.len();
            if len == 0 || !F::matches_chunk(arch) {
                continue;
            }
            if let Some(col_a) = unsafe { flat_col_slice::<A>(arch, len) } {
                f(arch.entities(), col_a);
            }
        }
    }

    #[inline]
    fn for_each_chunk2<A: Component, B: Component, F: QueryFilter>(
        &self,
        mut f: impl FnMut(&[Self::Entity], &[A], &[B]),
    ) {
        for arch in &self.archetypes {
            let len = arch.len();
            if len == 0 || !F::matches_chunk(arch) {
                continue;
            }
            unsafe {
                if let (Some(col_a), Some(col_b)) = (
                    flat_col_slice::<A>(arch, len),
                    flat_col_slice::<B>(arch, len),
                ) {
                    f(arch.entities(), col_a, col_b);
                }
            }
        }
    }

    #[inline]
    fn for_each_chunk3<A: Component, B: Component, C: Component, F: QueryFilter>(
        &self,
        mut f: impl FnMut(&[Self::Entity], &[A], &[B], &[C]),
    ) {
        for arch in &self.archetypes {
            let len = arch.len();
            if len == 0 || !F::matches_chunk(arch) {
                continue;
            }
            unsafe {
                if let (Some(col_a), Some(col_b), Some(col_c)) = (
                    flat_col_slice::<A>(arch, len),
                    flat_col_slice::<B>(arch, len),
                    flat_col_slice::<C>(arch, len),
                ) {
                    f(arch.entities(), col_a, col_b, col_c);
                }
            }
        }
    }

    #[inline]
    fn for_each_chunk4<A: Component, B: Component, C: Component, D: Component, F: QueryFilter>(
        &self,
        mut f: impl FnMut(&[Self::Entity], &[A], &[B], &[C], &[D]),
    ) {
        for arch in &self.archetypes {
            let len = arch.len();
            if len == 0 || !F::matches_chunk(arch) {
                continue;
            }
            unsafe {
                if let (Some(col_a), Some(col_b), Some(col_c), Some(col_d)) = (
                    flat_col_slice::<A>(arch, len),
                    flat_col_slice::<B>(arch, len),
                    flat_col_slice::<C>(arch, len),
                    flat_col_slice::<D>(arch, len),
                ) {
                    f(arch.entities(), col_a, col_b, col_c, col_d);
                }
            }
        }
    }

    #[inline]
    fn for_each_chunk5<
        A: Component,
        B: Component,
        C: Component,
        D: Component,
        G: Component,
        F: QueryFilter,
    >(
        &self,
        mut f: impl FnMut(&[Self::Entity], &[A], &[B], &[C], &[D], &[G]),
    ) {
        for arch in &self.archetypes {
            let len = arch.len();
            if len == 0 || !F::matches_chunk(arch) {
                continue;
            }
            unsafe {
                if let (Some(col_a), Some(col_b), Some(col_c), Some(col_d), Some(col_g)) = (
                    flat_col_slice::<A>(arch, len),
                    flat_col_slice::<B>(arch, len),
                    flat_col_slice::<C>(arch, len),
                    flat_col_slice::<D>(arch, len),
                    flat_col_slice::<G>(arch, len),
                ) {
                    f(arch.entities(), col_a, col_b, col_c, col_d, col_g);
                }
            }
        }
    }

    #[inline]
    fn for_each_chunk_mut1<A: Component, F: QueryFilter>(
        &mut self,
        mut f: impl FnMut(&[Self::Entity], &mut [A]),
    ) {
        let mut touched = false;
        for arch in &mut self.archetypes {
            let len = arch.len();
            if len == 0 || !F::matches_chunk(arch) {
                continue;
            }
            let entities_ptr = arch.entities().as_ptr();
            if let Some(col_a) = unsafe { flat_col_slice_mut::<A>(arch, len) } {
                let entities = unsafe { std::slice::from_raw_parts(entities_ptr, len) };
                f(entities, col_a);
                touched = true;
            }
        }
        if touched {
            self.epoch = self.epoch.wrapping_add(1);
        }
    }

    #[inline]
    fn for_each_chunk_mut2<A: Component, B: Component, F: QueryFilter>(
        &mut self,
        mut f: impl FnMut(&[Self::Entity], &mut [A], &[B]),
    ) {
        assert_ne!(
            ComponentTypeId::of::<A>(),
            ComponentTypeId::of::<B>(),
            "Cannot borrow the same component type A and B as both mutable and immutable"
        );
        let mut touched = false;
        for arch in &mut self.archetypes {
            let len = arch.len();
            if len == 0 || !F::matches_chunk(arch) {
                continue;
            }
            let entities_ptr = arch.entities().as_ptr();
            let col_b_ptr = match unsafe { flat_col_slice::<B>(arch, len) } {
                Some(s) => s.as_ptr(),
                None => continue,
            };
            if let Some(col_a) = unsafe { flat_col_slice_mut::<A>(arch, len) } {
                let entities = unsafe { std::slice::from_raw_parts(entities_ptr, len) };
                let col_b = unsafe { std::slice::from_raw_parts(col_b_ptr, len) };
                f(entities, col_a, col_b);
                touched = true;
            }
        }
        if touched {
            self.epoch = self.epoch.wrapping_add(1);
        }
    }

    #[inline]
    fn for_each_chunk_mut3<A: Component, B: Component, C: Component, F: QueryFilter>(
        &mut self,
        mut f: impl FnMut(&[Self::Entity], &mut [A], &mut [B], &[C]),
    ) {
        let cid_a = ComponentTypeId::of::<A>();
        let cid_b = ComponentTypeId::of::<B>();
        let cid_c = ComponentTypeId::of::<C>();
        assert!(
            cid_a != cid_b && cid_a != cid_c && cid_b != cid_c,
            "Cannot alias component types in for_each_chunk_mut3"
        );
        let mut touched = false;
        for arch in &mut self.archetypes {
            let len = arch.len();
            if len == 0 || !F::matches_chunk(arch) {
                continue;
            }
            let entities_ptr = arch.entities().as_ptr();
            let col_c_ptr = match unsafe { flat_col_slice::<C>(arch, len) } {
                Some(s) => s.as_ptr(),
                None => continue,
            };
            let col_a_ptr = match unsafe { flat_col_slice_mut::<A>(arch, len) } {
                Some(s) => s.as_mut_ptr(),
                None => continue,
            };
            let col_b_ptr = match unsafe { flat_col_slice_mut::<B>(arch, len) } {
                Some(s) => s.as_mut_ptr(),
                None => continue,
            };
            unsafe {
                let entities = std::slice::from_raw_parts(entities_ptr, len);
                let col_a = std::slice::from_raw_parts_mut(col_a_ptr, len);
                let col_b = std::slice::from_raw_parts_mut(col_b_ptr, len);
                let col_c = std::slice::from_raw_parts(col_c_ptr, len);
                f(entities, col_a, col_b, col_c);
            }
            touched = true;
        }
        if touched {
            self.epoch = self.epoch.wrapping_add(1);
        }
    }

    fn par_for_each_chunk_mut1<A: Component, F: QueryFilter>(
        &mut self,
        f: impl Fn(&[Self::Entity], &mut [A]) + Send + Sync,
    ) {
        const SUBCHUNK: usize = 4096;
        let mut tasks: Vec<(usize, usize, usize)> = Vec::new();
        for arch in &mut self.archetypes {
            let len = arch.len();
            if len == 0 || !F::matches_chunk(arch) {
                continue;
            }
            let ent_ptr = arch.entities().as_ptr() as usize;
            if let Some(col_a) = unsafe { flat_col_slice_mut::<A>(arch, len) } {
                let a_ptr = col_a.as_mut_ptr() as usize;
                let mut offset = 0;
                while offset < len {
                    let take = (len - offset).min(SUBCHUNK);
                    tasks.push((
                        ent_ptr + offset * std::mem::size_of::<EntityId>(),
                        a_ptr + offset * std::mem::size_of::<A>(),
                        take,
                    ));
                    offset += take;
                }
            }
        }
        if !tasks.is_empty() {
            tasks.into_par_iter().for_each(|(e_addr, a_addr, len)| unsafe {
                let ents = std::slice::from_raw_parts(e_addr as *const EntityId, len);
                let col_a = std::slice::from_raw_parts_mut(a_addr as *mut A, len);
                f(ents, col_a);
            });
            self.epoch = self.epoch.wrapping_add(1);
        }
    }

    fn par_for_each_chunk_mut2<A: Component, B: Component, F: QueryFilter>(
        &mut self,
        f: impl Fn(&[Self::Entity], &mut [A], &[B]) + Send + Sync,
    ) {
        const SUBCHUNK: usize = 4096;
        let cid_a = ComponentTypeId::of::<A>();
        let cid_b = ComponentTypeId::of::<B>();
        assert_ne!(cid_a, cid_b, "Cannot alias A and B in par_for_each_chunk_mut2");
        let mut tasks: Vec<(usize, usize, usize, usize)> = Vec::new();
        for arch in &mut self.archetypes {
            let len = arch.len();
            if len == 0 || !F::matches_chunk(arch) {
                continue;
            }
            let ent_ptr = arch.entities().as_ptr() as usize;
            let b_ptr = match unsafe { flat_col_slice::<B>(arch, len) } {
                Some(s) => s.as_ptr() as usize,
                None => continue,
            };
            if let Some(col_a) = unsafe { flat_col_slice_mut::<A>(arch, len) } {
                let a_ptr = col_a.as_mut_ptr() as usize;
                let mut offset = 0;
                while offset < len {
                    let take = (len - offset).min(SUBCHUNK);
                    tasks.push((
                        ent_ptr + offset * std::mem::size_of::<EntityId>(),
                        a_ptr + offset * std::mem::size_of::<A>(),
                        b_ptr + offset * std::mem::size_of::<B>(),
                        take,
                    ));
                    offset += take;
                }
            }
        }
        if !tasks.is_empty() {
            tasks
                .into_par_iter()
                .for_each(|(e_addr, a_addr, b_addr, len)| unsafe {
                    let ents = std::slice::from_raw_parts(e_addr as *const EntityId, len);
                    let col_a = std::slice::from_raw_parts_mut(a_addr as *mut A, len);
                    let col_b = std::slice::from_raw_parts(b_addr as *const B, len);
                    f(ents, col_a, col_b);
                });
            self.epoch = self.epoch.wrapping_add(1);
        }
    }
}
