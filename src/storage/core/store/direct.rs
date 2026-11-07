//! 强类型直达批量更新通道 (Typed Direct Bypass)
//!
//! 在极速吞吐与确定性丢弃旧值 (Discard Mode) 场景下，绕过类型擦除与反射抽象，
//! 将强类型组件批次直接以最低硬件开销写入物理列存。
//!
//! ### 安全与性能不变式
//! 1. **前置毒化防护**：进入时校验 `check_poisoned()?`，保障故障隔离；
//! 2. **代号原子预检**：提前校验并计算 `next_epoch`，杜绝中途代号溢出；
//! 3. **Panic 与故障毒化防线**：使用 `CorePoisonGuard`，若发生用户 Drop Panic 或中途校验失败，自动毒化 Store 实行故障隔离；
//! 4. **局部缓存预绑定**：相同 Archetype 连续命中时零 B 树查询与零重复索引寻址；
//! 5. **实体与行映射强校验**：严格校验 `allocator.get_location` 与 `arch.entities()[row] == entity`；
//! 6. **分级物理列写入**：
//!    - POD 类型 (`!needs_drop::<T>()`)：使用 `copy_nonoverlapping` 单向 Store；
//!    - Non-POD 类型 (`needs_drop::<T>()`)：使用 `ptr::replace` 写入新值并安全 `drop(old)`；
//! 7. **流式直达发布**：仅在产生有效物理写入且全部实体写入成功后推进代号并解毒；若中途失败则 Store 进入毒化状态以隔离脏数据。

use super::Store;
use crate::binding::EntityId;
use crate::context::ModelComponent;
use crate::error::StateError;
use serde::de::DeserializeOwned;
use serde::Serialize;

/// 强类型直达批量更新批次 (Typed Batch)。
#[derive(Debug, Clone)]
pub struct TypedBatch<Item> {
    pub(crate) entries: Vec<Item>,
}

impl<Item> TypedBatch<Item> {
    #[inline]
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
        }
    }

    #[inline]
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            entries: Vec::with_capacity(capacity),
        }
    }

    #[inline]
    pub fn reserve(&mut self, additional: usize) {
        self.entries.reserve(additional);
    }

    #[inline]
    pub fn record(&mut self, item: Item) {
        self.entries.push(item);
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    #[inline]
    pub fn clear(&mut self) {
        self.entries.clear();
    }

    #[inline]
    pub fn into_entries(self) -> Vec<Item> {
        self.entries
    }

    #[inline]
    pub fn as_slice(&self) -> &[Item] {
        &self.entries
    }
}

impl<Item> Default for TypedBatch<Item> {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

impl<Item> From<Vec<Item>> for TypedBatch<Item> {
    #[inline]
    fn from(entries: Vec<Item>) -> Self {
        Self { entries }
    }
}

impl<Item> From<TypedBatch<Item>> for Vec<Item> {
    #[inline]
    fn from(batch: TypedBatch<Item>) -> Self {
        batch.entries
    }
}

impl<Item> IntoIterator for TypedBatch<Item> {
    type Item = Item;
    type IntoIter = std::vec::IntoIter<Item>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.entries.into_iter()
    }
}

impl<'a, Item> IntoIterator for &'a TypedBatch<Item> {
    type Item = &'a Item;
    type IntoIter = std::slice::Iter<'a, Item>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.entries.iter()
    }
}

/// 强类型直达更新事务收据 (Typed Receipt)。
#[derive(Debug, Clone)]
pub struct TypedReceipt<Item> {
    pub(crate) previous_values: Vec<Item>,
}

impl<Item> TypedReceipt<Item> {
    #[inline]
    pub fn len(&self) -> usize {
        self.previous_values.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.previous_values.is_empty()
    }

    #[inline]
    pub fn previous_values(&self) -> &[Item] {
        &self.previous_values
    }

    #[inline]
    pub fn into_inverse(self) -> TypedBatch<Item> {
        TypedBatch {
            entries: self.previous_values,
        }
    }
}

/// 强类型直达批量更新 Trait (Typed Direct Bypass)。
///
/// 允许零中间分配、零反射开销地将强类型组件批次直接单向写回物理列存。
pub trait TypedDirectBatchUpdate: Sized + 'static {
    /// 执行强类型直达批量更新并丢弃旧值 (Discard Mode)。
    fn apply_batch_discard(
        store: &mut Store,
        updates: impl Iterator<Item = Self>,
    ) -> Result<(), StateError>;

    /// 执行强类型直达批量更新并保留旧值收据 (Keep Mode)。
    fn apply_batch_keep(
        store: &mut Store,
        updates: impl Iterator<Item = Self>,
    ) -> Result<TypedReceipt<Self>, StateError>;

    /// 内部统一处理实现（通过编译期 const KEEP 单态化分级展开）。
    #[doc(hidden)]
    fn apply_batch_direct_internal<const KEEP: bool>(
        store: &mut Store,
        updates: impl Iterator<Item = Self>,
    ) -> Result<Option<TypedReceipt<Self>>, StateError>;
}

/// 物理列槽位极速写入底层辅助函数（丢弃旧值模式）。
///
/// 依据类型是否包含析构逻辑，由 LLVM 编译期单态化常量折叠消除无关分支：
/// - 若 `!needs_drop::<T>()`（POD）：使用 `std::ptr::write` 执行单向 Store；
/// - 若 `needs_drop::<T>()`（Non-POD）：使用 `ptr::replace` 写入新值并安全 `drop(old)`。
#[inline(always)]
unsafe fn write_slot<T>(col_ptr: *mut T, row: usize, value: T) {
    // SAFETY: 调用方保证 col_ptr 有效、row 处于物理列有效范围内，且生命周期受排他写借用约束。
    unsafe {
        let dst_ptr = col_ptr.add(row);
        if !std::mem::needs_drop::<T>() {
            std::ptr::write(dst_ptr, value);
        } else {
            let old = std::ptr::replace(dst_ptr, value);
            drop(old);
        }
    }
}

/// 物理列槽位极速置换底层辅助函数（保留旧值模式）。
///
/// 无论类型是否为 POD，均使用 `ptr::replace` 原位置换并提取旧值所有权返回。
#[inline(always)]
unsafe fn write_slot_keep<T>(col_ptr: *mut T, row: usize, value: T) -> T {
    // SAFETY: 调用方保证 col_ptr 有效、row 处于物理列有效范围内，且生命周期受排他写借用约束。
    unsafe {
        let dst_ptr = col_ptr.add(row);
        std::ptr::replace(dst_ptr, value)
    }
}

#[doc(hidden)]
#[inline(always)]
pub(crate) unsafe fn write_slot_for_macro<T>(col_ptr: *mut T, row: usize, value: T) {
    // SAFETY: 透传给 write_slot，前置安全契约由调用点保持。
    unsafe { write_slot(col_ptr, row, value) }
}

#[doc(hidden)]
#[inline(always)]
pub(crate) unsafe fn write_slot_keep_for_macro<T>(col_ptr: *mut T, row: usize, value: T) -> T {
    // SAFETY: 透传给 write_slot_keep，前置安全契约由调用点保持。
    unsafe { write_slot_keep(col_ptr, row, value) }
}

///// 强类型直达批量更新元组实现宏。
///
/// 为 `(EntityId, T0)` 到 `(EntityId, T0, ..., T15)` 提供统一的直接物理批量写回通道。
macro_rules! impl_typed_direct_batch {
    ($($idx:tt: $T:ident),+) => {
        impl<$($T),+> $crate::storage::core::store::direct::TypedDirectBatchUpdate for ($crate::binding::EntityId, $($T,)+)
        where
            $($T: $crate::context::ModelComponent + serde::Serialize + serde::de::DeserializeOwned + 'static,)+
        {
            #[inline]
            fn apply_batch_discard(
                store: &mut $crate::storage::core::store::Store,
                updates: impl Iterator<Item = Self>,
            ) -> Result<(), $crate::error::StateError> {
                Self::apply_batch_direct_internal::<false>(store, updates).map(|_| ())
            }

            #[inline]
            fn apply_batch_keep(
                store: &mut $crate::storage::core::store::Store,
                updates: impl Iterator<Item = Self>,
            ) -> Result<$crate::storage::core::store::direct::TypedReceipt<Self>, $crate::error::StateError> {
                Self::apply_batch_direct_internal::<true>(store, updates)
                    .map(|opt| opt.expect("keep mode returns Some receipt"))
            }

            #[allow(non_snake_case)]
            #[inline(always)]
            fn apply_batch_direct_internal<const KEEP: bool>(
                store: &mut $crate::storage::core::store::Store,
                updates: impl Iterator<Item = Self>,
            ) -> Result<Option<$crate::storage::core::store::direct::TypedReceipt<Self>>, $crate::error::StateError> {
                // 1. 毒化检查与代号预检
                store.check_poisoned()?;
                let next_epoch = store.next_epoch()?;
                let mut poison_guard = $crate::storage::core::store::CorePoisonGuard::new(&store.poisoned);

                // 2. 静态组件类型互斥校验（Anti-aliasing，杜绝元组内重复组件导致的物理别名写冲突）
                const ARITY: usize = 0 $(+ { let _ = stringify!($idx); 1 })+;
                if ARITY > 1 {
                    let type_ids = [$($crate::id::ComponentTypeId::of::<$T>()),+];
                    let type_names = [$(std::any::type_name::<$T>()),+];
                    for i in 0..type_ids.len() {
                        for j in (i + 1)..type_ids.len() {
                            if type_ids[i] == type_ids[j] {
                                return Err($crate::error::StateError::ComponentDuplicateInPatch {
                                    entity: $crate::binding::EntityId::from_u128(0),
                                    type_name: type_names[i].to_string(),
                                });
                            }
                        }
                    }
                }

                // 3. 局部缓存与收据容器初始化
                struct ArchetypeCache<$($T),+> {
                    arch_id: u32,
                    entities: *const $crate::binding::EntityId,
                    entities_len: usize,
                    cols: ($(*mut $T,)+),
                    chunk_epochs: [*mut u64; ARITY],
                }

                let mut cache: Option<ArchetypeCache<$($T),+>> = None;
                let mut previous_values = if KEEP {
                    let (lower_bound, _) = updates.size_hint();
                    Vec::with_capacity(lower_bound)
                } else {
                    Vec::new()
                };
                let mut written_count: usize = 0;

                // 4. 极速直达迭代
                for item in updates {
                    let (entity, $($T,)+) = item;

                    let location = store
                        .allocator
                        .get_location(entity)
                        .ok_or($crate::error::StateError::EntityNotFound(entity))?;

                    let current_arch_id = location.archetype_id;

                    // 检查局部缓存命中：相同 Archetype 连续命中时零 B 树查询
                    let (entities_ptr, entities_len, cols, chunk_epochs) = match &cache {
                        Some(c) if c.arch_id == current_arch_id => {
                            (c.entities, c.entities_len, c.cols, c.chunk_epochs)
                        }
                        _ => {
                            let arch = store
                                .archetypes
                                .get_mut(current_arch_id as usize)
                                .ok_or_else(|| {
                                    $crate::error::StateError::AccessDenied(format!(
                                        "Entity {entity} references an unknown archetype"
                                    ))
                                })?;
                            let entities_ptr = arch.entities().as_ptr();
                            let entities_len = arch.entities().len();
                            let cols = (
                                $({
                                    let typed_col = arch.get_typed_column_mut_ptr::<$T>(
                                        $crate::id::ComponentTypeId::of::<$T>(),
                                        entity,
                                    )?;
                                    let col_slice = unsafe { (*typed_col).as_mut_slice() };
                                    if col_slice.len() != entities_len {
                                        return Err($crate::error::StateError::AccessDenied(format!(
                                            "Archetype {current_arch_id} column row count {} does not match entity count {entities_len}",
                                            col_slice.len()
                                        )));
                                    }
                                    col_slice.as_mut_ptr()
                                },)+
                            );
                            let chunk_epochs = [
                                $(
                                    arch.prepare_column_chunk_epochs_mut_ptr(
                                        $crate::id::ComponentTypeId::of::<$T>(),
                                    )
                                ),+
                            ];
                            cache = Some(ArchetypeCache {
                                arch_id: current_arch_id,
                                entities: entities_ptr,
                                entities_len,
                                cols,
                                chunk_epochs,
                            });
                            (entities_ptr, entities_len, cols, chunk_epochs)
                        }
                    };

                    // 校验行界限与实体一致性
                    let row = location.row_index as usize;
                    if row >= entities_len || unsafe { *entities_ptr.add(row) } != entity {
                        return Err($crate::error::StateError::AccessDenied(format!(
                            "Entity {entity} has an inconsistent archetype row"
                        )));
                    }

                    // 物理写入 / 置换并精确更新所属 64 行块纪元
                    let chunk_idx = row >> 6;
                    unsafe {
                        $(
                            if !chunk_epochs[$idx].is_null() {
                                *chunk_epochs[$idx].add(chunk_idx) = next_epoch;
                            }
                        )+
                    }
                    if KEEP {
                        let old_tuple = unsafe {
                            (
                                entity,
                                $(
                                    $crate::storage::core::store::direct::write_slot_keep_for_macro(cols.$idx, row, $T),
                                )+
                            )
                        };
                        previous_values.push(old_tuple);
                    } else {
                        unsafe {
                            $(
                                $crate::storage::core::store::direct::write_slot_for_macro(cols.$idx, row, $T);
                            )+
                        }
                    }
                    written_count += 1;
                }

                // 5. 成功后推进代号并解除毒化守卫（仅在产生实际写入时推进代号）
                if written_count > 0 {
                    store.epoch = next_epoch;
                }
                poison_guard.disarm();

                if KEEP {
                    Ok(Some($crate::storage::core::store::direct::TypedReceipt { previous_values }))
                } else {
                    Ok(None)
                }
            }
        }
    };
}

// -----------------------------------------------------------------------------
// 为 (EntityId, T0) 至 (EntityId, T0, ..., T15) 实现 Trait
// -----------------------------------------------------------------------------
impl_typed_direct_batch!(0: T0);
impl_typed_direct_batch!(0: T0, 1: T1);
impl_typed_direct_batch!(0: T0, 1: T1, 2: T2);
impl_typed_direct_batch!(0: T0, 1: T1, 2: T2, 3: T3);
impl_typed_direct_batch!(0: T0, 1: T1, 2: T2, 3: T3, 4: T4);
impl_typed_direct_batch!(0: T0, 1: T1, 2: T2, 3: T3, 4: T4, 5: T5);
impl_typed_direct_batch!(0: T0, 1: T1, 2: T2, 3: T3, 4: T4, 5: T5, 6: T6);
impl_typed_direct_batch!(0: T0, 1: T1, 2: T2, 3: T3, 4: T4, 5: T5, 6: T6, 7: T7);
impl_typed_direct_batch!(0: T0, 1: T1, 2: T2, 3: T3, 4: T4, 5: T5, 6: T6, 7: T7, 8: T8);
impl_typed_direct_batch!(0: T0, 1: T1, 2: T2, 3: T3, 4: T4, 5: T5, 6: T6, 7: T7, 8: T8, 9: T9);
impl_typed_direct_batch!(0: T0, 1: T1, 2: T2, 3: T3, 4: T4, 5: T5, 6: T6, 7: T7, 8: T8, 9: T9, 10: T10);
impl_typed_direct_batch!(0: T0, 1: T1, 2: T2, 3: T3, 4: T4, 5: T5, 6: T6, 7: T7, 8: T8, 9: T9, 10: T10, 11: T11);
impl_typed_direct_batch!(0: T0, 1: T1, 2: T2, 3: T3, 4: T4, 5: T5, 6: T6, 7: T7, 8: T8, 9: T9, 10: T10, 11: T11, 12: T12);
impl_typed_direct_batch!(0: T0, 1: T1, 2: T2, 3: T3, 4: T4, 5: T5, 6: T6, 7: T7, 8: T8, 9: T9, 10: T10, 11: T11, 12: T12, 13: T13);
impl_typed_direct_batch!(0: T0, 1: T1, 2: T2, 3: T3, 4: T4, 5: T5, 6: T6, 7: T7, 8: T8, 9: T9, 10: T10, 11: T11, 12: T12, 13: T13, 14: T14);
impl_typed_direct_batch!(0: T0, 1: T1, 2: T2, 3: T3, 4: T4, 5: T5, 6: T6, 7: T7, 8: T8, 9: T9, 10: T10, 11: T11, 12: T12, 13: T13, 14: T14, 15: T15);

// -----------------------------------------------------------------------------
// Store 固有直达通道方法
// -----------------------------------------------------------------------------
impl Store {
    /// 单组件极速直达通道：将 `(EntityId, T)` 序列批量直接写入物理列，丢弃旧值。
    ///
    /// 适用场景：高频同构批量数据刷新（如传感器流、时间步推演）。
    #[inline]
    pub fn update_batch_typed_discard<T: ModelComponent + Serialize + DeserializeOwned + 'static>(
        &mut self,
        updates: impl IntoIterator<Item = (EntityId, T)>,
    ) -> Result<(), StateError> {
        self.update_batch_typed_tuple_discard(updates)
    }

    /// 多组件元组极速直达通道：支持 `(EntityId, T0)` 到 `(EntityId, T0, ..., T15)` 元组批次极速直写，丢弃旧值。
    #[inline]
    pub fn update_batch_typed_tuple_discard<Item: TypedDirectBatchUpdate>(
        &mut self,
        updates: impl IntoIterator<Item = Item>,
    ) -> Result<(), StateError> {
        Item::apply_batch_discard(self, updates.into_iter())
    }

    /// 多组件元组流式直达 Keep 通道：支持 `(EntityId, T0)` 到 `(EntityId, T0, ..., T15)` 元组流式提交并保留收据。
    #[inline]
    pub fn update_batch_typed<Item: TypedDirectBatchUpdate>(
        &mut self,
        updates: impl IntoIterator<Item = Item>,
    ) -> Result<TypedReceipt<Item>, StateError> {
        Item::apply_batch_keep(self, updates.into_iter())
    }

    /// 提交强类型批次并丢弃旧值 (Discard Mode)。
    #[inline]
    pub fn commit_typed_batch_discard<Item: TypedDirectBatchUpdate>(
        &mut self,
        batch: TypedBatch<Item>,
    ) -> Result<(), StateError> {
        self.update_batch_typed_tuple_discard(batch.entries)
    }

    /// 提交强类型批次并保留旧值收据 (Keep Mode)。
    #[inline]
    pub fn commit_typed_batch<Item: TypedDirectBatchUpdate>(
        &mut self,
        batch: TypedBatch<Item>,
    ) -> Result<TypedReceipt<Item>, StateError> {
        self.update_batch_typed(batch.entries)
    }
}
