//! 统一多组件原位联合借用与修改目标 (ModifyTarget)
//!
//! 支持单组件引用 `&mut T` 与 1..16 元组 `(A, B, ...)`，
//! 通过单次原型检索、单次行对齐校验、单次代际递增，在单次事件闭包内提供安全且零别名的联合修改能力。

use crate::storage::archetype::Archetype;
use crate::ModelComponent;
use crate::id::EntityId;
use crate::{ComponentTypeId, StateError};
use serde::{de::DeserializeOwned, Serialize};

pub(crate) mod sealed {
    use super::*;

    #[allow(private_interfaces, private_bounds)]
    pub trait ModifyFetch<'a> {
        type View;
        fn fetch_mut(
            arch: &'a mut Archetype,
            row: usize,
            entity: EntityId,
        ) -> Result<Self::View, StateError>;
    }
}

/// 原位修改目标 trait，由 `&mut T` 与多元组 `(A, B, ...)` 自动实现。
#[allow(private_interfaces, private_bounds)]
pub trait ModifyTarget<'a>: sealed::ModifyFetch<'a> {
    /// 校验所有组件类型互不相同（防别名安全保证）
    fn validate_disjoint() -> Result<(), StateError>;
}

// -----------------------------------------------------------------------------
// 1. 单组件引用 `&'a mut T`
// -----------------------------------------------------------------------------
#[allow(private_interfaces, private_bounds)]
impl<'a, T: ModelComponent + Serialize + DeserializeOwned + 'static> sealed::ModifyFetch<'a> for &'a mut T {
    type View = &'a mut T;

    #[inline]
    fn fetch_mut(
        arch: &'a mut Archetype,
        row: usize,
        entity: EntityId,
    ) -> Result<Self::View, StateError> {
        let cid = ComponentTypeId::of::<T>();
        let col = arch.get_typed_column_mut::<T>(cid, entity)?;
        col.get_mut(row).ok_or_else(|| {
            StateError::AccessDenied(format!("Entity {entity} row {row} out of bounds in archetype"))
        })
    }
}

impl<'a, T: ModelComponent + Serialize + DeserializeOwned + 'static> ModifyTarget<'a> for &'a mut T {
    #[inline]
    fn validate_disjoint() -> Result<(), StateError> {
        Ok(())
    }
}

macro_rules! check_disjoint_pairs {
    ($head_id:ident, $head_type:ty, $($tail_id:ident),*) => {
        $(
            if $head_id == $tail_id {
                return Err(StateError::AccessDenied(format!(
                    "Aliasing violation: component '{}' requested multiple times in modify",
                    std::any::type_name::<$head_type>()
                )));
            }
        )*
    };
}

macro_rules! check_all_disjoint {
    ($head_id:ident, $head_type:ty) => {
        let _ = $head_id;
    };
    ($head_id:ident, $head_type:ty, $($tail_id:ident, $tail_type:ty),+) => {
        check_disjoint_pairs!($head_id, $head_type, $($tail_id),+);
        check_all_disjoint!($($tail_id, $tail_type),+);
    };
}

// -----------------------------------------------------------------------------
// 2. 宏实现 2..16 元组联合借用 (A, B), (A, B, C), ...
// -----------------------------------------------------------------------------
macro_rules! impl_modify_target_tuple {
    ($($T:ident),+) => {
        #[allow(private_interfaces, private_bounds)]
        impl<'a, $($T: ModelComponent + Serialize + DeserializeOwned + 'static),+> sealed::ModifyFetch<'a> for ($($T,)+) {
            type View = ($(&'a mut $T,)+);

            #[inline]
            #[allow(non_snake_case)]
            fn fetch_mut(
                arch: &'a mut Archetype,
                row: usize,
                entity: EntityId,
            ) -> Result<Self::View, StateError> {
                <Self as ModifyTarget<'a>>::validate_disjoint()?;
                $(
                    let $T: *mut crate::storage::column::TypedColumn<$T> =
                        arch.get_typed_column_mut_ptr::<$T>(ComponentTypeId::of::<$T>(), entity)?;
                )+
                // # Safety Invariants
                // 1. 列类型反别名：`validate_disjoint` 确保元组内任意两个分量的 `ComponentTypeId` 绝对互斥；
                // 2. 列指针非空有效：`get_typed_column_mut_ptr` 确保原型包含该列，且类型下转型完全匹配强类型容器；
                // 3. 每列行有效性独立检查：不仅实体行号有效，各组件列内部独立执行 `(*$T).get_mut(row)` 校验行边界；
                // 4. 存储稳定性与无冲突：持有 `&mut Store` 独占借用，闭包执行期间底层列向量不扩容、不收缩、不重分配；
                // 5. 生命周期不逃逸：HRTB 生命周期约束引用作用域完全绑定在修改闭包内；
                // 6. 零大小类型 (ZST) 安全保证：对于 ZST，解引用裸指针不访问物理内存，满足 Rust 引用非空、对齐、生命周期有效性与排他性约束。
                $(
                    let $T = unsafe {
                        (*$T).get_mut(row).ok_or_else(|| {
                            StateError::AccessDenied(format!(
                                "Entity {entity} row {row} out of bounds in archetype"
                            ))
                        })?
                    };
                )+
                Ok(($($T,)+))
            }
        }

        impl<'a, $($T: ModelComponent + Serialize + DeserializeOwned + 'static),+> ModifyTarget<'a> for ($($T,)+) {
            #[inline]
            #[allow(non_snake_case)]
            fn validate_disjoint() -> Result<(), StateError> {
                $(
                    let $T = ComponentTypeId::of::<$T>();
                )+
                check_all_disjoint!($($T, $T),+);
                Ok(())
            }
        }
    };
}

impl_modify_target_tuple!(A);
impl_modify_target_tuple!(A, B);
impl_modify_target_tuple!(A, B, C);
impl_modify_target_tuple!(A, B, C, D);
impl_modify_target_tuple!(A, B, C, D, E);
impl_modify_target_tuple!(A, B, C, D, E, F);
impl_modify_target_tuple!(A, B, C, D, E, F, G);
impl_modify_target_tuple!(A, B, C, D, E, F, G, H);
impl_modify_target_tuple!(A, B, C, D, E, F, G, H, I);
impl_modify_target_tuple!(A, B, C, D, E, F, G, H, I, J);
impl_modify_target_tuple!(A, B, C, D, E, F, G, H, I, J, K);
impl_modify_target_tuple!(A, B, C, D, E, F, G, H, I, J, K, L);
impl_modify_target_tuple!(A, B, C, D, E, F, G, H, I, J, K, L, M);
impl_modify_target_tuple!(A, B, C, D, E, F, G, H, I, J, K, L, M, N);
impl_modify_target_tuple!(A, B, C, D, E, F, G, H, I, J, K, L, M, N, O);
impl_modify_target_tuple!(A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P);

