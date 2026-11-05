use smallvec::SmallVec;
use crate::error::StateError;
use crate::id::ComponentTypeId;
use crate::storage::archetype::Archetype;
use crate::storage::column::AnyColumn;
use crate::storage::registry::ComponentRegistry;

/// Receipt pool indices increase strictly in preparation order: every nonempty
/// in-place plan takes the next appended updated-entity index. Structural and
/// standalone-native updates can introduce gaps, but cannot reorder the plans;
/// scratch reuse preserves that order. This is about the receipt's ID directory,
/// not physical entity rows (which may move independently).
///
/// For a strictly increasing sequence of n integers, last - first == n - 1 iff
/// every step is one. Reuse that invariant instead of rescanning the segment or
/// adding ordinary-only tracking locals to the generic preparation frame.
#[inline]
pub(super) fn receipt_pool_range_is_contiguous<const NATIVE: bool>(
    plans: &[crate::storage::core::column::PlannedInPlacePatch],
) -> bool {
    let Some(first) = plans.first() else { return true; };
    // Keep native preparation's original linear check in its own specialization;
    // ordinary-only endpoint arithmetic must not add live temporaries to it.
    if NATIVE {
        return (0..plans.len()).all(|offset| {
            plans[offset].pool_index == first.pool_index + offset as u32
        });
    }
    debug_assert!(
        plans.windows(2).all(|pair| pair[0].pool_index < pair[1].pool_index),
        "prepared receipt pool indices must increase strictly",
    );
    let last = plans.last().unwrap();
    u32::try_from(plans.len() - 1).is_ok_and(|distance| {
        last.pool_index.checked_sub(first.pool_index) == Some(distance)
    })
}



/// 原型列访问与校验缓存记录
///
/// # 容量规划与分配契约
/// - **内联容量设为 16 个组件列**（`SmallVec<[...; 16]>`）：绝大多数业务实体的组件数量在 1~8 之间，
///   16 项内联保证典型工作负载下列缓存完全在栈上分配（0 次堆内存分配）。
/// - **堆溢出降级**：当单个原型的组件数超过 16 时，SmallVec 会自动透明平滑降级为堆分配，保证功能完备性。
pub(crate) struct ArchetypeRecord<'a> {
    pub(crate) arch_id: u32,
    pub(crate) columns: SmallVec<[(ComponentTypeId, &'a dyn AnyColumn, bool); 16]>,
}

/// 跨实体原型列缓存追踪器 (ColumnCacheTracker)
///
/// # 设计目标与容量契约
/// - **多原型栈内联容量**：内联 4 个不同的原型记录（`SmallVec<[ArchetypeRecord; 4]>`）。
///   在同一批次涉及 1~4 个原型的场景（绝大多数微批次与分块更新），全部在栈上就地缓存；
///   当单批次跨越 5 个及以上原型时，SmallVec 溢出至堆分配，功能不受影响。
/// - **活跃原型 O(1) 短路指针**：通过 `active_idx` 记录当前实体命中的原型记录索引。
///   连续实体处于相同原型时，原型定位开销为 O(1) 单指针比对（无需遍历 records）。
///
/// # 校验等价性证明与不变量 (Validation Equivalence Contract)
/// 本追踪器通过 `ColumnValidation::Verified` 状态实现了同原型组件“首次严检，后续复用”的性能提升，
/// 其与“对每个实体的每个组件值逐一做完整注册表与物理列契约校验”具有完全的数学与类型等价性：
/// 1. **前置值类型全覆盖核验**：在批次准备前置阶段，`entries.for_each_value` 已经对该批次中输入的所有
///    组件值实例完成了具体的 Rust 类型与 Payload 格式合法性校验；
/// 2. **拓扑与行有效性单射约束**：`validate_archetype_layout` 已核验列行数与实体数严格对齐；且通过
///    `src_arch.entities().get(row_index) == Some(&id)` 确认了实体与行号的双向单射映射；
/// 3. **单列物理单例一致性**：在同一个 Archetype 内部，同一 `ComponentTypeId` 对应的物理列实例
///    `&dyn AnyColumn` 在整个批次执行生命周期内地址、类型与元数据不变。
/// 因此：只要该列在首次被该原型访问时通过了 `verify_column_contract`（核验物理类型、列布局及注册表定义），
/// 后续属于该原型的所有实体的同组件写入，物理安全和类型安全严格等价，无需重复核验。
pub(crate) struct ColumnCacheTracker<'a> {
    enabled: bool,
    records: SmallVec<[ArchetypeRecord<'a>; 4]>,
    active_idx: Option<usize>,
}

impl<'a> ColumnCacheTracker<'a> {
    #[inline]
    pub(crate) fn new(enabled: bool) -> Self {
        Self {
            enabled,
            records: SmallVec::new(),
            active_idx: None,
        }
    }

    /// 仅断开活跃原型指针，保留已验证的记录。
    /// 用于 Create/Delete 等改变当前实体上下文但不修改原型列结构的场景。
    /// prepare 阶段是纯规划，不执行结构变更，列引用在整个 prepare 期间始终有效。
    #[inline]
    pub(crate) fn deactivate(&mut self) {
        self.active_idx = None;
    }

    #[inline]
    pub(crate) fn visit_archetype(&mut self, arch_id: u32) {
        if !self.enabled {
            return;
        }
        if let Some(idx) = self.active_idx {
            if self.records[idx].arch_id == arch_id {
                return;
            }
        }
        
        // 查找是否已经存在记录
        for (i, rec) in self.records.iter().enumerate() {
            if rec.arch_id == arch_id {
                self.active_idx = Some(i);
                return;
            }
        }
        
        // 未找到，新增一条记录
        self.records.push(ArchetypeRecord {
            arch_id,
            columns: SmallVec::new(),
        });
        self.active_idx = Some(self.records.len() - 1);
    }

    #[inline]
    pub(crate) fn get_or_insert_column(
        &mut self,
        src_arch: &'a Archetype,
        type_id: ComponentTypeId,
    ) -> Option<&'a dyn AnyColumn> {
        self.visit_column(src_arch, type_id).map(|(column, _)| column)
    }

    /// Reuse the same table/column directory for physical preflight.
    pub(crate) fn visit_column(
        &mut self,
        src_arch: &'a Archetype,
        type_id: ComponentTypeId,
    ) -> Option<(&'a dyn AnyColumn, bool)> {
        let Some(idx) = self.active_idx else {
            let col = src_arch.columns().get(&type_id)?.as_ref();
            return Some((col, true));
        };
        
        let record = &mut self.records[idx];
        for (cached_type_id, col, _) in &record.columns {
            if *cached_type_id == type_id {
                return Some((*col, false));
            }
        }
        
        if let Some(col) = src_arch.columns().get(&type_id) {
            let col_ref = col.as_ref();
            record.columns.push((type_id, col_ref, false));
            Some((col_ref, true))
        } else {
            None
        }
    }

    /// 复用前置扫描已验证的类型信息，对同原型同组件仅做一次列契约验证。
    ///
    /// # 不变量依赖
    /// - 前置 `entries.for_each_value` 已对该 ComponentTypeId 的所有值完成类型合法性核验
    /// - `validate_archetype_layout` 已确认列行数与实体行数一致
    /// - `src_arch.entities().get(row_index) == Some(&id)` 已确认行有效
    /// - 因此行界限检查和值类型检查均已覆盖，仅需首次验证列物理实例的类型一致性
    #[inline]
    pub(crate) fn validate_or_lookup(
        &mut self,
        src_arch: &'a Archetype,
        type_id: ComponentTypeId,
        registry: &ComponentRegistry,
    ) -> Result<ColumnValidation<'a>, StateError> {
        let Some(idx) = self.active_idx else {
            // 缓存未启用或缺少活跃原型上下文时，回退到直查并验证列
            if let Some(col) = src_arch.columns().get(&type_id) {
                let col_ref = col.as_ref();
                if Self::verify_column_contract(col_ref, type_id, registry)? {
                    return Ok(ColumnValidation::Verified);
                } else {
                    return Ok(ColumnValidation::Fallback(col_ref));
                }
            }
            return Ok(ColumnValidation::NotFound);
        };

        // 查缓存
        let record = &mut self.records[idx];
        for (cached_type_id, cached_col_ref, validated) in &record.columns {
            if *cached_type_id == type_id {
                if *validated {
                    // 列类型已验证，直接返回通过
                    return Ok(ColumnValidation::Verified);
                }
                // 列存在但未完成验证（如单实体批次没命中 registry）
                return Ok(ColumnValidation::Fallback(*cached_col_ref));
            }
        }

        // 缓存未命中或未验证：查原型并登记
        if let Some(col) = src_arch.columns().get(&type_id) {
            let col_ref = col.as_ref();
            if Self::verify_column_contract(col_ref, type_id, registry)? {
                record.columns.push((type_id, col_ref, true));
                Ok(ColumnValidation::Verified)
            } else {
                record.columns.push((type_id, col_ref, false));
                Ok(ColumnValidation::Fallback(col_ref))
            }
        } else {
            Ok(ColumnValidation::NotFound)
        }
    }

    /// 验证列物理实例的类型与注册表定义一致。
    /// 返回 Ok(true) 表示校验通过，Ok(false) 表示注册表无此定义需要 fallback。
    #[inline]
    fn verify_column_contract(
        col: &dyn AnyColumn,
        type_id: ComponentTypeId,
        registry: &ComponentRegistry,
    ) -> Result<bool, StateError> {
        let Some((expected_rust_id, expected_name)) = registry.get_type_info(&type_id) else {
            // 注册表未覆盖此类型（例如未注册的动态组件）
            // 安全回退：调用方将回退到 col.validate_erased
            return Ok(false);
        };
        // 验证列物理实例的 type_id 一致
        if col.type_id() != type_id {
            return Err(StateError::AccessDenied(format!(
                "Column type_id {:?} does not match expected {:?}",
                col.type_id(), type_id,
            )));
        }
        // 验证列物理实例的 rust_type_id 与前置扫描一致（如果前置扫描已确定）
        if let Some(expected) = expected_rust_id {
            if let Some(col_rust_id) = col.rust_type_id() {
                if col_rust_id != expected {
                    return Err(StateError::AccessDenied(format!(
                        "Column rust TypeId {:?} does not match verified {:?} for {}",
                        col_rust_id, expected, expected_name,
                    )));
                }
            }
        }
        // 验证列物理实例的 type_name 与前置扫描一致
        if col.type_name() != expected_name {
            return Err(StateError::AccessDenied(format!(
                "Column type_name {} does not match verified {}",
                col.type_name(), expected_name,
            )));
        }
        Ok(true)
    }
}

/// 列契约校验状态与分发结果
pub(crate) enum ColumnValidation<'a> {
    /// 物理列实例已在当前原型中完成契约核验，同原型后续实体直接复用校验结果，0 重复开销
    Verified,
    /// 目标组件在当前原型中不存在
    NotFound,
    /// 列存在但属于未在注册表中统一定义的特殊列，需回退至动态逐项核验路径
    Fallback(&'a dyn AnyColumn),
}
