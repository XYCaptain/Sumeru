//! 类型擦除的通用列存储接口体系 (AnyColumn 及核心正交领域特质)

use super::archive::{ArchivedColumnData, ColumnCodecId, RawColumnArchive};
use crate::error::StateError;
use crate::id::{ComponentTypeId, EntityId, SchemaVersion};
use crate::schema::ComponentSchema;
use smallvec::SmallVec;
use std::any::{Any, TypeId};
use std::fmt;
use std::marker::PhantomData;
use std::ops::Range;
use std::sync::Arc;

/// 列级擦除值原生行操作虚表 (ColumnOpsVTable)
///
/// 纯函数指针表，允许直面物理列内存执行纳秒级原地操作。
#[derive(Debug, Clone, Copy)]
pub(crate) struct ColumnOpsVTable {
    pub(crate) assign_raw: unsafe fn(raw_ptr: *mut u8, row: usize, value: ErasedComponentValue),
    pub(crate) replace_raw: unsafe fn(
        raw_ptr: *mut u8,
        row: usize,
        value: ErasedComponentValue,
    ) -> ErasedComponentValue,
    pub(crate) new_typed_owned_block: fn(
        component: crate::id::ComponentTypeId,
        capacity: usize,
    ) -> Arc<dyn ErasedBlock>,
}

pub(crate) mod sealed {
    /// 仅限包内构造的内部受控能力证明 Token。
    /// 由于字段 pub(crate) 且构造方法不对外暴露，任何外部 crate 均无法伪造或构造该 Token。
    #[derive(Debug, Clone, Copy)]
    pub struct InternalControlledToken(pub(crate) ());

    impl InternalControlledToken {
        #[inline]
        pub(crate) const fn new() -> Self {
            Self(())
        }
    }
}

/// Opaque internal capability used by controlled ErasedBlock implementations.
/// The sealed module is crate-visible; external blocks cannot manufacture it.
pub struct BorrowedArchiveRows<'a> {
    pub(super) archive: RawColumnArchive,
    pub(super) slices: SmallVec<[super::archive::RawArchiveSlice; 4]>,
    pub(super) _borrow: PhantomData<&'a dyn ErasedBlock>,
}

/// 只读类型擦除物理块契约 (Send + Sync)
#[allow(private_interfaces)]
pub trait ErasedBlock: std::any::Any + Send + Sync {
    /// Internal borrowed-encoding capability. Unsupported external blocks are
    /// rejected before encoding instead of silently cloning their values.
    #[doc(hidden)]
    fn borrow_archive_rows<'a>(
        &'a self,
        _rows: &mut BorrowedArchiveRows<'a>,
        _range: Range<usize>,
    ) -> Result<(), StateError> {
        Err(StateError::AccessDenied(
            "Block has no borrowed archive capability".into(),
        ))
    }
    fn len(&self) -> usize;
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
    fn as_any(&self) -> &dyn std::any::Any;
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any;
    fn into_any_box(self: Box<Self>) -> Box<dyn std::any::Any>;
    fn into_any_arc(self: Arc<Self>) -> Arc<dyn std::any::Any + Send + Sync>;
    fn estimated_bytes(&self) -> usize {
        0
    }
    /// Consume owned payloads. Generic external blocks may clone; native blocks
    /// override this to move values, including non-Copy values, exactly once.
    fn into_values(
        self: Box<Self>,
        component: ComponentTypeId,
        range: Range<usize>,
    ) -> Result<Vec<ErasedComponentValue>, StateError> {
        if range.start > range.end || range.end > self.len() {
            return Err(StateError::BatchConflict(
                "Invalid owned block range".into(),
            ));
        }
        range.map(|row| self.clone_value(row, component)).collect()
    }
    /// Unsupported blocks fail during preparation, before the Store is touched.
    fn clone_value(
        &self,
        _row: usize,
        _component: ComponentTypeId,
    ) -> Result<ErasedComponentValue, StateError> {
        Err(StateError::AccessDenied(
            "Block does not support inverse materialization".into(),
        ))
    }
    fn take_value(
        &mut self,
        row: usize,
        component: ComponentTypeId,
    ) -> Result<ErasedComponentValue, StateError> {
        self.clone_value(row, component)
    }
    /// 包内受控移动能力证明。私有 Token 保证外部 crate 绝对无法实现此方法以伪造受控能力！
    #[doc(hidden)]
    fn internal_controlled_token(&self) -> Option<sealed::InternalControlledToken> {
        None
    }
    /// Return the Rust TypeId and type name of the component at row if available without cloning.
    fn type_info(
        &self,
        _row: usize,
        _component: ComponentTypeId,
    ) -> Option<(std::any::TypeId, &'static str)> {
        None
    }
    /// 擦除块是否可以直接作为物理列原生数据源 (具有 ColumnBlock<T> 或 Vec<T> 的直接移动能力)
    fn is_native_column_block(&self) -> bool {
        false
    }

    /// Internal capability for retaining a borrowed receipt while handing a
    /// typed copy to the existing consuming undo. External/erased blocks keep
    /// using clone_value; only controlled native blocks opt in.
    #[doc(hidden)]
    fn clone_native_range(&self, _range: Range<usize>) -> Option<Box<dyn ErasedBlock>> {
        None
    }

    #[doc(hidden)]
    fn value_any_ref(&self, _row: usize) -> Option<&(dyn std::any::Any + Send + Sync)> {
        None
    }

    #[doc(hidden)]
    fn row_size_bytes(&self, _row: usize) -> usize {
        0
    }

    #[doc(hidden)]
    fn clone_erased_arc(&self) -> Result<Arc<dyn ErasedBlock>, StateError> {
        Err(StateError::AccessDenied(
            "Block does not support arc cloning".into(),
        ))
    }

    #[doc(hidden)]
    fn push_from_erased(&mut self, _value: &mut ErasedComponentValue) -> Result<(), StateError> {
        Err(StateError::AccessDenied(
            "Block does not support push_from_erased".into(),
        ))
    }

    #[doc(hidden)]
    fn extend_from_erased_pairs(
        &mut self,
        values: &mut [(EntityId, ErasedComponentValue)],
    ) -> Result<(), StateError> {
        for (_, val) in values.iter_mut() {
            self.push_from_erased(val)?;
        }
        Ok(())
    }

    #[doc(hidden)]
    fn extend_from_erased_indexed(
        &mut self,
        values: &mut [(EntityId, ErasedComponentValue)],
        indices: &[usize],
    ) -> Result<(), StateError> {
        for &idx in indices {
            self.push_from_erased(&mut values[idx].1)?;
        }
        Ok(())
    }

    #[doc(hidden)]
    fn drain_into_erased(
        &mut self,
        _range: Range<usize>,
        _component: ComponentTypeId,
    ) -> Result<Vec<ErasedComponentValue>, StateError> {
        Err(StateError::AccessDenied(
            "Block does not support drain_into_erased".into(),
        ))
    }

    #[doc(hidden)]
    fn reserve_exact_total(&mut self, _total_capacity: usize) {}

    #[doc(hidden)]
    fn append_from_other(&mut self, _other: &mut dyn ErasedBlock) -> Result<usize, StateError> {
        Err(StateError::AccessDenied(
            "Block does not support append_from_other".into(),
        ))
    }

    #[doc(hidden)]
    #[allow(private_interfaces)]
    unsafe fn swap_prepared_column_rows(
        &mut self,
        _descriptor: &ColumnDescriptor,
        _rows: &[PreparedRow],
    ) {
        unreachable!("swap_prepared_column_rows requires TypedOwnedBlock")
    }

    #[doc(hidden)]
    #[allow(private_interfaces)]
    unsafe fn swap_strided_column_rows(
        &mut self,
        _descriptor: &ColumnDescriptor,
        _first_row: usize,
        _count: usize,
    ) {
        unreachable!("swap_strided_column_rows requires TypedOwnedBlock")
    }

    #[doc(hidden)]
    #[allow(private_interfaces)]
    unsafe fn write_strided_column_rows(
        &self,
        _descriptor: &ColumnDescriptor,
        _first_row: usize,
        _count: usize,
    ) {
        unreachable!("write_strided_column_rows requires TypedOwnedBlock")
    }
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct PreparedRow {
    pub(crate) slot: usize,
    pub(crate) row: usize,
    pub(crate) tracking: usize,
}

/// Keeps allocations alive, but grants no access permission.
#[derive(Clone)]
pub(crate) struct AllocationPin(pub(crate) Arc<dyn std::any::Any + Send + Sync>);
impl std::fmt::Debug for AllocationPin {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_tuple("AllocationPin")
            .field(&Arc::strong_count(&self.0))
            .finish()
    }
}

/// 轻量列内存与类型描述符 (ColumnDescriptor)
///
/// 在独占访问边界预先解析；按连续物理偏移寻址。
/// 借出切片后不再可变重借用整列，也不重新分离或移动后备块。
#[derive(Debug, Clone)]
pub(crate) struct ColumnDescriptor {
    pub raw_ptr: *mut u8,
    pub len: usize,
    pub rust_type_id: Option<std::any::TypeId>,
    #[allow(dead_code)]
    pub type_name: &'static str,
    pub(crate) ops: ColumnOpsVTable,
    pub(crate) archive: Option<super::archive::RawColumnArchive>,
    pub(crate) writable: bool,
    pub(crate) _allocation_pin: Option<AllocationPin>,
    pub(crate) physical_version: u64,
}

unsafe impl Send for ColumnDescriptor {}
unsafe impl Sync for ColumnDescriptor {}

/// 原位组件操作类型（保留原始 Replace 或 Upsert 语义）
#[derive(Debug)]
pub enum InPlaceOp {
    Replace(ErasedComponentValue),
    Upsert(ErasedComponentValue),
}

impl InPlaceOp {
    #[inline]
    pub(crate) fn into_value(self) -> ErasedComponentValue {
        match self {
            Self::Replace(v) | Self::Upsert(v) => v,
        }
    }

    #[inline]
    #[allow(dead_code)]
    pub(crate) fn value_ref(&self) -> &ErasedComponentValue {
        match self {
            Self::Replace(v) | Self::Upsert(v) => v,
        }
    }

    #[inline]
    #[allow(dead_code)]
    pub(crate) fn value_mut(&mut self) -> &mut ErasedComponentValue {
        match self {
            Self::Replace(v) | Self::Upsert(v) => v,
        }
    }
}

/// 原地组件替换集：单组件内联，多组件引用批次共享数组的独占区间。
#[derive(Debug)]
pub(crate) enum InPlaceReplacements {
    Empty,
    Single(ComponentTypeId, InPlaceOp),
    Multiple(std::ops::Range<usize>),
    // Value range and prepared receipt-column range. Equal component layouts
    // share the second range; neither index refers to mutable Store addresses.
    // Single-row segments leave it empty: there is no repeated lookup to amortize.
    Takeover(std::ops::Range<usize>, std::ops::Range<usize>),
    /// Values remain owned by typed lanes; this range contains only references.
    NativeMultiple(std::ops::Range<usize>),
}

impl InPlaceReplacements {
    #[inline]
    pub(crate) fn is_empty(&self) -> bool {
        match self {
            Self::Empty => true,
            Self::Single(..) => false,
            Self::Multiple(v) | Self::Takeover(v, _) | Self::NativeMultiple(v) => v.is_empty(),
        }
    }
}

/// 内部已校验的原地补丁计划项（纯列值写入，绝不迁移 Archetype 或移动实体）
#[derive(Debug)]
pub(crate) struct PlannedInPlacePatch {
    pub entity: crate::binding::EntityId,
    pub location: crate::storage::allocator::EntityLocation,
    pub replacements: InPlaceReplacements,
    pub pool_index: u32,
}

/// 列物理形态与元数据：负责列类型标识、动态反射与底层物理内存寻址
#[allow(dead_code)]
pub(crate) trait ColumnLayout {
    /// 组件类型逻辑 ID (FNV-1a 哈希)
    fn type_id(&self) -> ComponentTypeId;

    /// 组件类型静态名称
    fn type_name(&self) -> &'static str;

    /// Rust 原生 TypeId；未定型/字节暂存列 (RawColumn) 返回 None
    fn rust_type_id(&self) -> Option<std::any::TypeId>;

    /// 获取动态 Any 借用（用于安全向下转型 downcast）
    fn as_any(&self) -> &dyn Any;

    /// 获取动态 Any 可变借用（用于安全向下转型 downcast）
    fn as_any_mut(&mut self) -> &mut dyn Any;

    /// 获取底层原生内存指针与长度。
    ///
    /// # Safety
    /// 必须在独占/互斥保护下调用，调用方必须保证指针访问不违反别名与生命周期约束。
    unsafe fn raw_parts_mut(&mut self) -> (*mut u8, usize);

    /// 获取底层只读物理内存指针与行数（无 &mut self 要求，消除反向构造整列可变借用的风险）。
    ///
    /// # Safety
    /// 必须在布局固定/只读保护下调用，调用方必须保证指针访问不违反别名与生命周期约束。
    unsafe fn raw_parts_read(&self) -> (*const u8, usize);

    /// Called only at an exclusive layout boundary. Default backends own their
    /// storage; typed Arc-backed columns override this without detaching data.
    fn access_parts(&mut self) -> (*mut u8, usize, bool) {
        let (ptr, len) = unsafe { self.raw_parts_mut() };
        (ptr, len, true)
    }

    /// 获取列级原生行操作纯函数指针虚表
    fn ops_vtable(&self) -> ColumnOpsVTable;

    /// Row-only cloning under a compatible lease; never borrow the whole column.
    fn shared_clone_raw(
        &self,
    ) -> Option<unsafe fn(*const u8, usize, ComponentTypeId) -> ErasedComponentValue> {
        None
    }
    fn shared_push_clone_raw(
        &self,
    ) -> Option<unsafe fn(*const u8, usize, &mut dyn AnyColumn) -> Result<(), StateError>> {
        None
    }
    fn shared_push_clone_range_raw(
        &self,
    ) -> Option<unsafe fn(*const u8, usize, usize, &mut dyn AnyColumn) -> Result<(), StateError>>
    {
        None
    }
    fn shared_element_size(&self) -> usize {
        0
    }
}

/// 行结构与物理槽位管理：负责单列维度的行数统计、容量管理及紧凑行删除
#[allow(dead_code)]
pub(crate) trait ColumnRows {
    /// 获取当前列行数
    fn len(&self) -> usize;

    /// 判断列是否为空
    #[inline]
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// 预留行物理容量
    fn reserve(&mut self, additional: usize);

    /// 清空列内所有行数据（保留底层连续分配容量）
    fn clear_rows(&mut self);

    /// ECS O(1) Swap-Remove 紧凑行删除（不关心内部值类型，仅执行槽位物理置换）
    fn swap_remove(&mut self, row: usize);
}

/// 组件值安全操作：负责组件值的安全借用、擦除值的类型受检读写、覆盖写入、原位置换与批量 Patch
#[allow(dead_code)]
pub(crate) trait ColumnValueOps {
    /// 获取指定行组件值的只读类型擦除借用（用于 EntityRef 检查与诊断）
    fn value_ref(&self, row: usize) -> Option<&(dyn Any + Send + Sync)>;

    /// 读出指定行的克隆擦除值
    fn clone_erased(&self, row: usize) -> Result<ErasedComponentValue, StateError>;

    // ---- 擦除值修改协议 ----

    /// 校验目标行索引与值的类型兼容性
    fn validate_erased(&self, row: usize, value: &ErasedComponentValue) -> Result<(), StateError>;

    /// 校验追加写入值的类型兼容性
    fn validate_erased_for_push(&self, value: &ErasedComponentValue) -> Result<(), StateError>;

    /// 免检追加新值到末尾
    fn push_erased_infallible(&mut self, value: ErasedComponentValue);

    /// 免检原位置换：写入新值并将旧值换出
    fn replace_erased_infallible(
        &mut self,
        row: usize,
        value: ErasedComponentValue,
    ) -> ErasedComponentValue;

    /// 免检覆盖写入：直接覆盖旧值并析构（无需收据的极速路径，默认提供回退实现）
    fn assign_erased_infallible(&mut self, row: usize, value: ErasedComponentValue) {
        drop(self.replace_erased_infallible(row, value));
    }

    /// 免检原地交换：将 slot 中的新值写入行，并将原行的旧值回填至 slot（零分配收据接管通道）
    fn swap_erased_infallible(&mut self, row: usize, slot: &mut ErasedComponentValue);

    /// 移除行并将其打包为擦除值返回
    fn swap_remove_erased(&mut self, row: usize) -> ErasedComponentValue;

    // ---- 批量原位更新性能通道 ----

    /// 单次虚表分发批量提交切片，直接丢弃旧值（默认循环实现，具体列可覆盖优化）
    fn discard_in_place_single_chunk(&mut self, chunk: &mut [PlannedInPlacePatch]) {
        for plan in chunk {
            let InPlaceReplacements::Single(_, op) =
                std::mem::replace(&mut plan.replacements, InPlaceReplacements::Empty)
            else {
                unreachable!("validated single component chunk")
            };
            self.assign_erased_infallible(plan.location.row_index as usize, op.into_value());
        }
    }

    /// 专用提交期独占原位置换：直接写入已准备好的独占块，绝对不克隆
    ///
    /// # Safety
    /// 目标行必须已预准备独占块，且类型匹配。
    unsafe fn replace_exclusive_unchecked(
        &mut self,
        row: usize,
        value: ErasedComponentValue,
    ) -> ErasedComponentValue {
        self.replace_erased_infallible(row, value)
    }

    /// 专用提交期独占覆盖写入：直接写入已准备好的独占块，绝对不克隆
    ///
    /// # Safety
    /// 目标行必须已预准备独占块，且类型匹配。
    unsafe fn assign_exclusive_unchecked(&mut self, row: usize, value: ErasedComponentValue) {
        unsafe {
            drop(self.replace_exclusive_unchecked(row, value));
        }
    }
}

/// 列快照持久化：负责整列连续快照载荷的归档、校验、还原及行级增量 WAL 编解码
#[allow(dead_code)]
pub(crate) trait ColumnSnapshot {
    /// Copy the optional schema-bound function table, not live column data.
    fn raw_archive_capability(&self) -> Option<super::archive::RawColumnArchive> {
        None
    }

    // ---- 通道 A：整列连续快照持久化 (rkyv 高性能零拷贝通道) ----

    /// 整列归档：将整列数据编码为连续快照载荷
    fn archive_column(&self) -> Result<ArchivedColumnData, StateError>;

    /// Archive privately owned erased values using this registered schema.
    /// Does not borrow Current, clone T, or materialize a temporary typed column.
    fn archive_erased_values(
        &self,
        _values: &[&ErasedComponentValue],
    ) -> Result<ArchivedColumnData, StateError> {
        Err(StateError::AccessDenied(
            "Column has no registered archive capability".into(),
        ))
    }

    /// 归档校验：验证归档数据的类型 ID、模式版本、行数契约与字节合法性
    fn validate_archived_column(&self, archived: &ArchivedColumnData) -> Result<(), StateError>;

    /// 整列恢复：在校验通过后，直接整列恢复数据
    fn restore_archived_column(&mut self, archived: &ArchivedColumnData) -> Result<(), StateError>;

    /// 获取列绑定的快照归档编解码标识
    fn codec(&self) -> ColumnCodecId;

    /// 获取列绑定的模式版本
    fn schema_version(&self) -> u32;

    /// 检查该列是否已绑定可用快照归档能力 (如整列 rkyv 归档)
    fn has_archive_capability(&self) -> bool {
        false
    }

    // ---- 通道 B：行级增量 WAL 编解码 (bincode 点位日志通道) ----

    /// 序列化指定行为独立字节载荷（用于增量操作日志记录）
    fn serialize_row(&self, row: usize) -> Result<Vec<u8>, StateError>;

    /// 反序列化字节载荷并追加为新行
    fn push_deserialized(&mut self, bytes: &[u8]) -> Result<(), StateError>;

    /// 反序列化字节载荷并原位覆盖指定行
    fn set_deserialized(&mut self, row: usize, bytes: &[u8]) -> Result<(), StateError>;

    /// 预校验反序列化字节载荷在目标行的有效性
    fn validate_deserialized(&self, row: usize, bytes: &[u8]) -> Result<(), StateError>;

    /// 将字节载荷解码为擦除组件值（仅在显式 checkpoint/WAL 恢复边界调用）
    fn decode_erased(&self, bytes: &[u8]) -> Result<ErasedComponentValue, StateError>;
}

/// 测试专用实验性操作特征 (单独特征保留，仅在 cfg(test) 下提供实验性原地收据方法)
#[cfg(test)]
pub(crate) trait ColumnExperimentOps {
    fn experiment_swap_retained(&mut self, _row: usize, _value: &mut ErasedComponentValue) {
        panic!("receipt experiment requires a typed column")
    }

    fn experiment_assign_retained(&mut self, _row: usize, _value: &ErasedComponentValue) {
        panic!("receipt experiment requires a typed column")
    }
}

/// 非测试环境下为空特征，保证零运行时与虚表开销
#[cfg(not(test))]
pub(crate) trait ColumnExperimentOps {}

/// 类型擦除通用列接口 (AnyColumn)
///
/// 组合 4 个正交领域特质及测试实验特征，承载整列生命周期与跨原型列行搬迁协议。
#[allow(dead_code)]
pub(crate) trait AnyColumn:
    ColumnLayout
    + ColumnRows
    + ColumnValueOps
    + ColumnSnapshot
    + ColumnExperimentOps
    + Send
    + Sync
    + 'static
{
    // ---- 整列生命周期与原型工厂 ----

    /// 深拷贝：克隆整列及其包含的所有数据行
    fn clone_box(&self) -> Box<dyn AnyColumn>;

    /// 仅克隆列的模式原型与编解码虚表，不克隆任何数据行
    fn empty_clone(&self) -> Box<dyn AnyColumn>;

    // ---- 跨列行搬迁协作协议 (Row Migration Protocol) ----

    /// 阶段 1：预检查源行与目标列类型的相容性
    fn validate_copy_to(&self, src_row: usize, dst: &dyn AnyColumn) -> Result<(), StateError>;

    /// 阶段 2（受检）：验证并拷贝源行到目标列末尾（统一默认实现，消除冗余）
    fn copy_row_to(&self, src_row: usize, dst: &mut dyn AnyColumn) -> Result<(), StateError> {
        self.validate_copy_to(src_row, dst)?;
        self.copy_row_to_infallible(src_row, dst);
        Ok(())
    }

    /// 阶段 2（免检克隆）：免检将源行值深克隆追加到目标列末尾
    fn copy_row_to_infallible(&self, src_row: usize, dst: &mut dyn AnyColumn);

    /// 阶段 2（所有权移动）：免检将源行以 swap_remove 方式移出并追加到目标列
    fn move_row_to_infallible(&mut self, src_row: usize, dst: &mut dyn AnyColumn);

    /// 返回针对 `Direct(Owned)` 独占连续列的单态化行搬移函数指针（供 `FlatColumnEntry` 缓存内联使用）
    fn raw_move_row_direct_fn(&self) -> Option<unsafe fn(*mut (), usize, *mut ())> {
        None
    }

    /// 返回针对 `Direct(Owned)` 独占连续列的单态化 swap_remove 函数指针（供 `FlatColumnEntry` 缓存内联使用）
    fn raw_swap_remove_direct_fn(&self) -> Option<unsafe fn(*mut (), usize)> {
        None
    }

    /// 返回针对 `Direct(Owned)` 独占连续列的单态化 push_erased 函数指针（供 `FlatColumnEntry` 缓存内联使用）
    fn raw_push_erased_direct_fn(&self) -> Option<unsafe fn(*mut (), &mut ErasedComponentValue)> {
        None
    }

    // ---- 跨列能力升级协议 ----

    /// 若当前列未绑定可用归档能力，而 source 已绑定，则升级归档虚表能力
    fn upgrade_archive_capability_from(&mut self, _source: &dyn AnyColumn) -> bool {
        false
    }

    /// 获取当前列物理版本号
    fn physical_version(&self) -> u64 {
        0
    }

    /// 检查当前列是否处于 16KB 硬件页对齐模式
    fn is_page_aligned(&self) -> bool {
        false
    }

    /// 将当前列提升为 16KB 硬件页对齐模式（若支持）
    fn promote_to_page_aligned(&mut self) {}

    /// 导出 16KB 硬件页对齐的 Metal 零拷贝指针与页对齐字节长度
    fn metal_raw_parts(&self) -> Option<(*mut u8, usize)> {
        None
    }

    /// 检查该列当前是否处于分段态（开源连续列恒为 false）
    fn is_segmented(&self) -> bool {
        false
    }

    /// 获取当前列的分段数量
    fn segment_count(&self) -> usize {
        if self.is_empty() {
            0
        } else {
            1
        }
    }

    /// 协同物化：将列转换为物理连续的独占列
    fn ensure_contiguous(&mut self) -> Result<(), StateError> {
        Ok(())
    }

    /// Prepare a detached contiguous replacement without publishing it.
    fn prepare_contiguous(&mut self) -> Result<Option<Box<dyn AnyColumn>>, StateError> {
        if !self.is_segmented() {
            return Ok(None);
        }
        let mut column = self.clone_box();
        column.ensure_contiguous()?;
        Ok(Some(column))
    }

    fn supports_freeze(&self) -> bool {
        false
    }

    /// Build a column sharing a captured immutable backing.
    fn column_from_frozen(
        &self,
        _block: std::sync::Arc<dyn ErasedBlock>,
    ) -> Result<Box<dyn AnyColumn>, StateError> {
        Err(StateError::AccessDenied(
            "frozen restoration unsupported".into(),
        ))
    }

    /// Build a writable private column by moving `Vec<T>` out of an exclusively owned `TypedOwnedBlock<T>`.
    fn column_from_typed_owned(
        &self,
        _block: std::sync::Arc<dyn ErasedBlock>,
    ) -> Result<Box<dyn AnyColumn>, StateError> {
        Err(StateError::AccessDenied(
            "typed owned restoration unsupported".into(),
        ))
    }

    /// 冻结当前列为不可变共享物理块并返回擦除句柄
    fn freeze_column(&mut self) -> Result<std::sync::Arc<dyn ErasedBlock>, StateError> {
        Err(StateError::AccessDenied(
            "freeze_column not supported on this column".into(),
        ))
    }

    /// 从擦除物理块中恢复单行旧值
    fn restore_row_from_erased_block(
        &mut self,
        _target_row: usize,
        _block: &dyn ErasedBlock,
        _source_idx: usize,
    ) -> Result<(), StateError> {
        Err(StateError::AccessDenied(
            "restore_row_from_erased_block not supported on this column".into(),
        ))
    }

    /// 从擦除物理块中恢复连续切片旧值
    fn restore_range_from_erased_block(
        &mut self,
        _target_start: usize,
        _len: usize,
        _block: &dyn ErasedBlock,
        _source_start: usize,
    ) -> Result<(), StateError> {
        Err(StateError::AccessDenied(
            "restore_range_from_erased_block not supported on this column".into(),
        ))
    }

    /// 消费原生擦除块直接按离散目标行号写入，跳过中间 Vec<ErasedComponentValue>
    fn apply_native_block_scatter(
        &mut self,
        _block: Box<dyn ErasedBlock>,
        _block_range: std::ops::Range<usize>,
        _target_rows: &[usize],
    ) -> Result<(), StateError> {
        Err(StateError::AccessDenied(
            "apply_native_block_scatter not supported on this column".into(),
        ))
    }

    /// 消费原生擦除块直接作为连续段换入目标列
    fn apply_native_block_range(
        &mut self,
        _block: Box<dyn ErasedBlock>,
        _block_range: std::ops::Range<usize>,
        _target_row_start: usize,
    ) -> Result<(), StateError> {
        Err(StateError::AccessDenied(
            "apply_native_block_range not supported on this column".into(),
        ))
    }

    /// 消费/共享原生擦除块直接作为连续段换入目标列
    fn apply_native_shared_block_range(
        &mut self,
        _block: std::sync::Arc<dyn ErasedBlock>,
        _block_range: std::ops::Range<usize>,
        _target_row_start: usize,
    ) -> Result<(), StateError> {
        Err(StateError::AccessDenied(
            "apply_native_shared_block_range not supported on this column".into(),
        ))
    }

    /// 分配具有预留容量的增量追加列，返回 `(列实例, 底层基指针, 实际物理容量)`。
    fn new_append_column(&self, _capacity: usize) -> Option<(Box<dyn AnyColumn>, *mut u8, usize)> {
        None
    }

    /// 从独占源列移出 `src_row` 行元素并写入预分配追加块基指针的 `dst_slot` 槽位。
    ///
    /// # Safety
    /// - `dst_base` 必须指向同类型 `T` 的有效追加块内存，且 `dst_slot` 处于未初始化且未越界状态。
    /// - `src` 必须为同类型独占列且 `src_row < src.len()`。
    unsafe fn move_into_append_ptr(
        &self,
        _dst_base: *mut u8,
        _dst_slot: usize,
        _src: &mut dyn AnyColumn,
        _src_row: usize,
    ) {
        unreachable!("move_into_append_ptr only called on append-capable TypedColumn")
    }

    /// 回收追加块尾部的 `slot_start..slot_start + count` 槽位。
    ///
    /// # Safety
    /// - `base` 必须指向同类型 `T` 的追加块基地址，且 `slot_start..slot_start + count` 已初始化且无外部借用。
    unsafe fn reclaim_append_tail_ptr(
        &self,
        _base: *mut u8,
        _slot_start: usize,
        _count: usize,
    ) -> Option<Box<dyn AnyColumn>> {
        unreachable!("reclaim_append_tail_ptr only called on append-capable TypedColumn")
    }

    /// 在追加块销毁或转回独占 Core 列前，将底层 `Vec<T>` 的长度同步为已初始化槽位数。
    ///
    /// # Safety
    /// - `0..initialized_len` 槽位必须均已初始化。
    unsafe fn sync_append_len(&mut self, _initialized_len: usize) {}

    /// 按给定的行重排索引 `order` 构建重排后的新列。
    fn reordered_column(
        &self,
        _order: &[usize],
        _group_spans: Option<&[std::ops::Range<usize>]>,
    ) -> Result<Box<dyn AnyColumn>, StateError> {
        Err(StateError::AccessDenied(
            "reordered_column not supported on this column".into(),
        ))
    }
}

pub(crate) trait ComponentValue: Send + Sync {
    fn as_any(&self) -> &dyn Any;
    fn as_any_mut(&mut self) -> &mut dyn Any;
    fn clone_box(&self) -> Box<dyn ComponentValue>;
    fn encode(&self) -> Result<Vec<u8>, StateError>;
    fn empty_column(&self) -> Box<dyn AnyColumn>;
    fn into_any(self: Box<Self>) -> Box<dyn Any>;
    fn schema(&self, version: SchemaVersion) -> ComponentSchema;
}

pub(crate) const INLINE_SIZE: usize = 32;
pub(crate) const INLINE_ALIGN: usize = 16;

#[repr(C, align(16))]
pub(crate) struct InlineBuffer {
    pub(crate) bytes: [std::mem::MaybeUninit<u8>; INLINE_SIZE],
}

impl InlineBuffer {
    #[inline]
    pub(crate) const fn uninit() -> Self {
        Self {
            bytes: [std::mem::MaybeUninit::uninit(); INLINE_SIZE],
        }
    }
}

pub(crate) struct InlineVTable {
    pub(crate) rust_type_id: TypeId,
    pub(crate) size: usize,
    pub(crate) align: usize,
    pub(crate) drop: Option<unsafe fn(*mut u8)>,
    pub(crate) clone: unsafe fn(*const u8, *mut u8),
    pub(crate) encode: unsafe fn(*const u8) -> Result<Vec<u8>, StateError>,
    pub(crate) empty_column: fn() -> Box<dyn AnyColumn>,
    pub(crate) schema: fn(SchemaVersion) -> ComponentSchema,
}

pub(crate) trait HasInlineVTable: 'static {
    const VTABLE: InlineVTable;
}

pub(crate) enum ValueRepr {
    Inline {
        buffer: InlineBuffer,
        vtable: &'static InlineVTable,
    },
    Heap(Box<dyn ComponentValue>),
    Placeholder {
        rust_type_id: TypeId,
    },
}

impl Clone for ValueRepr {
    fn clone(&self) -> Self {
        match self {
            ValueRepr::Inline { buffer, vtable } => {
                let mut new_buffer = InlineBuffer::uninit();
                unsafe {
                    (vtable.clone)(
                        buffer.bytes.as_ptr() as *const u8,
                        new_buffer.bytes.as_mut_ptr() as *mut u8,
                    );
                }
                ValueRepr::Inline {
                    buffer: new_buffer,
                    vtable,
                }
            }
            ValueRepr::Heap(boxed) => ValueRepr::Heap(boxed.clone_box()),
            ValueRepr::Placeholder { rust_type_id } => ValueRepr::Placeholder {
                rust_type_id: *rust_type_id,
            },
        }
    }
}

/// An owned component with both its schema identity and concrete Rust type.
pub struct ErasedComponentValue {
    pub(super) component_type_id: ComponentTypeId,
    pub(super) type_name: &'static str,
    pub(crate) repr: ValueRepr,
}

impl ErasedComponentValue {
    #[inline(always)]
    pub(crate) fn needs_drop(&self) -> bool {
        match &self.repr {
            ValueRepr::Inline { vtable, .. } => vtable.drop.is_some(),
            ValueRepr::Heap(_) => true,
            ValueRepr::Placeholder { .. } => false,
        }
    }
}

impl Drop for ErasedComponentValue {
    #[inline]
    fn drop(&mut self) {
        if let ValueRepr::Inline { buffer, vtable } = &mut self.repr {
            if let Some(drop_fn) = vtable.drop {
                unsafe {
                    drop_fn(buffer.bytes.as_mut_ptr() as *mut u8);
                }
            }
        }
    }
}

impl Clone for ErasedComponentValue {
    fn clone(&self) -> Self {
        Self {
            component_type_id: self.component_type_id,
            type_name: self.type_name,
            repr: self.repr.clone(),
        }
    }
}

impl fmt::Debug for ErasedComponentValue {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ErasedComponentValue")
            .field("component_type_id", &self.component_type_id)
            .field("type_name", &self.type_name)
            .finish_non_exhaustive()
    }
}
