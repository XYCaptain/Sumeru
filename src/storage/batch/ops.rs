use crate::binding::EntityId;
use crate::context::ModelComponent;
use crate::error::StateError;
use crate::id::ComponentTypeId;
use crate::storage::ErasedComponentValue;
use crate::storage::column::InPlaceOp;
use serde::de::DeserializeOwned;
use serde::Serialize;
use std::collections::BTreeMap;

/// 实体组件束 (EntityBundle)。
///
/// 用于一次性构建实体初始携带的全部组件。在构建时严格校验组件唯一性，禁止重复组件。
#[derive(Debug)]
pub struct EntityBundle {
    components: BTreeMap<ComponentTypeId, ErasedComponentValue>,
    pub(crate) pool_head: u32,
    pub(crate) pool_tail: u32,
    pub(crate) pool_len: u32,
}

impl Default for EntityBundle {
    fn default() -> Self {
        Self::new()
    }
}

impl EntityBundle {
    pub fn new() -> Self {
        Self {
            components: BTreeMap::new(),
            pool_head: u32::MAX,
            pool_tail: u32::MAX,
            pool_len: 0,
        }
    }

    pub(crate) fn from_components(
        components: BTreeMap<ComponentTypeId, ErasedComponentValue>,
    ) -> Self {
        Self {
            components,
            pool_head: u32::MAX,
            pool_tail: u32::MAX,
            pool_len: 0,
        }
    }

    /// 链式插入强类型组件。若已有同类型组件，返回 `ComponentDuplicateInPatch` 错误。
    pub fn with<T: ModelComponent + Serialize + DeserializeOwned + 'static>(
        mut self,
        value: T,
    ) -> Result<Self, StateError> {
        self.insert(value)?;
        Ok(self)
    }

    /// 插入强类型组件。若已有同类型组件，返回 `ComponentDuplicateInPatch` 错误。
    pub fn insert<T: ModelComponent + Serialize + DeserializeOwned + 'static>(
        &mut self,
        value: T,
    ) -> Result<&mut Self, StateError> {
        let erased = ErasedComponentValue::new(value);
        self.insert_erased(erased)
    }

    /// 链式插入类型擦除组件。若已有同类型组件，返回 `ComponentDuplicateInPatch` 错误。
    pub fn with_erased(mut self, value: ErasedComponentValue) -> Result<Self, StateError> {
        self.insert_erased(value)?;
        Ok(self)
    }

    /// 插入类型擦除组件。若已有同类型组件，返回 `ComponentDuplicateInPatch` 错误。
    pub fn insert_erased(&mut self, value: ErasedComponentValue) -> Result<&mut Self, StateError> {
        let type_id = value.component_type_id();
        if self.components.contains_key(&type_id) {
            return Err(StateError::ComponentDuplicateInPatch {
                entity: EntityId::from_u128(0),
                type_name: value.type_name().to_string(),
            });
        }
        self.components.insert(type_id, value);
        Ok(self)
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.components.is_empty() && self.pool_len == 0
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.components.len() + self.pool_len as usize
    }

    #[inline]
    pub fn components(&self) -> &BTreeMap<ComponentTypeId, ErasedComponentValue> {
        &self.components
    }

    #[inline]
    pub fn into_components(self) -> BTreeMap<ComponentTypeId, ErasedComponentValue> {
        self.components
    }

    #[inline]
    pub fn replace_or_insert_erased(&mut self, value: ErasedComponentValue) {
        self.components.insert(value.component_type_id(), value);
    }

    #[inline]
    pub(crate) fn set_pooled(
        &mut self,
        cid: ComponentTypeId,
        value: ErasedComponentValue,
        pool: &mut Vec<PoolEntry>,
    ) -> Result<bool, StateError> {
        if self.components.is_empty() {
            if self.pool_len > 0
                && self.pool_tail == self.pool_head.wrapping_add(self.pool_len - 1)
            {
                let start = self.pool_head as usize;
                let end = start + self.pool_len as usize;
                for entry in &mut pool[start..end] {
                    if entry.type_id == cid {
                        entry.op = PatchOp::Insert(value);
                        return Ok(false);
                    }
                }
            } else {
                let mut curr = self.pool_head;
                while curr != u32::MAX {
                    let entry = &mut pool[curr as usize];
                    if entry.type_id == cid {
                        entry.op = PatchOp::Insert(value);
                        return Ok(false);
                    }
                    curr = entry.next;
                }
            }
            CompactPatchOps::check_pool_growth(pool.len(), 1)?;
            let idx = pool.len() as u32;
            pool.push(PoolEntry {
                type_id: cid,
                op: PatchOp::Insert(value),
                next: u32::MAX,
            });
            if self.pool_len == 0 {
                self.pool_head = idx;
                self.pool_tail = idx;
            } else {
                pool[self.pool_tail as usize].next = idx;
                self.pool_tail = idx;
            }
            self.pool_len += 1;
            Ok(true)
        } else {
            let added = self.components.insert(cid, value).is_none();
            Ok(added)
        }
    }

    #[inline]
    pub(crate) fn for_each_pooled_value<'a>(
        &self,
        pool: &'a [PoolEntry],
        mut f: impl FnMut(&'a ErasedComponentValue),
    ) {
        if self.pool_len > 0 {
            if self.pool_tail == self.pool_head.wrapping_add(self.pool_len - 1) {
                let start = self.pool_head as usize;
                let end = start + self.pool_len as usize;
                for pe in &pool[start..end] {
                    if let PatchOp::Insert(v) | PatchOp::Replace(v) | PatchOp::Upsert(v) = &pe.op {
                        f(v);
                    }
                }
            } else {
                let mut curr = self.pool_head;
                while curr != u32::MAX {
                    let pe = &pool[curr as usize];
                    if let PatchOp::Insert(v) | PatchOp::Replace(v) | PatchOp::Upsert(v) = &pe.op {
                        f(v);
                    }
                    curr = pe.next;
                }
            }
        }
    }

    #[inline]
    pub(crate) fn for_each_pooled_cid(&self, pool: &[PoolEntry], mut f: impl FnMut(ComponentTypeId)) {
        if self.pool_len > 0 {
            if self.pool_tail == self.pool_head.wrapping_add(self.pool_len - 1) {
                let start = self.pool_head as usize;
                let end = start + self.pool_len as usize;
                for pe in &pool[start..end] {
                    f(pe.type_id);
                }
            } else {
                let mut curr = self.pool_head;
                while curr != u32::MAX {
                    let pe = &pool[curr as usize];
                    f(pe.type_id);
                    curr = pe.next;
                }
            }
        }
    }

    pub(crate) fn detach_pool(&mut self, pool: &mut [PoolEntry]) {
        if self.pool_len > 0 {
            let mut curr = self.pool_head;
            while curr != u32::MAX {
                let entry = &mut pool[curr as usize];
                let op = std::mem::replace(&mut entry.op, TOMBSTONE_OP);
                if let PatchOp::Insert(val) | PatchOp::Replace(val) | PatchOp::Upsert(val) = op {
                    self.components.insert(entry.type_id, val);
                }
                curr = entry.next;
            }
            self.pool_head = u32::MAX;
            self.pool_tail = u32::MAX;
            self.pool_len = 0;
        }
    }

    #[inline]
    pub(crate) fn take_sorted_components(
        mut self,
        pool: &mut [PoolEntry],
    ) -> smallvec::SmallVec<[(ComponentTypeId, ErasedComponentValue); 8]> {
        let mut out = smallvec::SmallVec::<[(ComponentTypeId, ErasedComponentValue); 8]>::new();
        if self.pool_len > 0 {
            out.reserve(self.pool_len as usize);
            let mut curr = self.pool_head;
            while curr != u32::MAX {
                let entry = &mut pool[curr as usize];
                let op = std::mem::replace(&mut entry.op, TOMBSTONE_OP);
                if let PatchOp::Insert(val) | PatchOp::Replace(val) | PatchOp::Upsert(val) = op {
                    out.push((entry.type_id, val));
                }
                curr = entry.next;
            }
            out.sort_unstable_by_key(|(id, _)| *id);
        } else if !self.components.is_empty() {
            out.reserve(self.components.len());
            for (id, val) in std::mem::take(&mut self.components) {
                out.push((id, val));
            }
        }
        out
    }
}

/// 单组件补丁细分操作 (PatchOp)
#[derive(Debug)]
pub enum PatchOp {
    /// 插入新组件：实体原先必须没有该组件，否则拒绝
    Insert(ErasedComponentValue),
    /// 替换组件：实体原先必须已有该组件，否则拒绝
    Replace(ErasedComponentValue),
    /// Upsert 组件：若已有则原地替换，若无则插入
    Upsert(ErasedComponentValue),
    /// 移除组件：实体原先必须已有该组件，否则拒绝
    Remove(ComponentTypeId),
}

impl PatchOp {
    pub fn component_type_id(&self) -> ComponentTypeId {
        match self {
            Self::Insert(val) => val.component_type_id(),
            Self::Replace(val) => val.component_type_id(),
            Self::Upsert(val) => val.component_type_id(),
            Self::Remove(id) => *id,
        }
    }

    pub fn type_name(&self) -> &str {
        match self {
            Self::Insert(val) => val.type_name(),
            Self::Replace(val) => val.type_name(),
            Self::Upsert(val) => val.type_name(),
            Self::Remove(_) => "Component",
        }
    }

    #[inline]
    pub(crate) fn into_in_place_op(self) -> InPlaceOp {
        match self {
            Self::Replace(v) => InPlaceOp::Replace(v),
            Self::Upsert(v) => InPlaceOp::Upsert(v),
            _ => unreachable!("prevalidated in-place operations"),
        }
    }

    #[inline]
    pub(crate) fn needs_drop(&self) -> bool {
        match self {
            Self::Insert(val) | Self::Replace(val) | Self::Upsert(val) => val.needs_drop(),
            Self::Remove(_) => false,
        }
    }
}

/// 用于移出 PoolEntry.op 时快速填补的零分配常量墓碑操作，避免重复计算 TypeId 哈希。
pub(crate) const TOMBSTONE_OP: PatchOp = PatchOp::Remove(ComponentTypeId::new(0));

/// 共享命令池中的单条组件更新记录。
#[derive(Debug)]
pub struct PoolEntry {
    pub type_id: ComponentTypeId,
    pub op: PatchOp,
    pub next: u32,
}

/// 单实体补丁组件操作的内部紧凑表示：
/// - `Empty`: 空补丁（0 次堆分配）
/// - `Single`: 单组件补丁（0 次堆分配）
/// - `Double`: 双组件补丁（0 次堆分配）
/// - `Multiple`: 多组件补丁（独立向量回退）
/// - `Pooled`: 共享池化多组件补丁（通过链表索引引用 CommandBuffer.pool，0 逐实体堆分配）
#[derive(Debug, Default)]
pub(crate) enum CompactPatchOps {
    #[default]
    Empty,
    Single(ComponentTypeId, PatchOp),
    Double(
        (ComponentTypeId, PatchOp),
        (ComponentTypeId, PatchOp),
    ),
    Multiple(Vec<(ComponentTypeId, PatchOp)>),
    Pooled { head: u32, tail: u32, len: usize },
}

impl CompactPatchOps {
    #[inline]
    pub fn len(&self) -> usize {
        match self {
            Self::Empty => 0,
            Self::Single(..) => 1,
            Self::Double(..) => 2,
            Self::Multiple(v) => v.len(),
            Self::Pooled { len, .. } => *len,
        }
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        match self {
            Self::Empty => true,
            Self::Pooled { len, .. } => *len == 0,
            _ => false,
        }
    }

    #[inline]
    pub fn contains_key(&self, type_id: &ComponentTypeId) -> bool {
        match self {
            Self::Empty => false,
            Self::Single(id, _) => id == type_id,
            Self::Double((id1, _), (id2, _)) => id1 == type_id || id2 == type_id,
            Self::Multiple(v) => v.iter().any(|(id, _)| id == type_id),
            Self::Pooled { .. } => panic!("contains_key on Pooled ops requires pool context, use contains_key_pooled"),
        }
    }

    #[inline]
    pub fn contains_key_pooled(&self, type_id: &ComponentTypeId, pool: &[PoolEntry]) -> bool {
        match self {
            Self::Empty => false,
            Self::Single(id, _) => id == type_id,
            Self::Double((id1, _), (id2, _)) => id1 == type_id || id2 == type_id,
            Self::Multiple(v) => v.iter().any(|(id, _)| id == type_id),
            Self::Pooled { head, len, .. } => {
                if *len == 0 {
                    return false;
                }
                let mut curr = *head;
                while curr != u32::MAX {
                    let entry = &pool[curr as usize];
                    if entry.type_id == *type_id {
                        return true;
                    }
                    curr = entry.next;
                }
                false
            }
        }
    }

    pub fn insert_op(&mut self, type_id: ComponentTypeId, op: PatchOp) {
        let old = std::mem::replace(self, Self::Empty);
        *self = match old {
            Self::Empty => Self::Single(type_id, op),
            Self::Single(id1, op1) => Self::Double((id1, op1), (type_id, op)),
            Self::Double(first, second) => {
                let mut v = Vec::with_capacity(4);
                v.push(first);
                v.push(second);
                v.push((type_id, op));
                Self::Multiple(v)
            }
            Self::Multiple(mut v) => {
                v.push((type_id, op));
                Self::Multiple(v)
            }
            Self::Pooled { .. } => panic!("insert_op on Pooled ops requires pool context, use insert_op_pooled"),
        };
    }

    // u32::MAX is a link terminator, never a valid entry index.
    pub(crate) fn check_pool_growth(current: usize, additional: usize) -> Result<(), StateError> {
        if current.checked_add(additional).is_none_or(|end| end > u32::MAX as usize) {
            return Err(StateError::CapacityExhausted("Command operation pool index exhausted".into()));
        }
        Ok(())
    }

    pub fn insert_op_pooled(&mut self, type_id: ComponentTypeId, op: PatchOp, pool: &mut Vec<PoolEntry>) -> Result<(), StateError> {
        let additional = match self {
            Self::Double(..) => 3,
            Self::Pooled { len, .. } => {
                Self::check_pool_growth(*len, 1)?;
                1
            }
            _ => 0,
        };
        if additional > 0 {
            Self::check_pool_growth(pool.len(), additional)?;
            // Reserve before changing ownership or links, including on allocation failure.
            pool.try_reserve(additional)
                .map_err(|e| StateError::CapacityExhausted(e.to_string()))?;
        }
        let old = std::mem::replace(self, Self::Empty);
        *self = match old {
            Self::Empty => Self::Single(type_id, op),
            Self::Single(id1, op1) => Self::Double((id1, op1), (type_id, op)),
            Self::Double((id1, op1), (id2, op2)) => {
                let h = pool.len() as u32;
                pool.push(PoolEntry { type_id: id1, op: op1, next: h + 1 });
                pool.push(PoolEntry { type_id: id2, op: op2, next: h + 2 });
                pool.push(PoolEntry { type_id, op, next: u32::MAX });
                Self::Pooled { head: h, tail: h + 2, len: 3 }
            }
            Self::Pooled { head, tail, len } => {
                let new_idx = pool.len() as u32;
                pool[tail as usize].next = new_idx;
                pool.push(PoolEntry { type_id, op, next: u32::MAX });
                Self::Pooled { head, tail: new_idx, len: len + 1 }
            }
            Self::Multiple(mut v) => {
                v.push((type_id, op));
                Self::Multiple(v)
            }
        };
        Ok(())
    }

    pub fn replace_op_pooled(
        &mut self,
        type_id: ComponentTypeId,
        op: PatchOp,
        pool: &mut [PoolEntry],
    ) -> bool {
        match self {
            Self::Empty => false,
            Self::Single(id, existing) => {
                if *id == type_id {
                    *existing = op;
                    true
                } else {
                    false
                }
            }
            Self::Double((id1, op1), (id2, op2)) => {
                if *id1 == type_id {
                    *op1 = op;
                    true
                } else if *id2 == type_id {
                    *op2 = op;
                    true
                } else {
                    false
                }
            }
            Self::Multiple(v) => {
                for (id, existing) in v {
                    if *id == type_id {
                        *existing = op;
                        return true;
                    }
                }
                false
            }
            Self::Pooled { head, len, .. } => {
                if *len > 0 {
                    let mut curr = *head;
                    while curr != u32::MAX {
                        let entry = &mut pool[curr as usize];
                        if entry.type_id == type_id {
                            entry.op = op;
                            return true;
                        }
                        curr = entry.next;
                    }
                }
                false
            }
        }
    }

    pub fn into_btree_map(self) -> BTreeMap<ComponentTypeId, PatchOp> {
        match self {
            Self::Empty => BTreeMap::new(),
            Self::Single(id, op) => {
                let mut map = BTreeMap::new();
                map.insert(id, op);
                map
            }
            Self::Double((id1, op1), (id2, op2)) => {
                let mut map = BTreeMap::new();
                map.insert(id1, op1);
                map.insert(id2, op2);
                map
            }
            Self::Multiple(v) => v.into_iter().collect(),
            Self::Pooled { .. } => panic!("into_btree_map on Pooled ops requires pool context"),
        }
    }


    pub fn for_each_value<'a>(&'a self, mut f: impl FnMut(&'a ErasedComponentValue)) {
        match self {
            Self::Empty => {}
            Self::Single(_, op) => match op {
                PatchOp::Insert(v) | PatchOp::Replace(v) | PatchOp::Upsert(v) => f(v),
                PatchOp::Remove(_) => {}
            },
            Self::Double((_, op1), (_, op2)) => {
                for op in [op1, op2] {
                    match op {
                        PatchOp::Insert(v) | PatchOp::Replace(v) | PatchOp::Upsert(v) => f(v),
                        PatchOp::Remove(_) => {}
                    }
                }
            }
            Self::Multiple(v) => {
                for (_, op) in v {
                    match op {
                        PatchOp::Insert(v) | PatchOp::Replace(v) | PatchOp::Upsert(v) => f(v),
                        PatchOp::Remove(_) => {}
                    }
                }
            }
            Self::Pooled { .. } => panic!("for_each_value on Pooled ops requires pool context"),
        }
    }

    pub fn for_each_op_with_pool<'a>(
        &'a self,
        pool: Option<&'a [PoolEntry]>,
        mut f: impl FnMut(ComponentTypeId, &'a PatchOp),
    ) {
        match self {
            Self::Empty => {}
            Self::Single(id, op) => f(*id, op),
            Self::Double((id1, op1), (id2, op2)) => {
                f(*id1, op1);
                f(*id2, op2);
            }
            Self::Multiple(v) => {
                for (id, op) in v {
                    f(*id, op);
                }
            }
            Self::Pooled { head, len, .. } => {
                if *len > 0 {
                    let p = pool.expect("pool must exist for Pooled ops");
                    let mut curr = *head;
                    while curr != u32::MAX {
                        let entry = &p[curr as usize];
                        f(entry.type_id, &entry.op);
                        curr = entry.next;
                    }
                }
            }
        }
    }

    pub fn for_each_op<'a>(&'a self, f: impl FnMut(ComponentTypeId, &'a PatchOp)) {
        self.for_each_op_with_pool(None, f);
    }
    pub fn into_iter_with_pool(self, pool: Option<&mut [PoolEntry]>) -> CompactOpsIter {
        match self {
            Self::Empty => CompactOpsIter::Empty,
            Self::Single(id, op) => CompactOpsIter::Single(Some((id, op))),
            Self::Double(first, second) => CompactOpsIter::Double(Some(first), Some(second)),
            Self::Multiple(v) => CompactOpsIter::Multiple(v.into_iter()),
            Self::Pooled { head, len, .. } => {
                let p = pool.expect("pool must exist for Pooled ops");
                let mut v = Vec::with_capacity(len);
                let mut curr = head;
                while curr != u32::MAX {
                    let entry = &mut p[curr as usize];
                    let op = std::mem::replace(&mut entry.op, TOMBSTONE_OP);
                    v.push((entry.type_id, op));
                    curr = entry.next;
                }
                CompactOpsIter::Multiple(v.into_iter())
            }
        }
    }
}

pub(crate) enum CompactOpsIter {
    Empty,
    Single(Option<(ComponentTypeId, PatchOp)>),
    Double(
        Option<(ComponentTypeId, PatchOp)>,
        Option<(ComponentTypeId, PatchOp)>,
    ),
    Multiple(std::vec::IntoIter<(ComponentTypeId, PatchOp)>),
}

impl Iterator for CompactOpsIter {
    type Item = (ComponentTypeId, PatchOp);

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        match self {
            Self::Empty => None,
            Self::Single(opt) => opt.take(),
            Self::Double(first, second) => first.take().or_else(|| second.take()),
            Self::Multiple(iter) => iter.next(),
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = match self {
            Self::Empty => 0,
            Self::Single(opt) => {
                if opt.is_some() {
                    1
                } else {
                    0
                }
            }
            Self::Double(first, second) => {
                (if first.is_some() { 1 } else { 0 }) + (if second.is_some() { 1 } else { 0 })
            }
            Self::Multiple(iter) => iter.len(),
        };
        (len, Some(len))
    }
}

impl ExactSizeIterator for CompactOpsIter {}

impl IntoIterator for CompactPatchOps {
    type Item = (ComponentTypeId, PatchOp);
    type IntoIter = CompactOpsIter;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        match self {
            Self::Empty => CompactOpsIter::Empty,
            Self::Single(id, op) => CompactOpsIter::Single(Some((id, op))),
            Self::Double(first, second) => CompactOpsIter::Double(Some(first), Some(second)),
            Self::Multiple(v) => CompactOpsIter::Multiple(v.into_iter()),
            Self::Pooled { .. } => panic!("into_iter on Pooled ops requires pool context, use into_iter_with_pool"),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub(crate) enum PatchSemantics {
    #[default]
    Empty,
    InPlace,
    Structural,
    Conditional,
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct NativeUpdateRef {
    lane: u32,
    // One-based row gives Option<NativeUpdateRef> an 8-byte niche layout.
    row: std::num::NonZeroU32,
}

impl NativeUpdateRef {
    #[inline]
    pub(crate) fn new(lane: usize, row: usize) -> Result<Self, StateError> {
        let lane = u32::try_from(lane)
            .map_err(|_| StateError::CapacityExhausted("Native lane index exhausted".into()))?;
        let row = u32::try_from(row)
            .ok()
            .and_then(|row| row.checked_add(1))
            .and_then(std::num::NonZeroU32::new)
            .ok_or_else(|| StateError::CapacityExhausted("Native row index exhausted".into()))?;
        Ok(Self { lane, row })
    }

    #[inline]
    pub(crate) fn lane(self) -> usize {
        self.lane as usize
    }

    #[inline]
    pub(crate) fn row(self) -> usize {
        self.row.get() as usize - 1
    }
}

/// 实体多组件原子更新补丁 (EntityPatch)。
///
/// 严格约束：同一补丁内对同一组件类型只允许定义一个操作，重复操作直接拒绝。
#[derive(Debug)]
pub struct EntityPatch {
    ops: CompactPatchOps,
    semantics: PatchSemantics,
    // CommandBuffer-only association. Native values remain in typed lanes;
    // this header does not own or wrap them as PatchOp values.
    pub(crate) native_head: Option<NativeUpdateRef>,
}

impl Default for EntityPatch {
    fn default() -> Self {
        Self::new()
    }
}

impl EntityPatch {
    pub fn new() -> Self {
        Self {
            ops: CompactPatchOps::Empty,
            semantics: PatchSemantics::Empty,
            native_head: None,
        }
    }

    /// 从暂存槽的 BTreeMap 确定性构造原子替换补丁。
    /// 不变量：槽位映射保证组件类型 ID 严格唯一；
    /// 若为空映射则保持 Empty 语义，非空则设为 InPlace 替换。
    pub(crate) fn from_staged_map(
        staged: std::collections::BTreeMap<ComponentTypeId, ErasedComponentValue>,
    ) -> Self {
        if staged.is_empty() {
            return Self::new();
        }
        let mut patch = Self::new();
        for (cid, erased) in staged {
            patch.ops.insert_op(cid, PatchOp::Replace(erased));
        }
        patch.semantics = PatchSemantics::InPlace;
        patch
    }

    #[inline]
    pub(crate) fn semantics(&self) -> PatchSemantics {
        self.semantics
    }

    pub(crate) fn ops(&self) -> &CompactPatchOps {
        &self.ops
    }

    #[allow(dead_code)]
    pub(crate) fn ops_mut(&mut self) -> &mut CompactPatchOps {
        &mut self.ops
    }

    pub(crate) fn mutate_op_pooled(
        &mut self,
        type_id: ComponentTypeId,
        type_name: &str,
        op: PatchOp,
        sem_hint: PatchSemantics,
        pool: &mut Vec<PoolEntry>,
    ) -> Result<(), StateError> {
        if self.ops.contains_key_pooled(&type_id, pool) {
            return Err(StateError::ComponentDuplicateInPatch {
                entity: EntityId::from_u128(0),
                type_name: type_name.to_string(),
            });
        }
        self.ops.insert_op_pooled(type_id, op, pool)?;
        match sem_hint {
            PatchSemantics::InPlace => {
                if self.semantics == PatchSemantics::Empty {
                    self.semantics = PatchSemantics::InPlace;
                }
            }
            PatchSemantics::Structural => {
                self.semantics = PatchSemantics::Structural;
            }
            PatchSemantics::Conditional => {
                if self.semantics == PatchSemantics::Empty || self.semantics == PatchSemantics::InPlace {
                    self.semantics = PatchSemantics::Conditional;
                }
            }
            _ => {}
        }
        Ok(())
    }

    pub(crate) fn set_op_pooled(
        &mut self,
        type_id: ComponentTypeId,
        op: PatchOp,
        sem_hint: PatchSemantics,
        pool: &mut Vec<PoolEntry>,
    ) -> Result<(), StateError> {
        if !self.ops.replace_op_pooled(type_id, op, pool) {
            unreachable!("checked by contains_key_pooled");
        }
        match sem_hint {
            PatchSemantics::InPlace => {
                if self.semantics == PatchSemantics::Empty {
                    self.semantics = PatchSemantics::InPlace;
                }
            }
            PatchSemantics::Structural => {
                self.semantics = PatchSemantics::Structural;
            }
            PatchSemantics::Conditional => {
                if self.semantics == PatchSemantics::Empty
                    || self.semantics == PatchSemantics::InPlace
                {
                    self.semantics = PatchSemantics::Conditional;
                }
            }
            _ => {}
        }
        Ok(())
    }

    pub(crate) fn detach_pool(&mut self, pool: &mut [PoolEntry]) {
        if matches!(self.ops, CompactPatchOps::Pooled { .. }) {
            let ops = std::mem::replace(&mut self.ops, CompactPatchOps::Empty);
            self.ops = CompactPatchOps::Multiple(ops.into_iter_with_pool(Some(pool)).collect());
        }
    }

    fn check_duplicate(&self, type_id: ComponentTypeId, type_name: &str) -> Result<(), StateError> {
        if self.ops.contains_key(&type_id) {
            Err(StateError::ComponentDuplicateInPatch {
                entity: EntityId::from_u128(0),
                type_name: type_name.to_string(),
            })
        } else {
            Ok(())
        }
    }

    /// 插入新组件（必须不存在）
    pub fn insert<T: ModelComponent + Serialize + DeserializeOwned + 'static>(
        &mut self,
        value: T,
    ) -> Result<&mut Self, StateError> {
        self.insert_erased(ErasedComponentValue::new(value))
    }

    pub fn insert_erased(&mut self, value: ErasedComponentValue) -> Result<&mut Self, StateError> {
        let type_id = value.component_type_id();
        self.check_duplicate(type_id, value.type_name())?;
        self.ops.insert_op(type_id, PatchOp::Insert(value));
        self.semantics = PatchSemantics::Structural;
        Ok(self)
    }

    /// 替换已有组件（必须已存在）
    pub fn replace<T: ModelComponent + Serialize + DeserializeOwned + 'static>(
        &mut self,
        value: T,
    ) -> Result<&mut Self, StateError> {
        self.replace_erased(ErasedComponentValue::new(value))
    }

    pub fn replace_erased(&mut self, value: ErasedComponentValue) -> Result<&mut Self, StateError> {
        let type_id = value.component_type_id();
        self.check_duplicate(type_id, value.type_name())?;
        self.ops.insert_op(type_id, PatchOp::Replace(value));
        if self.semantics == PatchSemantics::Empty {
            self.semantics = PatchSemantics::InPlace;
        }
        Ok(self)
    }

    /// Upsert 组件（存在则替换，不存在则新增）
    pub fn upsert<T: ModelComponent + Serialize + DeserializeOwned + 'static>(
        &mut self,
        value: T,
    ) -> Result<&mut Self, StateError> {
        self.upsert_erased(ErasedComponentValue::new(value))
    }

    pub fn upsert_erased(&mut self, value: ErasedComponentValue) -> Result<&mut Self, StateError> {
        let type_id = value.component_type_id();
        self.check_duplicate(type_id, value.type_name())?;
        self.ops.insert_op(type_id, PatchOp::Upsert(value));
        if self.semantics == PatchSemantics::Empty || self.semantics == PatchSemantics::InPlace {
            self.semantics = PatchSemantics::Conditional;
        }
        Ok(self)
    }

    pub fn replace_or_upsert_erased(&mut self, value: ErasedComponentValue) {
        let type_id = value.component_type_id();
        match &mut self.ops {
            CompactPatchOps::Empty => {
                self.ops = CompactPatchOps::Single(type_id, PatchOp::Upsert(value));
            }
            CompactPatchOps::Single(id, op) => {
                if *id == type_id {
                    *op = match op {
                        PatchOp::Insert(_) => PatchOp::Insert(value),
                        PatchOp::Replace(_) => PatchOp::Replace(value),
                        PatchOp::Upsert(_) => PatchOp::Upsert(value),
                        PatchOp::Remove(_) => PatchOp::Upsert(value),
                    };
                } else {
                    let (old_id, old_op) = (*id, std::mem::replace(op, TOMBSTONE_OP));
                    self.ops = CompactPatchOps::Double((old_id, old_op), (type_id, PatchOp::Upsert(value)));
                }
            }
            CompactPatchOps::Double((id1, op1), (id2, op2)) => {
                if *id1 == type_id {
                    *op1 = PatchOp::Upsert(value);
                } else if *id2 == type_id {
                    *op2 = PatchOp::Upsert(value);
                } else {
                    let mut v = Vec::with_capacity(3);
                    v.push((*id1, std::mem::replace(op1, TOMBSTONE_OP)));
                    v.push((*id2, std::mem::replace(op2, TOMBSTONE_OP)));
                    v.push((type_id, PatchOp::Upsert(value)));
                    self.ops = CompactPatchOps::Multiple(v);
                }
            }
            CompactPatchOps::Multiple(v) => {
                if let Some(entry) = v.iter_mut().find(|(id, _)| *id == type_id) {
                    entry.1 = PatchOp::Upsert(value);
                } else {
                    v.push((type_id, PatchOp::Upsert(value)));
                }
            }
            CompactPatchOps::Pooled { .. } => {
                panic!("replace_or_upsert_erased on Pooled ops requires pool context");
            }
        }
        if self.semantics == PatchSemantics::Empty || self.semantics == PatchSemantics::InPlace {
            self.semantics = PatchSemantics::Conditional;
        }
    }

    /// 移除已有组件（必须已存在）
    pub fn remove<T: ModelComponent + 'static>(&mut self) -> Result<&mut Self, StateError> {
        let type_name = std::any::type_name::<T>();
        let type_id = ComponentTypeId::from_name(type_name);
        self.remove_by_id(type_id, type_name)
    }

    pub fn remove_by_id(
        &mut self,
        type_id: ComponentTypeId,
        type_name: &str,
    ) -> Result<&mut Self, StateError> {
        self.check_duplicate(type_id, type_name)?;
        self.ops.insert_op(type_id, PatchOp::Remove(type_id));
        self.semantics = PatchSemantics::Structural;
        Ok(self)
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.ops.is_empty()
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.ops.len()
    }

    #[inline]
    pub fn into_ops(self) -> BTreeMap<ComponentTypeId, PatchOp> {
        debug_assert!(self.native_head.is_none(), "native lanes must be consumed by their owner");
        self.ops.into_btree_map()
    }

    #[inline]
    pub(crate) fn into_compact_ops(self) -> CompactPatchOps {
        debug_assert!(self.native_head.is_none(), "detach native references before consuming ordinary operations");
        self.ops
    }

    #[inline]
    pub(crate) fn for_each_value<'a>(&'a self, f: impl FnMut(&'a ErasedComponentValue)) {
        self.ops.for_each_value(f);
    }

    #[inline]
    pub fn for_each_op_with_pool<'a>(
        &'a self,
        pool: Option<&'a [PoolEntry]>,
        f: impl FnMut(ComponentTypeId, &'a PatchOp),
    ) {
        self.ops.for_each_op_with_pool(pool, f);
    }

    #[inline]
    pub fn for_each_op<'a>(&'a self, f: impl FnMut(ComponentTypeId, &'a PatchOp)) {
        self.ops.for_each_op(f);
    }
}

/// 批量操作容器中的单实体指令 (BatchOp)
#[derive(Debug)]
pub enum BatchOp {
    CreateEntity { id: EntityId, bundle: EntityBundle },
    PatchEntity { id: EntityId, patch: EntityPatch },
    DeleteEntity { id: EntityId },
}

impl BatchOp {
    #[inline]
    pub fn entity(&self) -> EntityId {
        match self {
            Self::CreateEntity { id, .. } => *id,
            Self::PatchEntity { id, .. } => *id,
            Self::DeleteEntity { id } => *id,
        }
    }

    #[inline]
    pub(crate) fn for_each_value<'a>(&'a self, mut f: impl FnMut(&'a ErasedComponentValue)) {
        match self {
            Self::CreateEntity { bundle, .. } => {
                for v in bundle.components().values() {
                    f(v);
                }
            }
            Self::PatchEntity { patch, .. } => {
                patch.for_each_value(f);
            }
            Self::DeleteEntity { .. } => {}
        }
    }
}
