//! 领域无关只读 Store View 与模型执行上下文。
//!
//! 纯底层状态存储上下文，不包含上层仿真推进或业务调度状态。

/// 模型状态组件标记 trait (ModelComponent)。
///
/// 任何强类型状态组件均实现该 trait，供底层存储与执行上下文安全访问。
///
/// This is a type/transport bound, NOT proof of snapshot or undo semantics.
/// Managed state must be mutated through Store write access. Interior mutation
/// through shared handles/atomics bypasses leases and write-head tracking; such
/// external resources are not covered by baseline, Capture or Receipt guarantees.
/// Clone must preserve the intended historical value: a shallow shared mutable
/// handle is not an independent snapshot. No runtime check enforces this policy.
pub trait ModelComponent: Clone + Send + Sync + 'static {}
impl<T: Clone + Send + Sync + 'static> ModelComponent for T {}

/// 纯只读数据读取上下文 (StoreReadContext, ADR-05)。
pub struct StoreReadContext<'a, T: ModelComponent> {
    data: &'a T,
}

impl<'a, T: ModelComponent> Clone for StoreReadContext<'a, T> {
    #[inline]
    fn clone(&self) -> Self {
        *self
    }
}

impl<'a, T: ModelComponent> Copy for StoreReadContext<'a, T> {}

impl<'a, T: ModelComponent> StoreReadContext<'a, T> {
    #[inline]
    pub fn new(data: &'a T) -> Self {
        Self { data }
    }

    #[inline]
    pub fn get(&self) -> &'a T {
        self.data
    }

    #[inline]
    pub fn state(&self) -> &'a T {
        self.data
    }
}

/// 底层状态存储受限暂存可写上下文 (StoreWriteContext, ADR-05)。
pub struct StoreWriteContext<'a, State: ModelComponent> {
    initial_state: &'a State,
    staged_state: Option<State>,
}

impl<'a, State: ModelComponent> StoreWriteContext<'a, State> {
    #[inline]
    pub fn new(initial_state: &'a State) -> Self {
        Self {
            initial_state,
            staged_state: None,
        }
    }

    #[inline]
    pub fn current(&self) -> &State {
        self.staged_state.as_ref().unwrap_or(self.initial_state)
    }

    #[inline]
    pub fn state(&self) -> &State {
        self.current()
    }

    #[inline]
    pub fn set(&mut self, new_state: State) {
        self.staged_state = Some(new_state);
    }

    #[inline]
    pub fn update<F: FnOnce(&mut State)>(&mut self, f: F) {
        let mut state = self
            .staged_state
            .take()
            .unwrap_or_else(|| self.initial_state.clone());
        f(&mut state);
        self.staged_state = Some(state);
    }

    #[inline]
    pub fn take_staged(self) -> Option<State> {
        self.staged_state
    }

    #[inline]
    pub fn into_mutation(self) -> Option<State> {
        self.staged_state
    }

    #[inline]
    pub fn has_mutations(&self) -> bool {
        self.staged_state.is_some()
    }
}

