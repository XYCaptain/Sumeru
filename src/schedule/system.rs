//! 调度系统定义与自动类型推导原语 (`System` & `IntoSystem`)
//!
//! # 理论支撑与文献证明
//!
//! 1. **零注解权限推导 (Compile-Time Access Deduction)**:
//!    基于 Sumeru 的 `QueryMutShape`，系统所需查询的元组类型 `Q`（如 `(&mut Position, &Velocity)`）
//!    在编译期即可通过 `Q::access_descriptors()` 自动展开为精确的只读/可变组件类型集合，
//!    完全无需用户手动书写任何宏或在运行时声明依赖。
//!
//! 2. **三级缓存与拓扑增量协同 (Flecs Level-2 Index Caching)**:
//!    每个系统内联持有独立的 `Query<Q, F>` 句柄。当 `Store` 的原型集合未发生结构变化
//!    （`archetype_set_generation` 保持一致）时，系统直接复用已预绑定的物理列槽位索引，
//!    执行期开销降至绝对最低。

use std::marker::PhantomData;
use crate::storage::core::store::Store;
use crate::storage::query::handle::{Query, QueryBuilder};
use crate::storage::query::QueryMutShape;
use crate::storage::query::spec::{All, QueryFilter};
use crate::schedule::descriptor::AccessDescriptor;

/// 调度系统核心接口
pub trait System: Send + 'static {
    /// 系统的诊断名称
    fn name(&self) -> &'static str;

    /// 系统的组件访问权限描述符
    fn access(&self) -> &AccessDescriptor;

    /// 系统初始化与元数据预绑定（在调度开始前执行）
    fn initialize(&mut self, store: &Store);

    /// 执行系统主体逻辑
    fn run(&mut self, store: &mut Store);
}

/// 支持转换为 [`System`] 的类型特征
pub trait IntoSystem<Marker>: Sized {
    type System: System;

    fn into_system(self) -> Self::System;
}

// ---------------------------------------------------------------------------
// 1. 原生 `&mut Store` 闭包/函数系统（默认独占全局访问）
// ---------------------------------------------------------------------------

pub struct ExclusiveFunctionSystem<F> {
    name: &'static str,
    f: F,
    access: AccessDescriptor,
}

impl<F> ExclusiveFunctionSystem<F>
where
    F: FnMut(&mut Store) + Send + 'static,
{
    pub fn new(name: &'static str, f: F) -> Self {
        Self {
            name,
            f,
            access: AccessDescriptor::new_exclusive(),
        }
    }

    /// 覆盖默认的独占权限为显式指定的权限（用于用户明确知道其仅做局部修改的场景）
    pub fn with_access(mut self, access: AccessDescriptor) -> Self {
        self.access = access;
        self
    }
}

impl<F> System for ExclusiveFunctionSystem<F>
where
    F: FnMut(&mut Store) + Send + 'static,
{
    #[inline(always)]
    fn name(&self) -> &'static str {
        self.name
    }

    #[inline(always)]
    fn access(&self) -> &AccessDescriptor {
        &self.access
    }

    fn initialize(&mut self, _store: &Store) {}

    #[inline(always)]
    fn run(&mut self, store: &mut Store) {
        (self.f)(store);
    }
}

// ---------------------------------------------------------------------------
// 2. 基于单 `Query<Q, F>` 的强类型自动推导系统
// ---------------------------------------------------------------------------

pub struct QuerySystem<Q, F, Func>
where
    Q: QueryMutShape,
    F: QueryFilter + Clone,
{
    name: &'static str,
    filter: F,
    query: Option<Query<Q, F>>,
    func: Func,
    access: AccessDescriptor,
}

impl<Q, F, Func> QuerySystem<Q, F, Func>
where
    Q: QueryMutShape,
    F: QueryFilter + Clone,
    Func: FnMut(&mut Query<Q, F>, &mut Store) + Send + 'static,
{
    pub fn new(name: &'static str, filter: F, func: Func) -> Self {
        let mut access = AccessDescriptor::new();
        for (cid, _, is_write) in Q::access_descriptors() {
            if is_write {
                access.writes.insert(cid);
            } else {
                access.reads.insert(cid);
            }
        }

        Self {
            name,
            filter,
            query: None,
            func,
            access,
        }
    }
}

impl<Q, F, Func> System for QuerySystem<Q, F, Func>
where
    Q: QueryMutShape,
    F: QueryFilter + Clone,
    Func: FnMut(&mut Query<Q, F>, &mut Store) + Send + 'static,
{
    #[inline(always)]
    fn name(&self) -> &'static str {
        self.name
    }

    #[inline(always)]
    fn access(&self) -> &AccessDescriptor {
        &self.access
    }

    fn initialize(&mut self, store: &Store) {
        if let Some(query) = &mut self.query {
            let _ = query.ensure_cache_current(store);
        } else {
            self.query = QueryBuilder::<Q, F>::with_filter(self.filter.clone())
                .build(store)
                .ok();
        }
    }

    #[inline(always)]
    fn run(&mut self, store: &mut Store) {
        if self.query.is_none() {
            self.initialize(store);
        } else if let Some(query) = &mut self.query {
            let _ = query.ensure_cache_current(store);
        }
        if let Some(query) = &mut self.query {
            (self.func)(query, store);
        }
    }
}

// ---------------------------------------------------------------------------
// 3. IntoSystem 自动实现
// ---------------------------------------------------------------------------

pub struct ExclusiveMarker;
pub struct QueryMarker<Q, F>(PhantomData<(Q, F)>);
pub struct SystemMarker;

impl<S: System> IntoSystem<SystemMarker> for S {
    type System = S;

    #[inline(always)]
    fn into_system(self) -> Self::System {
        self
    }
}

impl<Func> IntoSystem<ExclusiveMarker> for Func
where
    Func: FnMut(&mut Store) + Send + 'static,
{
    type System = ExclusiveFunctionSystem<Func>;

    fn into_system(self) -> Self::System {
        ExclusiveFunctionSystem::new(std::any::type_name::<Func>(), self)
    }
}

/// 辅助宏：快速创建具名独占系统
#[inline]
pub fn exclusive_system(
    name: &'static str,
    f: impl FnMut(&mut Store) + Send + 'static,
) -> ExclusiveFunctionSystem<impl FnMut(&mut Store) + Send + 'static> {
    ExclusiveFunctionSystem::new(name, f)
}

/// 创建带自动类型推导的单 Query 系统
pub fn query_system<Q, F, Func>(
    name: &'static str,
    filter: F,
    func: Func,
) -> QuerySystem<Q, F, Func>
where
    Q: QueryMutShape,
    F: QueryFilter + Clone,
    Func: FnMut(&mut Query<Q, F>, &mut Store) + Send + 'static,
{
    QuerySystem::new(name, filter, func)
}

/// 创建无过滤条件的通用 Query 系统
pub fn simple_query_system<Q, Func>(
    name: &'static str,
    func: Func,
) -> QuerySystem<Q, All, Func>
where
    Q: QueryMutShape,
    Func: FnMut(&mut Query<Q, All>, &mut Store) + Send + 'static,
{
    QuerySystem::new(name, All, func)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
    struct Pos {
        x: f32,
        y: f32,
    }

    #[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
    struct Vel {
        dx: f32,
        dy: f32,
    }

    #[test]
    fn test_query_system_access_deduction() {
        let sys = simple_query_system::<(&mut Pos, &Vel), _>("movement", |q, store| {
            q.for_each_mut(store, |(_e, (p, v))| {
                p.x += v.dx;
                p.y += v.dy;
            })
            .unwrap();
        });

        assert_eq!(sys.name(), "movement");
        assert!(!sys.access().exclusive);
        assert_eq!(sys.access().writes.len(), 1); // Pos
        assert_eq!(sys.access().reads.len(), 1);  // Vel
        assert!(sys.access().writes.contains(&crate::id::ComponentTypeId::of::<Pos>()));
        assert!(sys.access().reads.contains(&crate::id::ComponentTypeId::of::<Vel>()));
    }
}
