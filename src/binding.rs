use serde::{Deserialize, Serialize};
use std::fmt::{Display, Formatter};

/// 实体状态物理权威分类 (StateClass)。
///
/// 决定状态组件的物理存储特征与生命周期归属。
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum StateClass {
    /// 业务服务域持久状态
    DurableService,
    /// 仿真推演域可回滚投机状态
    RollbackableModel,
    /// 已提交全局只读投影
    CommittedProjection,
    /// 算子执行瞬态内存
    EphemeralOperational,
}

impl Display for StateClass {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::DurableService => write!(f, "DurableService"),
            Self::RollbackableModel => write!(f, "RollbackableModel"),
            Self::CommittedProjection => write!(f, "CommittedProjection"),
            Self::EphemeralOperational => write!(f, "EphemeralOperational"),
        }
    }
}

pub use crate::id::{EntityId, ModelAddress};

