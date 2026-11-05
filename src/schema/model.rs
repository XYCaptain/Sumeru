//! 模型模式声明与组件读写屏障契约 (ModelSchema, BehaviorAccessPlan)

use crate::context::ModelComponent;
use crate::id::{ComponentTypeId, SchemaVersion};
use crate::schema::component::ComponentSchema;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

/// 行为组件访问屏障计划 (BehaviorAccessPlan)。
///
/// 严格定义执行行为对组件集合的读取与写入访问集，保证单写多读的因果安全性。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BehaviorAccessPlan {
    pub behavior_name: String,
    pub order: u32,
    pub read_set: BTreeSet<ComponentTypeId>,
    pub write_set: BTreeSet<ComponentTypeId>,
}

impl BehaviorAccessPlan {
    pub fn new(
        behavior_name: impl Into<String>,
        order: u32,
        read_set: impl IntoIterator<Item = ComponentTypeId>,
        write_set: impl IntoIterator<Item = ComponentTypeId>,
    ) -> Self {
        Self {
            behavior_name: behavior_name.into(),
            order,
            read_set: read_set.into_iter().collect(),
            write_set: write_set.into_iter().collect(),
        }
    }
}

/// 权威模型模式数据契约 (ModelSchema)。
///
/// 供注册、执行与恢复跨域共享。纯粹声明数据布局与读写屏障，
/// 严格不包含任何 LP、时钟或 P-DEVS 推进协议语义。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelSchema {
    pub model_name: String,
    pub version: SchemaVersion,
    pub schema_digest: u64,
    pub components: BTreeMap<ComponentTypeId, ComponentSchema>,
    pub behaviors: Vec<BehaviorAccessPlan>,
}

impl ModelSchema {
    pub fn new(
        model_name: impl Into<String>,
        version: SchemaVersion,
        components: impl IntoIterator<Item = ComponentSchema>,
        mut behaviors: Vec<BehaviorAccessPlan>,
    ) -> Result<Self, String> {
        let model_name = model_name.into();
        if model_name.trim().is_empty() || version.as_u32() == 0 {
            return Err("Model schema name and version must be valid".into());
        }
        let mut component_map = BTreeMap::new();
        for component in components {
            if component.type_id == ComponentTypeId::default()
                || component.type_name.trim().is_empty()
                || component.type_id != ComponentTypeId::from_name(&component.type_name)
                || component.version.as_u32() == 0
                || component.codec_digest == 0
                || !component.alignment.is_power_of_two()
                || component.size % component.alignment != 0
                || component_map.insert(component.type_id, component).is_some()
            {
                return Err("Invalid or duplicate component schema".into());
            }
        }
        if component_map.is_empty() {
            return Err("Model schema must declare at least one component".into());
        }
        behaviors.sort_by_key(|behavior| behavior.order);
        let mut orders = BTreeSet::new();
        let mut behavior_names = BTreeSet::new();
        let mut writers = BTreeSet::new();
        for behavior in &behaviors {
            if behavior.behavior_name.trim().is_empty()
                || !orders.insert(behavior.order)
                || !behavior_names.insert(behavior.behavior_name.clone())
            {
                return Err("Behavior names and execution order must be unique".into());
            }
            for component in behavior.read_set.union(&behavior.write_set) {
                if !component_map.contains_key(component) {
                    return Err(format!(
                        "Behavior {} references undeclared component {component}",
                        behavior.behavior_name
                    ));
                }
            }
            for component in &behavior.write_set {
                if !writers.insert(*component) {
                    return Err(format!(
                        "Component {component} has multiple writers; use one explicit composite behavior"
                    ));
                }
            }
        }
        let mut material = format!("{model_name}:{}", version.as_u32());
        for component in component_map.values() {
            material.push_str(&format!(
                ":{}:{}:{}:{}:{}:{}",
                component.type_id.as_u64(),
                component.type_name,
                component.version.as_u32(),
                component.codec_digest,
                component.size,
                component.alignment
            ));
        }
        for behavior in &behaviors {
            material.push_str(&format!(":{}:{}", behavior.order, behavior.behavior_name));
            for id in &behavior.read_set {
                material.push_str(&format!(":r{}", id.as_u64()));
            }
            for id in &behavior.write_set {
                material.push_str(&format!(":w{}", id.as_u64()));
            }
        }
        Ok(Self {
            model_name,
            version,
            schema_digest: super::stable_digest(&material),
            components: component_map,
            behaviors,
        })
    }

    pub fn single_component<T: ModelComponent + Serialize>(model_name: &'static str) -> Self {
        let component = ComponentSchema::of::<T>(SchemaVersion::V1);
        let id = component.type_id;
        Self::new(
            model_name,
            SchemaVersion::V1,
            [component],
            vec![BehaviorAccessPlan::new(model_name, 0, [id], [id])],
        )
        .expect("single-component schema is valid")
    }

    pub fn validate_digest(&self) -> Result<(), String> {
        if self
            .components
            .iter()
            .any(|(key, component)| *key != component.type_id)
        {
            return Err(format!(
                "Model schema component map key mismatch for {}",
                self.model_name
            ));
        }
        let rebuilt = Self::new(
            self.model_name.clone(),
            self.version,
            self.components.values().cloned(),
            self.behaviors.clone(),
        )?;
        (rebuilt.schema_digest == self.schema_digest)
            .then_some(())
            .ok_or_else(|| format!("Model schema digest mismatch for {}", self.model_name))
    }
}
