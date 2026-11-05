use super::*;

impl Store {
    pub(super) fn preverify_entries_and_native_types<const NATIVE: bool>(
        &self,
        entries: &impl PlanningInput,
        cache_across_entities: bool,
        native_values: &mut NativeLanes,
        new_definitions: &mut BTreeMap<ComponentTypeId, crate::storage::registry::ComponentDefinition>,
        prototypes: &mut BTreeMap<ComponentTypeId, Box<dyn AnyColumn>>,
    ) -> Result<usize, StateError> {
        let mut verified_cache: smallvec::SmallVec<
            [(ComponentTypeId, Option<std::any::TypeId>, &'static str); 8],
        > = smallvec::SmallVec::new();
        let mut total_values = 0;
        let mut err = None;
        entries.for_each_value(|value| {
            total_values += 1;
            if err.is_some() {
                return;
            }
            let type_id = value.component_type_id();
            let cached = verified_cache
                .iter()
                .find(|&&(tid, _, _)| tid == type_id)
                .map(|&(_, expected_rust_id, expected_name)| (expected_rust_id, expected_name));
            let verified = cached.or_else(|| self.component_registry.get_type_info(&type_id));

            if let Some((expected_rust_id, expected_name)) = verified {
                let val_name = value.type_name();
                if expected_rust_id.is_some_and(|expected| expected != value.rust_type_id())
                    || (!std::ptr::eq(val_name, expected_name) && val_name != expected_name)
                {
                    err = Some(StateError::AccessDenied(format!(
                        "Native component {} ({:?}) does not match column {} ({:?})",
                        val_name,
                        value.rust_type_id(),
                        expected_name,
                        expected_rust_id,
                    )));
                } else if cache_across_entities && cached.is_none() {
                    if verified_cache.len() == 8 {
                        verified_cache.remove(0);
                    }
                    verified_cache.push((type_id, expected_rust_id, expected_name));
                }
            } else if let Some(def) = new_definitions.get(&type_id) {
                if let Err(e) = def.validate_value_type(value) {
                    err = Some(e);
                } else if cache_across_entities {
                    let rust_id = def.rust_type_id();
                    let name = def.type_name();
                    if verified_cache.len() == 8 {
                        verified_cache.remove(0);
                    }
                    verified_cache.push((type_id, rust_id, name));
                }
            } else {
                let col = value.empty_column();
                if let Err(e) = col.validate_erased_for_push(value) {
                    err = Some(e);
                } else {
                    let rust_id = col.rust_type_id();
                    let name = col.type_name();
                    if cache_across_entities {
                        if verified_cache.len() == 8 {
                            verified_cache.remove(0);
                        }
                        verified_cache.push((type_id, rust_id, name));
                    }
                    prototypes.insert(type_id, col.empty_clone());
                    new_definitions
                        .insert(type_id, crate::storage::registry::ComponentDefinition::new(col));
                }
            }
        });
        if let Some(e) = err {
            return Err(e);
        }

        if NATIVE {
            for lane in native_values.iter_mut() {
                let meta = lane.metadata();
                if let Some(def) = self.component_registry
                    .get(&meta.type_id)
                    .or_else(|| new_definitions.get(&meta.type_id))
                {
                    if def.rust_type_id().is_some_and(|t| t != meta.rust_type_id)
                        || def.type_name() != meta.name
                    {
                        return Err(StateError::AccessDenied(
                            "Native batch type does not match registered component".into(),
                        ));
                    }
                } else {
                    let column = lane.empty_column();
                    prototypes.insert(meta.type_id, column.empty_clone());
                    new_definitions.insert(
                        meta.type_id,
                        crate::storage::registry::ComponentDefinition::new(column),
                    );
                }
                lane.mark_type_verified();
            }
        }
        Ok(total_values)
    }
}
