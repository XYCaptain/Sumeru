use super::*;
use std::collections::BTreeSet;

impl Store {
    pub(crate) fn prepare_input_values<const NATIVE: bool, const KEEP: bool>(
        &mut self,
        mut entries: impl PlanningInput,
        mut native_values: NativeLanes,
    ) -> Result<PreparedBatch, (StateError, Option<StoreBatch>)> {
        let is_recoverable = entries.is_recoverable_batch();
        if self.is_poisoned() {
            let recovered = is_recoverable.then(|| {
                unroll_planning_to_batch(
                    &mut Vec::new(),
                    Default::default(),
                    Vec::new(),
                    PlannedPatchValues::new(0),
                    None,
                    entries.drain_entries().0,
                )
            });
            return Err((StateError::SessionPoisoned, recovered));
        }

        let counts = entries.counts();
        let total_entries = entries.len();
        let cache_across_entities = total_entries > 1;
        // Promote pure native multi-component updates together. Mixed commands
        // keep their existing entity-major receipt contract. The compact lane
        // index overflow still uses the existing full-ID linked fallback.
        let columnar_native = KEEP && NATIVE && native_values.len() > 1
            && self.archetypes.len() <= usize::from(u16::MAX) + 1
            && entries.has_only_native_updates();

        let has_changes = counts.in_place_patches > 0
            || counts.structural_patches > 0
            || counts.conditional_patches > 0
            || counts.creates > 0
            || counts.deletes > 0
            || !native_values.is_empty();
        if has_changes {
            if let Err(e) = self.next_epoch() {
                let recovered = is_recoverable.then(|| {
                    unroll_planning_to_batch(
                        &mut Vec::new(),
                        Default::default(),
                        Vec::new(),
                        PlannedPatchValues::new(0),
                        None,
                        entries.drain_entries().0,
                    )
                });
                return Err((e, recovered));
            }
        }

        let in_place_capacity = counts.in_place_patches + counts.conditional_patches;
        let mut plan_scratch = PlanScratch::default();
        let PlanScratch {
            in_place_plans,
            native_references,
        } = &mut plan_scratch;
        if !NATIVE {
            in_place_plans.reserve_exact(in_place_capacity);
        }
        let in_place_reserve = in_place_capacity.saturating_mul(2).max(entries.pool_len());
        let mut in_place_values =
            crate::storage::batch::planned_patch_values::PlannedInPlaceValues::new(in_place_reserve);
        let mut native_plan_count = 0;
        // Upsert may only replace existing values. Reserve for known structural
        // work; actual conditional migrations grow these buffers as needed.
        // A single insertion must not reserve for every conditional entity.
        let structural_capacity = counts.creates + counts.deletes + counts.structural_patches;
        // A bounded, lazy seed avoids repeated tiny reallocations for small
        // inserting-upsert batches without reserving a large conditional batch.
        let conditional_seed = counts.conditional_patches.min(16);
        let structural_val_hint =
            entries.pool_len().max(counts.creates) + counts.structural_patches + conditional_seed;
        let mut plans = Vec::with_capacity(structural_capacity);
        let mut structural_values = PlannedPatchValues::new(structural_val_hint);
        let mut new_definitions =
            BTreeMap::<ComponentTypeId, crate::storage::registry::ComponentDefinition>::new();
        let mut prototypes = BTreeMap::<ComponentTypeId, Box<dyn AnyColumn>>::new();
        if let Err(e) = self.preverify_entries_and_native_types::<NATIVE>(
            &entries,
            cache_across_entities,
            &mut native_values,
            &mut new_definitions,
            &mut prototypes,
        ) {
            let recovered = is_recoverable.then(|| {
                unroll_planning_to_batch(
                    in_place_plans,
                    in_place_values,
                    plans,
                    structural_values,
                    None,
                    entries.drain_entries().0,
                )
            });
            return Err((e, recovered));
        }
        let mut layouts = BTreeMap::new();
        let mut structural_routes = BTreeMap::<StructuralRouteKey, u32>::new();
        let mut last_create: Option<(
            Vec<ComponentTypeId>,
            u32,
            smallvec::SmallVec<[ComponentTypeId; 8]>,
            [u8; 8],
        )> = None;
        let mut used_create_pool = false;
        let mut pool_has_drop = false;
        let mut validated_archetypes = ArchetypeValidationBitmap::new();
        let mut last_validated_arch: Option<u32> = None;
        let mut detached = Vec::new();
        let mut additions = BTreeMap::<u32, usize>::new();
        let mut last_addition: Option<(u32, usize)> = None;
        let mut creates = 0;
        let mut receipt = BatchReceipt::default();
        if KEEP && counts.creates > 0 {
            receipt.created_entities.reserve_exact(counts.creates);
        }
        let total_updates =
            counts.in_place_patches + counts.structural_patches + counts.conditional_patches;
        if KEEP && total_updates > 0 {
            receipt.updated_entities.reserve_exact(total_updates);
        }
        if KEEP && counts.deletes > 0 {
            receipt.deleted_entities.reserve_exact(counts.deletes);
        }

        let (mut entries_iter, mut pool) = entries.drain_entries();
        {
            let mut column_cache = ColumnCacheTracker::new(cache_across_entities);
            while let Some((id, entry)) = entries_iter.next() {
                macro_rules! plan_fail {
                    ($err:expr) => {
                        return Err((
                            $err,
                            is_recoverable.then(|| {
                                unroll_planning_to_batch(
                                    in_place_plans,
                                    in_place_values,
                                    plans,
                                    structural_values,
                                    Some((id, entry)),
                                    entries_iter,
                                )
                            }),
                        ))
                    };
                }

                match &entry {
                    CommandEntry::Create(bundle) => {
                        column_cache.deactivate();
                        if id.is_nil() {
                            plan_fail!(StateError::AccessDenied(
                                "Cannot create nil identity".into(),
                            ));
                        }
                    if self.allocator.is_alive(id) {
                        plan_fail!(StateError::EntityAlreadyExists(id));
                    }

                    let is_contiguous_pooled = bundle.components().is_empty()
                        && bundle.pool_len > 0
                        && bundle.pool_len <= 8
                        && bundle.pool_tail == bundle.pool_head.wrapping_add(bundle.pool_len - 1);

                    if is_contiguous_pooled {
                        if let Some((_, last_target, ref last_raw, last_perm)) =
                            last_create
                        {
                            let start = bundle.pool_head as usize;
                            let len = bundle.pool_len as usize;
                            if last_raw.len() == len
                                && pool[start..start + len]
                                    .iter()
                                    .zip(last_raw.iter())
                                    .all(|(pe, &cid)| pe.type_id == cid)
                            {
                                creates += 1;
                                if KEEP {
                                    receipt.created_entities.push(id);
                                }
                                match &mut last_addition {
                                    Some((aid, cnt)) if *aid == last_target => *cnt += 1,
                                    Some((aid, cnt)) => {
                                        *additions.entry(*aid).or_default() += *cnt;
                                        *aid = last_target;
                                        *cnt = 1;
                                    }
                                    None => last_addition = Some((last_target, 1)),
                                }
                                used_create_pool = true;
                                plans.push(PlannedOp::CreatePooled {
                                    entity: id,
                                    target_archetype: last_target,
                                    pool_start: bundle.pool_head,
                                    perm: last_perm,
                                    len: len as u8,
                                });
                                continue;
                            }
                        }
                    }

                    let mut raw_cids: smallvec::SmallVec<[ComponentTypeId; 8]> =
                        smallvec::SmallVec::new();
                    let mut type_keys: smallvec::SmallVec<[ComponentTypeId; 8]> =
                        smallvec::SmallVec::with_capacity(bundle.len());
                    type_keys.extend(bundle.components().keys().copied());
                    bundle.for_each_pooled_cid(pool, |cid| {
                        type_keys.push(cid);
                        if is_contiguous_pooled {
                            raw_cids.push(cid);
                        }
                    });
                    if bundle.pool_len > 0 {
                        type_keys.sort_unstable();
                    }

                    let is_same_create = if let Some((ref last_types, _, _, _)) = last_create {
                        type_keys.as_slice() == last_types.as_slice()
                    } else {
                        false
                    };

                    let mut cached_perm = [0u8; 8];
                    let target_id = if is_same_create {
                        if is_contiguous_pooled {
                            for (i, &scid) in type_keys.iter().enumerate() {
                                if let Some(pos) = raw_cids.iter().position(|&rcid| rcid == scid) {
                                    cached_perm[i] = pos as u8;
                                }
                            }
                        }
                        last_create.as_ref().unwrap().1
                    } else {
                        let comp_keys: Vec<_> = if cache_across_entities {
                            type_keys.to_vec()
                        } else {
                            Vec::new()
                        };
                        let types: BTreeSet<ComponentTypeId> =
                            type_keys.iter().copied().collect();
                        let target = if let Some(&id) = layouts.get(&types) {
                            id
                        } else if let Some(&id) = self.archetype_index.get(&types) {
                            if !cache_across_entities {
                                if let Err(e) = self.validate_archetype_layout(id as usize) {
                                    plan_fail!(e);
                                }
                            } else if last_validated_arch != Some(id) {
                                if validated_archetypes.insert(id) {
                                    if let Err(e) = self.validate_archetype_layout(id as usize) {
                                        plan_fail!(e);
                                    }
                                }
                                last_validated_arch = Some(id);
                            }
                            id
                        } else {
                            let id = match u32::try_from(self.archetypes.len() + detached.len()) {
                                Ok(i) => i,
                                Err(_) => plan_fail!(StateError::CapacityExhausted(
                                    "Archetype count exhausted".into(),
                                )),
                            };
                            let mut arch = Archetype::new(ArchetypeId(id), types.clone());
                            for type_id in &types {
                                let column = prototypes
                                    .get(type_id)
                                    .map(|p| p.empty_clone())
                                    .or_else(|| new_definitions.get(type_id).map(|d| d.create_empty_column()))
                                    .or_else(|| self.component_registry.spawn_column(*type_id))
                                    .or_else(|| self.find_prototype(*type_id));
                                let column = match column {
                                    Some(col) => col,
                                    None => plan_fail!(StateError::ComponentNotFound(format!("{type_id}"))),
                                };
                                arch.columns_mut().insert(*type_id, column);
                            }
                            detached.push(arch);
                            id
                        };
                        if is_contiguous_pooled {
                            let start = bundle.pool_head as usize;
                            let len = bundle.pool_len as usize;
                            if !pool_has_drop {
                                for pe in &pool[start..start + len] {
                                    if pe.op.needs_drop() {
                                        pool_has_drop = true;
                                        break;
                                    }
                                }
                            }
                            for (i, &scid) in type_keys.iter().enumerate() {
                                if let Some(pos) = raw_cids.iter().position(|&rcid| rcid == scid) {
                                    cached_perm[i] = pos as u8;
                                }
                            }
                        }
                        if cache_across_entities {
                            layouts.insert(types, target);
                            last_create = Some((comp_keys, target, raw_cids, cached_perm));
                        }
                        target
                    };

                    creates += 1;
                    if KEEP {
                        receipt.created_entities.push(id);
                    }
                    match &mut last_addition {
                        Some((aid, cnt)) if *aid == target_id => *cnt += 1,
                        Some((aid, cnt)) => {
                            *additions.entry(*aid).or_default() += *cnt;
                            *aid = target_id;
                            *cnt = 1;
                        }
                        None => last_addition = Some((target_id, 1)),
                    }
                    if is_contiguous_pooled {
                        used_create_pool = true;
                        plans.push(PlannedOp::CreatePooled {
                            entity: id,
                            target_archetype: target_id,
                            pool_start: bundle.pool_head,
                            perm: cached_perm,
                            len: bundle.pool_len as u8,
                        });
                    } else {
                        let CommandEntry::Create(bundle) = entry else { unreachable!() };
                        let sorted_components = bundle.take_sorted_components(pool);
                        let components = structural_values.append_values(sorted_components);
                        plans.push(PlannedOp::Create {
                            entity: id,
                            target_archetype: target_id,
                            components,
                        });
                    }
                }
                CommandEntry::NativeUpdate(head) => {
                    let loc = match self.allocator.get_location(id) {
                        Some(l) => l,
                        None => plan_fail!(StateError::EntityNotFound(id)),
                    };
                    let src_arch = match self.archetypes.get(loc.archetype_id as usize) {
                        Some(a) => a,
                        None => plan_fail!(StateError::AccessDenied(format!(
                            "Unknown archetype {}",
                            loc.archetype_id
                        ))),
                    };
                    if src_arch.entities().get(loc.row_index as usize) != Some(&id) {
                        plan_fail!(StateError::AccessDenied(format!(
                            "Entity {id} has inconsistent archetype row"
                        )));
                    }

                    if !cache_across_entities {
                        if let Err(e) = self.validate_archetype_layout(loc.archetype_id as usize) {
                            plan_fail!(e);
                        }
                    } else if last_validated_arch != Some(loc.archetype_id) {
                        if validated_archetypes.insert(loc.archetype_id) {
                            if let Err(e) = self.validate_archetype_layout(loc.archetype_id as usize) {
                                plan_fail!(e);
                            }
                        }
                        last_validated_arch = Some(loc.archetype_id);
                    }

                    if columnar_native {
                        // Native lanes validate/bind through their own existing
                        // archetype cache. This branch never reads the ordinary
                        // column directory; do not populate/switch it per entity.
                        let mut next = Some(*head);
                        while let Some(reference) = next {
                            let lane = &mut native_values[reference.lane()];
                            next = lane.next(reference.row());
                            if let Err(err) = lane.plan_columnar_member(
                                reference.row(), id, loc, src_arch, receipt.updated_entities.len(),
                            ) {
                                plan_fail!(err);
                            }
                        }
                        native_plan_count += 1;
                        receipt.updated_entities.push(id);
                        continue;
                    }

                    let lane = &mut native_values[head.lane()];
                    // This append-only ID list is shared by every update path.
                    // Standalone native targets reference it instead of copying IDs.
                    // Neither reordering nor mutation is permitted before lane commit.
                    match lane.try_plan_single(head.row(), id, loc, src_arch, receipt.updated_entities.len()) {
                        Ok(true) => {
                            native_plan_count += 1;
                            if KEEP {
                                receipt.updated_entities.push(id);
                            }
                            continue;
                        }
                        Ok(false) => {}
                        Err(e) => plan_fail!(e),
                    }
                    // Preserve typed ownership through commit. The common
                    // planner already validated entity identity and row layout;
                    // each lane reuses its existing per-archetype type cache.
                    if native_references.is_empty() {
                        native_references.reserve(in_place_capacity.saturating_mul(2));
                    }
                    let start = native_references.len();
                    let mut next = Some(*head);
                    while let Some(reference) = next {
                        let lane = &mut native_values[reference.lane()];
                        let (type_id, following) = match lane.plan_linked(reference.row(), id, loc, src_arch) {
                            Ok(value) => value,
                            Err(e) => plan_fail!(e),
                        };
                        native_references.push((type_id, reference));
                        next = following;
                    }
                    let range = start..native_references.len();
                    let replacements = InPlaceReplacements::NativeMultiple(range);
                    let pool_index = if KEEP {
                        receipt.updated_entities.push(id);
                        (receipt.updated_entities.len() - 1) as u32
                    } else {
                        0
                    };
                    in_place_plans.push(PlannedInPlacePatch {
                        entity: id,
                        location: loc,
                        replacements,
                        pool_index,
                    });
                }
                CommandEntry::Patch(patch) => {
                    let loc = match self.allocator.get_location(id) {
                        Some(l) => l,
                        None => plan_fail!(StateError::EntityNotFound(id)),
                    };
                    let src_arch = match self.archetypes.get(loc.archetype_id as usize) {
                        Some(a) => a,
                        None => plan_fail!(StateError::AccessDenied(format!(
                            "Unknown archetype {}",
                            loc.archetype_id
                        ))),
                    };
                    if src_arch.entities().get(loc.row_index as usize) != Some(&id) {
                        plan_fail!(StateError::AccessDenied(format!(
                            "Entity {id} has inconsistent archetype row"
                        )));
                    }

                    if !cache_across_entities {
                        if let Err(e) = self.validate_archetype_layout(loc.archetype_id as usize) {
                            plan_fail!(e);
                        }
                    } else if last_validated_arch != Some(loc.archetype_id) {
                        if validated_archetypes.insert(loc.archetype_id) {
                            if let Err(e) = self.validate_archetype_layout(loc.archetype_id as usize) {
                                plan_fail!(e);
                            }
                        }
                        last_validated_arch = Some(loc.archetype_id);
                    }

                    column_cache.visit_archetype(loc.archetype_id);

                    if patch.is_empty() && patch.native_head.is_none() {
                        continue;
                    }

                    let mut patch_err = None;
                    let mut has_structural_change = false;
                    // Record only actual additions/removals during validation.
                    // A cached route must not bypass any entity/value checks.
                    let mut route: Option<StructuralRouteKey> = None;

                    patch.for_each_op_with_pool(Some(&pool), |type_id, patch_op| {
                        if patch_err.is_some() {
                            return;
                        }
                        match patch_op {
                            PatchOp::Insert(value) => {
                                if src_arch.component_types().contains(&type_id) {
                                    patch_err = Some(StateError::ComponentAlreadyExists {
                                        entity: id,
                                        type_name: value.type_name().to_string(),
                                    });
                                    return;
                                }
                                has_structural_change = true;
                                route.get_or_insert_with(|| StructuralRouteKey::new(loc.archetype_id))
                                    .changes.push((type_id, true));
                            }
                            PatchOp::Replace(value) => {
                                // 复用前置扫描的类型验证结果：validate_or_lookup 对同原型
                                // 同组件仅做一次列契约验证，后续实体直接跳过。
                                // 行界限由 validate_archetype_layout + entities().get() 保证。
                                match column_cache.validate_or_lookup(src_arch, type_id, &self.component_registry) {
                                    Ok(ColumnValidation::Verified) => {} // 列存在且类型已验证
                                    Ok(ColumnValidation::Fallback(col)) => {
                                        // 缺乏前置缓存（如单实体批次），安全回退：验证列能接纳该值并检查行界限
                                        if let Err(e) = col.validate_erased(loc.row_index as usize, value) {
                                            patch_err = Some(e);
                                        }
                                    }
                                    Ok(ColumnValidation::NotFound) => {
                                        patch_err = Some(StateError::ComponentNotFoundInEntity {
                                            entity: id,
                                            type_name: value.type_name().to_string(),
                                        });
                                    }
                                    Err(e) => {
                                        patch_err = Some(e);
                                    }
                                }
                            }
                            PatchOp::Upsert(value) => {
                                // Upsert: 列存在时复用前置扫描的类型验证结果；
                                // 列不存在时进入原有的新增/迁移预检路径。
                                match column_cache.validate_or_lookup(src_arch, type_id, &self.component_registry) {
                                    Ok(ColumnValidation::Verified) => {} // 列存在且类型已验证
                                    Ok(ColumnValidation::Fallback(col)) => {
                                        // 缺乏前置缓存（如单实体批次），安全回退：验证列能接纳该值并检查行界限
                                        if let Err(e) = col.validate_erased(loc.row_index as usize, value) {
                                            patch_err = Some(e);
                                        }
                                    }
                                    Ok(ColumnValidation::NotFound) => {
                                        // 列不存在 → 新增组件，进入迁移预检
                                        has_structural_change = true;
                                        route.get_or_insert_with(|| StructuralRouteKey::new(loc.archetype_id))
                                            .changes.push((type_id, true));
                                    }
                                    Err(e) => {
                                        patch_err = Some(e);
                                    }
                                }
                            }
                            PatchOp::Remove(removed_id) => {
                                if !src_arch.component_types().contains(&removed_id) {
                                    patch_err = Some(StateError::ComponentNotFoundInEntity {
                                        entity: id,
                                        type_name: format!("{removed_id}"),
                                    });
                                    return;
                                }
                                has_structural_change = true;
                                route.get_or_insert_with(|| StructuralRouteKey::new(loc.archetype_id))
                                    .changes.push((*removed_id, false));
                            }
                        }
                    });
                    if NATIVE && patch_err.is_none() {
                        let mut next = patch.native_head;
                        while let Some(head) = next {
                            let lane = &native_values[head.lane()];
                            let meta = lane.metadata();
                            let is_present = route.as_ref().map_or_else(
                                || src_arch.component_types().contains(&meta.type_id),
                                |route| route.contains_after(src_arch.component_types(), meta.type_id),
                            );
                            if !is_present {
                                patch_err = Some(StateError::ComponentNotFoundInEntity {
                                    entity: id,
                                    type_name: meta.name.to_string(),
                                });
                                break;
                            }
                            if let Some(col) = column_cache.get_or_insert_column(src_arch, meta.type_id) {
                                if col.type_id() != meta.type_id {
                                    patch_err = Some(StateError::TypeMismatch {
                                        expected: format!("{}", col.type_id()),
                                        actual: meta.name.to_string(),
                                    });
                                    break;
                                }
                            }
                            next = lane.next(head.row());
                        }
                    }
                    if let Some(e) = patch_err {
                        plan_fail!(e);
                    }

                    let target_arch_id = if has_structural_change {
                        let mut route = route.expect("structural changes have a route delta");
                        let cache_route = cache_across_entities && route.cacheable();
                        let cached = if cache_route {
                            route.changes.sort_unstable();
                            structural_routes.get(&route).copied()
                        } else { None };
                        if let Some(tid) = cached {
                            Some(tid)
                        } else {
                            // Only a route miss constructs the full signature.
                            let target_types = route.target_types(src_arch.component_types());
                            let layout_cached = layouts.get(&target_types).copied();
                            let tid = if let Some(tid) = layout_cached {
                                tid
                            } else if let Some(&tid) = self.archetype_index.get(&target_types) {
                                if !cache_across_entities || validated_archetypes.insert(tid) {
                                    if let Err(e) = self.validate_archetype_layout(tid as usize) {
                                        plan_fail!(e);
                                    }
                                }
                                tid
                            } else {
                                let tid = match u32::try_from(self.archetypes.len() + detached.len()) {
                                    Ok(t) => t,
                                    Err(_) => plan_fail!(StateError::CapacityExhausted(
                                        "Archetype count exhausted".into(),
                                    )),
                                };
                                let mut arch =
                                    Archetype::new(ArchetypeId(tid), target_types.clone());
                                for type_id in &target_types {
                                    let column = prototypes
                                        .get(type_id)
                                        .map(|p| p.empty_clone())
                                        .or_else(|| new_definitions.get(type_id).map(|d| d.create_empty_column()))
                                        .or_else(|| self.component_registry.spawn_column(*type_id))
                                        .or_else(|| self.find_prototype(*type_id));
                                    let column = match column {
                                        Some(col) => col,
                                        None => plan_fail!(StateError::ComponentNotFound(format!("{type_id}"))),
                                    };
                                    arch.columns_mut().insert(*type_id, column);
                                }
                                detached.push(arch);
                                tid
                            };
                            if cache_across_entities && layout_cached.is_none() {
                                layouts.insert(target_types, tid);
                            }
                            if cache_route
                                && structural_routes.len() < MAX_STRUCTURAL_ROUTES
                            {
                                structural_routes.insert(route, tid);
                            }
                            Some(tid)
                        }
                    } else {
                        None
                    };

                    let CommandEntry::Patch(mut patch) = entry else { unreachable!() };
                    let native_head = if NATIVE {
                        patch.native_head.take()
                    } else {
                        None
                    };
                    let compact_ops = patch.into_compact_ops();

                    let is_single = matches!(compact_ops, CompactPatchOps::Single(_, _));
                    if is_single && native_head.is_none() && !has_structural_change {
                        if let CompactPatchOps::Single(type_id, op) = compact_ops {
                            let in_place_op = op.into_in_place_op();
                            let pool_index = if KEEP {
                                receipt.updated_entities.push(id);
                                (receipt.updated_entities.len() - 1) as u32
                            } else {
                                0
                            };
                            in_place_plans.push(PlannedInPlacePatch {
                                entity: id,
                                location: loc,
                                replacements: InPlaceReplacements::Single(type_id, in_place_op),
                                pool_index,
                            });
                            continue;
                        }
                    }

                    // Preserve only an already allocated ordinary buffer.
                    // The entity plan and the single/native execution paths
                    // keep their original representation.
                    if native_head.is_none() && !has_structural_change
                        && matches!(compact_ops, CompactPatchOps::Multiple(_))
                    {
                        let CompactPatchOps::Multiple(values) = compact_ops else { unreachable!() };
                        let values = values.into_iter().map(|(ty, op)| (ty, op.into_in_place_op())).collect();
                        let range = in_place_values.append_owned(values);
                        let pool_index = if KEEP {
                            receipt.updated_entities.push(id);
                            (receipt.updated_entities.len() - 1) as u32
                        } else {
                            0
                        };
                        in_place_plans.push(PlannedInPlacePatch {
                            entity: id, location: loc,
                            replacements: InPlaceReplacements::Multiple(range),
                            pool_index,
                        });
                        continue;
                    }

                    if native_head.is_none() && !has_structural_change
                        && matches!(compact_ops, CompactPatchOps::Pooled { .. })
                    {
                        let CompactPatchOps::Pooled { head, len, .. } = compact_ops else { unreachable!() };
                        struct PooledInPlaceIter<'a> {
                            curr: u32,
                            remaining: usize,
                            pool: &'a mut [crate::storage::batch::PoolEntry],
                        }
                        impl<'a> Iterator for PooledInPlaceIter<'a> {
                            type Item = (crate::ComponentTypeId, crate::storage::core::column::InPlaceOp);
                            #[inline]
                            fn next(&mut self) -> Option<Self::Item> {
                                if self.curr == u32::MAX {
                                    return None;
                                }
                                let entry = &mut self.pool[self.curr as usize];
                                self.curr = entry.next;
                                self.remaining = self.remaining.saturating_sub(1);
                                let op = std::mem::replace(&mut entry.op, crate::storage::batch::TOMBSTONE_OP);
                                let in_place_op = op.into_in_place_op();
                                Some((entry.type_id, in_place_op))
                            }
                            #[inline]
                            fn size_hint(&self) -> (usize, Option<usize>) {
                                (self.remaining, Some(self.remaining))
                            }
                        }
                        impl<'a> ExactSizeIterator for PooledInPlaceIter<'a> {}

                        let ops = PooledInPlaceIter {
                            curr: head,
                            remaining: len,
                            pool: &mut pool,
                        };
                        let range = in_place_values.append(ops);
                        let pool_index = if KEEP {
                            receipt.updated_entities.push(id);
                            (receipt.updated_entities.len() - 1) as u32
                        } else {
                            0
                        };
                        in_place_plans.push(PlannedInPlacePatch {
                            entity: id, location: loc,
                            replacements: InPlaceReplacements::Multiple(range),
                            pool_index,
                        });
                        continue;
                    }

                    let mut next = native_head;
                    let native_ops = std::iter::from_fn(|| {
                        let head = next?;
                        let lane = &mut native_values[head.lane()];
                        let (value, following) = lane.take(head.row());
                        next = following;
                        Some((value.component_type_id(), PatchOp::Replace(value)))
                    });
                    let ops = compact_ops.into_iter_with_pool(Some(&mut pool)).chain(native_ops);
                    if !has_structural_change {
                        // All component/row checks above finished before moving
                        // input. Keep multi-component values in one batch-owned
                        // array instead of SmallList -> per-entity Vec.
                        let range = in_place_values.append(ops.map(|(type_id, op)| {
                            (type_id, op.into_in_place_op())
                        }));
                        if native_head.is_some() {
                            in_place_values.sort_range(range.clone());
                        }
                        if !range.is_empty() {
                            let pool_index = if KEEP {
                                receipt.updated_entities.push(id);
                                (receipt.updated_entities.len() - 1) as u32
                            } else {
                                0
                            };
                            in_place_plans.push(PlannedInPlacePatch {
                                entity: id,
                                location: loc,
                                replacements: InPlaceReplacements::Multiple(range),
                                pool_index,
                            });
                        }
                        continue;
                    }

                    if plans.is_empty() {
                        plans.reserve(conditional_seed);
                    }
                    let mut in_place_replacements = SmallList::new();
                    let mut inserts = SmallList::new();
                    let mut removes = SmallList::new();
                    for (type_id, patch_op) in ops {
                        match patch_op {
                            PatchOp::Insert(value) => {
                                inserts.insert(type_id, value);
                            }
                            PatchOp::Replace(value) => {
                                in_place_replacements.push((type_id, value));
                            }
                            PatchOp::Upsert(value) => {
                                if src_arch.columns().contains_key(&type_id) {
                                    in_place_replacements.push((type_id, value));
                                } else {
                                    inserts.insert(type_id, value);
                                }
                            }
                            PatchOp::Remove(removed_id) => {
                                removes.push(removed_id);
                            }
                        }
                    }

                    {
                        let target_id = target_arch_id.unwrap();
                        *additions.entry(target_id).or_default() += 1;
                        if KEEP {
                            receipt.updated_entities.push(id);
                        }
                        removes.as_mut_slice().sort_unstable();
                        inserts.as_mut_slice().sort_unstable_by_key(|(id, _)| *id);
                        in_place_replacements.as_mut_slice().sort_unstable_by_key(|(id, _)| *id);
                        let new_components = structural_values.append_values(inserts);
                        let replaced_components = structural_values.append_values(in_place_replacements);
                        let removed_components = structural_values.append_removed(removes);
                        let pool_index = if KEEP {
                            (receipt.updated_entities.len() - 1) as u32
                        } else {
                            0
                        };
                        plans.push(PlannedOp::StructuralPatch {
                            entity: id,
                            location: loc,
                            target_archetype: target_id,
                            new_components,
                            replaced_components,
                            removed_components,
                            pool_index,
                        });
                    }
                }
                CommandEntry::Delete => {
                    column_cache.deactivate();
                    if !self.allocator.is_alive(id) {
                        plan_fail!(StateError::EntityNotFound(id));
                    }
                    if let Err(e) = self.allocator.validate_deallocate(id) {
                        plan_fail!(e);
                    }
                    let loc = match self.allocator.get_location(id) {
                        Some(l) => l,
                        None => plan_fail!(StateError::EntityNotFound(id)),
                    };
                    let arch = match self.archetypes.get(loc.archetype_id as usize) {
                        Some(a) => a,
                        None => plan_fail!(StateError::AccessDenied(format!(
                            "Unknown archetype {}",
                            loc.archetype_id
                        ))),
                    };
                    if arch.entities().get(loc.row_index as usize) != Some(&id) {
                        plan_fail!(StateError::AccessDenied(format!(
                            "Entity {id} has inconsistent archetype row"
                        )));
                    }
                    if !cache_across_entities {
                        if let Err(e) = self.validate_archetype_layout(loc.archetype_id as usize) {
                            plan_fail!(e);
                        }
                    } else if last_validated_arch != Some(loc.archetype_id) {
                        if validated_archetypes.insert(loc.archetype_id) {
                            if let Err(e) = self.validate_archetype_layout(loc.archetype_id as usize) {
                                plan_fail!(e);
                            }
                        }
                        last_validated_arch = Some(loc.archetype_id);
                    }
                    if KEEP {
                        receipt.deleted_entities.push(id);
                    }
                    plans.push(PlannedOp::Delete {
                        entity: id,
                        location: loc,
                        pool_index: 0,
                    });
                }
            }
        }
        }
        drop(entries_iter);
        if used_create_pool {
            structural_values.set_pool(entries.take_pool(), pool_has_drop);
        } else {
            entries.mark_pool_consumed();
        }
        if let Some((aid, cnt)) = last_addition {
            *additions.entry(aid).or_default() += cnt;
        }

        macro_rules! post_loop_fail {
            ($err:expr) => {
                return Err((
                    $err,
                    is_recoverable.then(|| {
                        unroll_planning_to_batch(
                            in_place_plans,
                            in_place_values,
                            plans,
                            structural_values,
                            None,
                            std::iter::empty(),
                        )
                    }),
                ))
            };
        }

        for (&id, &count) in &additions {
            let len = if (id as usize) < self.archetypes.len() {
                self.archetypes[id as usize].len()
            } else {
                0
            };
            if len
                .checked_add(count)
                .is_none_or(|len| len > u32::MAX as usize)
            {
                post_loop_fail!(StateError::CapacityExhausted(
                    "Archetype rows exhausted".into(),
                ));
            }
        }
        if let Err(e) = self.allocator.prepare_allocations(creates) {
            post_loop_fail!(e);
        }
        self.archetypes.reserve(detached.len());
        for (&id, &count) in &additions {
            if let Some(arch) = self.archetypes.get_mut(id as usize) {
                arch.reserve(count);
            } else {
                detached[id as usize - self.archetypes.len()].reserve(count);
            }
        }

        let mut max_discarded_per_op = 0usize;
        for plan in &plans {
            match plan {
                PlannedOp::StructuralPatch { location, .. } | PlannedOp::Delete { location, .. } => {
                    if let Some(arch) = self.archetypes.get(location.archetype_id as usize) {
                        max_discarded_per_op = max_discarded_per_op.max(arch.columns().len());
                    }
                }
                _ => {}
            }
        }
        let discarded_values = Vec::with_capacity(max_discarded_per_op);

        let has_changes = native_plan_count != 0
            || in_place_plans.iter().any(|p| !p.replacements.is_empty())
            || !plans.is_empty();
        if has_changes {
            if let Err(e) = self.next_epoch() {
                post_loop_fail!(e);
            }
            // Reject exhausted layout generation while the original batch is
            // still recoverable, not after consuming the prepared plan.
            if !detached.is_empty() {
                if let Err(e) = self.check_archetype_set_generation_capacity() {
                    post_loop_fail!(e);
                }
            }
        }
        // Linked plans address lanes by recording index. Removing a consumed
        // mixed/structural lane would renumber all subsequent references.
        if native_references.is_empty() {
            native_values.retain(|lane| lane.planned_len() != 0);
        }
        let commit_mode = if KEEP {
            crate::storage::batch::PreparedCommitMode::KeepReceipt
        } else {
            crate::storage::batch::PreparedCommitMode::DiscardReceipt
        };
        Ok(PreparedBatch {
            store_id: self.store_id,
            epoch: self.epoch,
            commit_mode,
            plan_scratch,
            in_place_values,
            native_plans: native_values,
            native_plan_count,
            structural_values,
            plans,
            detached,
            new_definitions,
            receipt,
            discarded_values,
        })
    }
}
