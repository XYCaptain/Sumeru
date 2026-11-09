use super::*;
use crate::id::ComponentTypeId;
use crate::storage::query::merged_access_descriptors;
use crate::{EntityBundle, Store};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct Pos {
    x: i32,
    y: i32,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct Vel {
    dx: i32,
    dy: i32,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct TagActive;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct TagDisabled;

#[derive(Clone, Copy, Debug)]
struct PosX;
impl ComponentField<Pos> for PosX {
    type Value = i32;
    fn extract(component: &Pos) -> &Self::Value {
        &component.x
    }
    fn field_name() -> &'static str {
        "x"
    }
}

#[test]
fn test_query_builder_with_all_filters() {
    let mut store = Store::new();
    // Entity 1: Pos { x: 10, y: 20 }, Vel { dx: 1, dy: 2 }, TagActive
    store
        .create_entity(
            EntityId::from_u128(1),
            EntityBundle::new()
                .with(Pos { x: 10, y: 20 })
                .unwrap()
                .with(Vel { dx: 1, dy: 2 })
                .unwrap()
                .with(TagActive)
                .unwrap(),
        )
        .unwrap();

    // Entity 2: Pos { x: 10, y: 30 }, Vel { dx: 3, dy: 4 }, TagDisabled
    store
        .create_entity(
            EntityId::from_u128(2),
            EntityBundle::new()
                .with(Pos { x: 10, y: 30 })
                .unwrap()
                .with(Vel { dx: 3, dy: 4 })
                .unwrap()
                .with(TagDisabled)
                .unwrap(),
        )
        .unwrap();

    // Entity 3: Pos { x: 20, y: 50 }, Vel { dx: 5, dy: 6 }, TagActive
    store
        .create_entity(
            EntityId::from_u128(3),
            EntityBundle::new()
                .with(Pos { x: 20, y: 50 })
                .unwrap()
                .with(Vel { dx: 5, dy: 6 })
                .unwrap()
                .with(TagActive)
                .unwrap(),
        )
        .unwrap();

    // Build query requiring Pos, Vel, TagActive, without TagDisabled, Pos.x == 10, Vel.dx > 0
    let query = QueryBuilder::<(&mut Pos, &Vel)>::new()
        .with::<TagActive>()
        .without::<TagDisabled>()
        .filter_eq::<Pos, PosX>(10)
        .filter::<Vel>(|v| v.dx > 0)
        .build(&store)
        .unwrap();

    assert_eq!(query.store_id(), store.store_id());
    assert_eq!(
        query.archetype_set_generation(),
        store.archetype_set_generation()
    );
    // Entity 1's archetype matches; Entity 2 has TagDisabled; Entity 3 shares archetype with 1
    assert_eq!(query.matched_count(), 1);
    let arch_id = query.matched_archetypes()[0];
    let arch = store.archetypes.iter().find(|a| a.id() == arch_id).unwrap();
    assert!(arch.component_types().contains(&ComponentTypeId::of::<Pos>()));
    assert!(arch.component_types().contains(&ComponentTypeId::of::<Vel>()));
    assert!(arch.component_types().contains(&ComponentTypeId::of::<TagActive>()));
    assert!(!arch.component_types().contains(&ComponentTypeId::of::<TagDisabled>()));

    // Check QueryEntityRequest
    let ids = [EntityId::from_u128(1), EntityId::from_u128(3)];
    let req = query.entity_request(&ids);
    assert_eq!(req.query().store_id(), store.store_id());
    assert_eq!(req.ids(), &ids);
}

#[test]
fn test_merged_access_descriptors_permission_semantics() {
    // Query shape has (&mut Pos, &Vel): Pos is write (true), Vel is read (false)
    let builder = QueryBuilder::<(&mut Pos, &Vel)>::new()
        .with::<TagActive>() // with does NOT add read descriptor
        .filter_eq::<Pos, PosX>(10) // reads Pos, but Pos is already write in Q -> remains write
        .filter::<Vel>(|v| v.dx > 0) // reads Vel, Vel is read in Q -> remains read
        .filter::<TagDisabled>(|_| true); // reads TagDisabled, not in Q -> added as read

    let descriptors = merged_access_descriptors::<(&mut Pos, &Vel), _>(&builder.filter);
    assert_eq!(descriptors.len(), 3);

    let pos_desc = descriptors
        .iter()
        .find(|(cid, _, _)| *cid == ComponentTypeId::of::<Pos>())
        .expect("Pos must be present");
    assert!(pos_desc.2, "Pos was &mut in Q, so merged descriptor must be write=true");

    let vel_desc = descriptors
        .iter()
        .find(|(cid, _, _)| *cid == ComponentTypeId::of::<Vel>())
        .expect("Vel must be present");
    assert!(!vel_desc.2, "Vel was & in Q, so merged descriptor must be write=false");

    let tag_disabled_desc = descriptors
        .iter()
        .find(|(cid, _, _)| *cid == ComponentTypeId::of::<TagDisabled>())
        .expect("TagDisabled must be present from filter");
    assert!(!tag_disabled_desc.2, "TagDisabled added by filter must be write=false");

    assert!(
        descriptors
            .iter()
            .all(|(cid, _, _)| *cid != ComponentTypeId::of::<TagActive>()),
        "TagActive (from with) must NOT be present in access descriptors"
    );
}

#[test]
fn test_matches_row_for_filter_eq_and_filter() {
    let mut store = Store::new();
    // Row 0: Pos { x: 10, y: 100 }, Vel { dx: 5, dy: 0 }
    store
        .create_entity(
            EntityId::from_u128(10),
            EntityBundle::new()
                .with(Pos { x: 10, y: 100 })
                .unwrap()
                .with(Vel { dx: 5, dy: 0 })
                .unwrap(),
        )
        .unwrap();
    // Row 1: Pos { x: 20, y: 200 }, Vel { dx: -3, dy: 0 }
    store
        .create_entity(
            EntityId::from_u128(20),
            EntityBundle::new()
                .with(Pos { x: 20, y: 200 })
                .unwrap()
                .with(Vel { dx: -3, dy: 0 })
                .unwrap(),
        )
        .unwrap();

    let query = QueryBuilder::<(&Pos,)>::new()
        .filter_eq::<Pos, PosX>(10)
        .filter::<Vel>(|v| v.dx > 0)
        .build(&store)
        .unwrap();

    let arch_id = query.matched_archetypes()[0];
    let arch = store.archetypes.iter().find(|a| a.id() == arch_id).unwrap();

    // Row 0 has x=10 and dx=5 > 0: matches!
    assert!(query.filter().matches_row(arch, 0));
    // Row 1 has x=20 and dx=-3: does not match!
    assert!(!query.filter().matches_row(arch, 1));
    // Row 2 is out of bounds: does not match!
    assert!(!query.filter().matches_row(arch, 2));

    // Test filter_eq alone
    let q_eq = QueryBuilder::<(&Pos,)>::new()
        .filter_eq::<Pos, PosX>(20)
        .build(&store)
        .unwrap();
    assert!(!q_eq.filter().matches_row(arch, 0));
    assert!(q_eq.filter().matches_row(arch, 1));

    // Test filter predicate alone
    let q_pred = QueryBuilder::<(&Pos,)>::new()
        .filter::<Vel>(|v| v.dx < 0)
        .build(&store)
        .unwrap();
    assert!(!q_pred.filter().matches_row(arch, 0));
    assert!(q_pred.filter().matches_row(arch, 1));
}

#[test]
fn test_query_ensure_cache_current_on_generation_change() {
    let mut store = Store::new();
    // Create entity with Pos and Vel
    store
        .create_entity(
            EntityId::from_u128(1),
            EntityBundle::new()
                .with(Pos { x: 10, y: 20 })
                .unwrap()
                .with(Vel { dx: 1, dy: 2 })
                .unwrap(),
        )
        .unwrap();

    let query = QueryBuilder::<(&Pos,)>::new()
        .with::<Vel>()
        .build(&store)
        .unwrap();

    assert_eq!(query.matched_count(), 1);
    let initial_gen = query.archetype_set_generation();

    // No changes to store -> refresh is a no-op
    query.with_matched_archetypes(&store, |_| Ok(())).unwrap();
    assert_eq!(query.archetype_set_generation(), initial_gen);
    assert_eq!(query.matched_count(), 1);

    // Now create an entity with Pos, Vel, AND TagActive -> creates a new Archetype!
    store
        .create_entity(
            EntityId::from_u128(2),
            EntityBundle::new()
                .with(Pos { x: 30, y: 40 })
                .unwrap()
                .with(Vel { dx: 3, dy: 4 })
                .unwrap()
                .with(TagActive)
                .unwrap(),
        )
        .unwrap();

    assert!(store.archetype_set_generation() > initial_gen);
    // Before refresh, query still holds the old cached count and gen
    assert_eq!(query.matched_count(), 1);
    assert_eq!(query.archetype_set_generation(), initial_gen);

    // Refresh query
    query.with_matched_archetypes(&store, |_| Ok(())).unwrap();
    assert_eq!(
        query.archetype_set_generation(),
        store.archetype_set_generation()
    );
    assert_eq!(query.matched_count(), 2);

    // Test mismatched store
    let other_store = Store::new();
    let err = query.with_matched_archetypes(&other_store, |_| Ok(())).unwrap_err();
    match err {
        StateError::StoreMismatch {
            expected_store,
            actual_store,
        } => {
            assert_eq!(expected_store, store.store_id());
            assert_eq!(actual_store, other_store.store_id());
        }
        other => panic!("expected StoreMismatch, got {other:?}"),
    }
}

#[test]
fn test_query_exclusive_lock_free_and_cached_slice_reuse() {
    let mut store = Store::new();
    let e1 = EntityId::from_u128(501);
    store
        .create_entity(
            e1,
            EntityBundle::new()
                .with(Pos { x: 10, y: 20 })
                .unwrap()
                .with(Vel { dx: 1, dy: 2 })
                .unwrap(),
        )
        .unwrap();

    let mut query = QueryBuilder::<(&mut Pos, &Vel)>::new()
        .build(&store)
        .unwrap();

    // 1. 缓存命中路径：测试连续切片直借的内存地址稳定性（无重复分配）
    let ptr1 = {
        query.with_matched_archetypes_mut(&store, |slice, _| {
            assert_eq!(slice.len(), 1);
            Ok(slice.as_ptr())
        }).unwrap()
    };
    let ptr2 = {
        query.with_matched_archetypes_mut(&store, |slice, _| {
            assert_eq!(slice.len(), 1);
            Ok(slice.as_ptr())
        }).unwrap()
    };
    // 缓存命中期间直接借用既有切片，指针绝对相等，证明无二次堆分配
    assert_eq!(ptr1, ptr2);

    // 2. 底层新增原型推进全局代际，触发缓存刷新扫描
    let e2 = EntityId::from_u128(502);
    store
        .create_entity(
            e2,
            EntityBundle::new()
                .with(Pos { x: 30, y: 40 })
                .unwrap()
                .with(Vel { dx: 3, dy: 4 })
                .unwrap()
                .with(TagActive)
                .unwrap(),
        )
        .unwrap();

    query.with_matched_archetypes_mut(&store, |slice, _| {
        assert_eq!(slice.len(), 2);
        Ok(())
    }).unwrap();
    assert_eq!(
        query.archetype_set_generation(),
        store.archetype_set_generation()
    );
}

#[test]
fn test_level2_matched_table_bindings_and_store_chunk_fast_path() {
    let mut store = Store::new();
    let e1 = EntityId::from_u128(701);
    let e2 = EntityId::from_u128(702);
    store
        .create_entity(
            e1,
            EntityBundle::new()
                .with(Pos { x: 10, y: 20 })
                .unwrap()
                .with(Vel { dx: 2, dy: 3 })
                .unwrap(),
        )
        .unwrap();

    let mut q_mut = Query::<(&mut Pos, &Vel)>::from_store(&store).unwrap();
    let q_ro = Query::<&Pos>::from_store(&store).unwrap();

    // 1. 验证 Level-2 预绑定列下标与 Archetype::component_types_slice 严格对应
    let bindings = q_mut.matched_bindings_arc();
    assert_eq!(bindings.len(), 1);
    let arch = &store.archetypes[bindings[0].archetype_id.0 as usize];
    assert_eq!(
        arch.component_types_slice()[bindings[0].flat_col_indices[0] as usize],
        ComponentTypeId::of::<Pos>()
    );
    assert_eq!(
        arch.component_types_slice()[bindings[0].flat_col_indices[1] as usize],
        ComponentTypeId::of::<Vel>()
    );

    // 2. 验证 Store::query_mut 不会破坏 flat_columns 缓存
    for (_id, (pos, vel)) in store.query_mut::<(&mut Pos, &Vel)>().unwrap() {
        pos.x += vel.dx;
        pos.y += vel.dy;
    }
    let arch_after_qmut = &store.archetypes[bindings[0].archetype_id.0 as usize];
    assert!(!arch_after_qmut.may_be_segmented_or_unaligned());
    assert_eq!(
        arch_after_qmut.flat_columns().len(),
        arch_after_qmut.component_types_slice().len()
    );

    // 3. 验证动态新增原型表后，Store::query_mut()?.chunks() 与 Query 缓存均正确工作
    store
        .insert((
            e2,
            (
                Pos { x: 100, y: 200 },
                Vel { dx: 5, dy: 6 },
                TagActive,
            ),
        ))
        .unwrap();

    for (_ents, (pos, vel)) in store.query_mut::<(&mut Pos, &Vel)>().unwrap().chunks() {
        for i in 0..pos.len() {
            pos[i].x += vel[i].dx;
            pos[i].y += vel[i].dy;
        }
    }
    q_mut.ensure_cache_current_mut(&store).unwrap();
    assert_eq!(q_mut.matched_bindings_arc().len(), 2);

    let mut sum_x = 0;
    for (_ents, pos) in store.query::<&Pos>().chunks() {
        for p in pos {
            sum_x += p.x;
        }
    }
    let _ = &q_ro;
    // e1: 10 + 2 + 2 = 14; e2: 100 + 5 = 105 -> sum_x = 119
    assert_eq!(sum_x, 119);
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct ChildOf;

#[test]
fn test_non_migrating_row_flags_and_structural_preservation() {
    let mut store = Store::new();
    let e1 = EntityId::from_u128(1001);
    let e2 = EntityId::from_u128(1002);
    let e3 = EntityId::from_u128(1003);

    store
        .insert((e1, (Pos { x: 1, y: 1 }, Vel { dx: 10, dy: 10 })))
        .unwrap();
    store
        .insert((e2, (Pos { x: 2, y: 2 }, Vel { dx: 20, dy: 20 })))
        .unwrap();
    store
        .insert((e3, (Pos { x: 3, y: 3 }, Vel { dx: 30, dy: 30 })))
        .unwrap();

    // Initially all flags are false
    assert!(!store.has_flag::<0>(e1));
    assert!(!store.has_flag::<0>(e2));

    // Set flag 0 on e1 and e3 (in-place without archetype migration!)
    let gen_before = store.archetype_set_generation();
    store.set_flag::<0>(e1, true).unwrap();
    store.set_flag::<0>(e3, true).unwrap();
    assert_eq!(store.archetype_set_generation(), gen_before);
    assert!(store.has_flag::<0>(e1));
    assert!(!store.has_flag::<0>(e2));
    assert!(store.has_flag::<0>(e3));

    // Query with_flag::<0> and without_flag::<0>
    let q_flagged = QueryBuilder::<&Pos>::new()
        .with_flag::<0>()
        .build(&store)
        .unwrap();
    let q_unflagged = QueryBuilder::<&Pos>::new()
        .without_flag::<0>()
        .build(&store)
        .unwrap();

    let mut flagged_xs = Vec::new();
    q_flagged
        .for_each(&store, |(_id, pos)| flagged_xs.push(pos.x))
        .unwrap();
    assert_eq!(flagged_xs, vec![1, 3]);

    let mut unflagged_xs = Vec::new();
    q_unflagged
        .for_each(&store, |(_id, pos)| unflagged_xs.push(pos.x))
        .unwrap();
    assert_eq!(unflagged_xs, vec![2]);

    // Structural migration: add TagActive to e1 -> e1 moves to a new archetype,
    // and e3 swap-removes into e1's old row 0! Both e1 and e3 must preserve their flags!
    store.add(e1, TagActive).unwrap();
    assert!(store.has_flag::<0>(e1));
    assert!(!store.has_flag::<0>(e2));
    assert!(store.has_flag::<0>(e3));

    let mut iter_flagged: Vec<i32> = store
        .query::<&Pos>()
        .with_flag::<0>()
        .iter()
        .map(|(_id, p)| p.x)
        .collect();
    iter_flagged.sort_unstable();
    assert_eq!(iter_flagged, vec![1, 3]);
}

#[test]
fn test_changed_since_chunk_epoch_dirty_tracking() {
    let mut store = Store::new();
    // Create 130 entities (spanning 3 64-row chunks: 0..64, 64..128, 128..130)
    for i in 0..130u128 {
        store
            .insert((
                EntityId::from_u128(2000 + i),
                (
                    Pos {
                        x: i as i32,
                        y: 0,
                    },
                    Vel { dx: 1, dy: 1 },
                ),
            ))
            .unwrap();
    }

    let baseline_epoch = store.epoch();

    // No writes after baseline_epoch -> changed_since returns 0 rows
    let q_changed = QueryBuilder::<&Pos>::new()
        .changed_since::<Pos>(baseline_epoch)
        .build(&store)
        .unwrap();
    let mut count = 0usize;
    q_changed
        .for_each(&store, |(_id, _pos)| count += 1)
        .unwrap();
    assert_eq!(count, 0);

    // Modify entity at index 70 (in chunk 1: rows 64..128)
    let e70 = EntityId::from_u128(2000 + 70);
    store
        .modify::<Pos, _>(e70, |p| p.y += 10)
        .unwrap();

    // Now only chunk 1 (64 rows: 64..128) should be visited by changed_since::<Pos>(baseline_epoch)!
    let mut visited_xs = Vec::new();
    q_changed
        .for_each(&store, |(_id, pos)| visited_xs.push(pos.x))
        .unwrap();
    assert_eq!(visited_xs.len(), 64);
    assert_eq!(visited_xs[0], 64);
    assert_eq!(visited_xs[63], 127);

    // Vel was NOT modified after baseline_epoch, so changed_since::<Vel> yields 0 rows
    let visited_vel: usize = store
        .query::<&Vel>()
        .changed_since::<Vel>(baseline_epoch)
        .iter()
        .count();
    assert_eq!(visited_vel, 0);
}

#[test]
fn test_pair_target_index_and_with_pair_query() {
    let mut store = Store::new();
    let parent_a = EntityId::from_u128(9001);
    let parent_b = EntityId::from_u128(9002);
    store.spawn_with_id(parent_a).unwrap();
    store.spawn_with_id(parent_b).unwrap();

    for i in 0..50u128 {
        let target = if i % 5 == 0 { parent_a } else { parent_b };
        store
            .insert((
                EntityId::from_u128(3000 + i),
                (
                    Pos {
                        x: i as i32,
                        y: 0,
                    },
                    crate::al::Pair::<ChildOf>::new(ChildOf, target),
                ),
            ))
            .unwrap();
    }

    // Query via persistent Query with_pair
    let q_children_a = QueryBuilder::<&Pos>::new()
        .with_pair::<ChildOf>(parent_a)
        .build(&store)
        .unwrap();
    let mut xs_a = Vec::new();
    q_children_a
        .for_each(&store, |(_id, pos)| xs_a.push(pos.x))
        .unwrap();
    assert_eq!(xs_a, vec![0, 5, 10, 15, 20, 25, 30, 35, 40, 45]);

    // Query via Store::sources_with_pair
    let sources_a = store.sources_with_pair::<ChildOf>(parent_a);
    assert_eq!(sources_a.len(), 10);

    // Mutate via Store::query_mut().with_pair
    for (_id, pos) in store
        .query_mut::<&mut Pos>()
        .unwrap()
        .with_pair::<ChildOf>(parent_a)
        .iter()
    {
        pos.y = 999;
    }
    let mut ys_a = Vec::new();
    q_children_a
        .for_each(&store, |(_id, pos)| ys_a.push(pos.y))
        .unwrap();
    assert!(ys_a.iter().all(|&y| y == 999));
}

#[test]
fn test_query_par_for_each_and_par_chunks_with_row_masks() {
    use std::sync::atomic::{AtomicI64, AtomicUsize, Ordering};

    let mut store = Store::new();
    // Spawn 10,000 entities spanning 3 4096-row subchunks (0..4096, 4096..8192, 8192..10000)
    for i in 0..10_000u128 {
        let e = EntityId::from_u128(10_000 + i);
        store
            .insert((
                e,
                (
                    Pos {
                        x: i as i32,
                        y: 1,
                    },
                    Vel { dx: 2, dy: 3 },
                ),
            ))
            .unwrap();
        if i % 10 == 0 {
            store.set_flag::<0>(e, true).unwrap();
        }
    }

    let q_flagged_mut = QueryBuilder::<(&mut Pos, &Vel)>::new()
        .with_flag::<0>()
        .build(&store)
        .unwrap();
    let baseline_epoch = store.epoch();

    let updated_count = AtomicUsize::new(0);
    q_flagged_mut
        .par_for_each_mut(&mut store, |(_id, (pos, vel))| {
            pos.x += vel.dx;
            pos.y += vel.dy;
            updated_count.fetch_add(1, Ordering::Relaxed);
        })
        .unwrap();
    assert_eq!(updated_count.load(Ordering::Relaxed), 1000);

    // Parallel read-only verification with flag filter
    let q_flagged_ro = QueryBuilder::<&Pos>::new()
        .with_flag::<0>()
        .build(&store)
        .unwrap();
    let sum_y = AtomicI64::new(0);
    q_flagged_ro
        .par_for_each(&store, |(_id, pos)| {
            assert_eq!(pos.y, 4);
            sum_y.fetch_add(pos.y as i64, Ordering::Relaxed);
        })
        .unwrap();
    assert_eq!(sum_y.load(Ordering::Relaxed), 4000);

    // Parallel chunk mutation across all 10,000 rows (dynamically planned by SubchunkPlanner:
    // for 16-byte row width (8B Pos + 8B Vel), 32KB / 16B = 2048 rows -> 5 subchunks)
    let q_all_chunk_mut = QueryBuilder::<(&mut Pos, &Vel)>::new()
        .build(&store)
        .unwrap();
    let subchunk_calls = AtomicUsize::new(0);
    let total_chunk_rows = AtomicUsize::new(0);
    q_all_chunk_mut
        .par_for_each_chunk_mut(&mut store, |(ents, (pos_slice, vel_slice))| {
            assert_eq!(ents.len(), pos_slice.len());
            assert_eq!(ents.len(), vel_slice.len());
            subchunk_calls.fetch_add(1, Ordering::Relaxed);
            total_chunk_rows.fetch_add(ents.len(), Ordering::Relaxed);
            for (p, v) in pos_slice.iter_mut().zip(vel_slice.iter()) {
                p.y += v.dy;
            }
        })
        .unwrap();
    let expected_subchunks = crate::storage::primitives::SubchunkPlanner::default()
        .plan_subchunks(std::mem::size_of::<Pos>() + std::mem::size_of::<Vel>(), 10_000)
        .len();
    assert_eq!(subchunk_calls.load(Ordering::Relaxed), expected_subchunks);
    assert_eq!(total_chunk_rows.load(Ordering::Relaxed), 10_000);

    // Parallel chunk read-only verification with ChangedSince
    let q_changed_ro = QueryBuilder::<&Pos>::new()
        .changed_since::<Pos>(baseline_epoch)
        .build(&store)
        .unwrap();
    let total_changed_rows = AtomicUsize::new(0);
    q_changed_ro
        .par_for_each_chunk(&store, |(ents, pos_slice)| {
            assert_eq!(ents.len(), pos_slice.len());
            total_changed_rows.fetch_add(ents.len(), Ordering::Relaxed);
        })
        .unwrap();
    assert_eq!(total_changed_rows.load(Ordering::Relaxed), 10_000);
}

#[test]
fn test_query_conflicts_with_and_al_par_chunk_epoch_tracking() {
    use crate::al::traits::{All, StateSpace};

    let mut store = Store::new();
    for i in 0..200u128 {
        store
            .insert((
                EntityId::from_u128(50_000 + i),
                (
                    Pos { x: i as i32, y: 0 },
                    Vel { dx: 1, dy: 2 },
                ),
            ))
            .unwrap();
    }

    let q_read_pos = QueryBuilder::<&Pos>::new().build(&store).unwrap();
    let q_read_pos_vel = QueryBuilder::<(&Pos, &Vel)>::new().build(&store).unwrap();
    let q_write_pos = QueryBuilder::<&mut Pos>::new().build(&store).unwrap();
    let q_write_vel = QueryBuilder::<&mut Vel>::new().build(&store).unwrap();
    let q_write_vel_filter_pos = QueryBuilder::<&mut Vel>::new()
        .changed_since::<Pos>(0)
        .build(&store)
        .unwrap();

    // Read-Read on same component is compatible
    assert!(q_read_pos.is_compatible_with(&q_read_pos_vel));
    // Write-Write on disjoint components is compatible
    assert!(q_write_pos.is_compatible_with(&q_write_vel));
    // Read-Write on Pos conflicts
    assert!(q_read_pos.conflicts_with(&q_write_pos));
    // Write(Pos) conflicts with Write(Vel) + FilterRead(Pos)
    assert!(q_write_pos.conflicts_with(&q_write_vel_filter_pos));

    // Verify AL StateSpace::par_for_each_chunk_mut2 stamps 64-row chunk epochs for ChangedSince
    let epoch_before = store.epoch();
    <Store as StateSpace>::par_for_each_chunk_mut2::<Pos, Vel, All>(
        &mut store,
        |_ents, pos_slice, vel_slice| {
            for (p, v) in pos_slice.iter_mut().zip(vel_slice.iter()) {
                p.x += v.dx;
                p.y += v.dy;
            }
        },
    );
    assert!(store.epoch() > epoch_before);

    let changed_pos_count = store
        .query::<&Pos>()
        .changed_since::<Pos>(epoch_before)
        .into_iter()
        .count();
    assert_eq!(changed_pos_count, 200);
}

#[test]
fn test_query_optional_components_ro_and_mut() {
    let mut store = Store::new();
    // Entity 1: has Pos and Vel
    store
        .create_entity(
            EntityId::from_u128(1),
            EntityBundle::new()
                .with(Pos { x: 10, y: 20 })
                .unwrap()
                .with(Vel { dx: 1, dy: 2 })
                .unwrap(),
        )
        .unwrap();

    // Entity 2: has only Pos
    store
        .create_entity(
            EntityId::from_u128(2),
            EntityBundle::new()
                .with(Pos { x: 100, y: 200 })
                .unwrap(),
        )
        .unwrap();

    // Test Read-only Query with Option<&Vel>
    let q_ro = QueryBuilder::<(&Pos, Option<&Vel>)>::new().build(&store).unwrap();
    let mut results = Vec::new();
    q_ro.for_each(&store, |(e, (p, v))| {
        results.push((e.to_u128(), p.x, v.map(|vel| vel.dx)));
    }).unwrap();
    results.sort_by_key(|r| r.0);
    assert_eq!(results, vec![(1, 10, Some(1)), (2, 100, None)]);

    // Test Read-only Chunk Query with Option<&Vel>
    let mut chunk_results = Vec::new();
    q_ro.for_each_chunk(&store, |(ents, (pos_slice, vel_slice))| {
        for (i, &e) in ents.iter().enumerate() {
            let dx = vel_slice.as_ref().map(|v| v[i].dx);
            chunk_results.push((e.to_u128(), pos_slice[i].x, dx));
        }
    }).unwrap();
    chunk_results.sort_by_key(|r| r.0);
    assert_eq!(chunk_results, vec![(1, 10, Some(1)), (2, 100, None)]);

    // Test Mutable Query with Option<&mut Vel>
    let q_mut = QueryBuilder::<(&mut Pos, Option<&mut Vel>)>::new().build(&store).unwrap();
    q_mut.for_each_mut(&mut store, |(_e, (p, v))| {
        p.x += 5;
        if let Some(vel) = v {
            vel.dx += 10;
        }
    }).unwrap();

    // Verify mutations
    let pos1 = store.get::<Pos>(EntityId::from_u128(1)).unwrap();
    let vel1 = store.get::<Vel>(EntityId::from_u128(1)).unwrap();
    assert_eq!(pos1.x, 15);
    assert_eq!(vel1.dx, 11);

    let pos2 = store.get::<Pos>(EntityId::from_u128(2)).unwrap();
    assert_eq!(pos2.x, 105);

    // Test Mutable Chunk Query with Option<&mut Vel>
    q_mut.for_each_chunk_mut(&mut store, |(_ents, (pos_slice, vel_slice))| {
        for p in pos_slice.iter_mut() {
            p.x += 1;
        }
        if let Some(vels) = vel_slice {
            for v in vels.iter_mut() {
                v.dx += 1;
            }
        }
    }).unwrap();

    let pos1_after = store.get::<Pos>(EntityId::from_u128(1)).unwrap();
    let vel1_after = store.get::<Vel>(EntityId::from_u128(1)).unwrap();
    assert_eq!(pos1_after.x, 16);
    assert_eq!(vel1_after.dx, 12);

    let pos2_after = store.get::<Pos>(EntityId::from_u128(2)).unwrap();
    assert_eq!(pos2_after.x, 106);
}

#[test]
fn test_optional_query_access_and_conflict() {
    let store = Store::new();
    let q_opt = QueryBuilder::<(&Pos, Option<&mut Vel>)>::new().build(&store).unwrap();
    let q_read_vel = QueryBuilder::<&Vel>::new().build(&store).unwrap();
    let q_write_vel = QueryBuilder::<&mut Vel>::new().build(&store).unwrap();
    let q_read_pos = QueryBuilder::<&Pos>::new().build(&store).unwrap();

    // q_opt may write Vel, so it conflicts with q_read_vel and q_write_vel
    assert!(q_opt.conflicts_with(&q_read_vel));
    assert!(q_opt.conflicts_with(&q_write_vel));
    // q_opt only reads Pos, so it is compatible with q_read_pos
    assert!(q_opt.is_compatible_with(&q_read_pos));

    let desc = q_opt.access_descriptor();
    assert!(desc.with_components.contains(&ComponentTypeId::of::<Pos>()));
    // Vel is optional, so it must not be in with_components
    assert!(!desc.with_components.contains(&ComponentTypeId::of::<Vel>()));
    assert!(desc.writes.contains(&ComponentTypeId::of::<Vel>()));
}



