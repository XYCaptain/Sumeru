use serde::{Deserialize, Serialize};

use crate::al::{EntityHandle, StateSpace, Tag, With, Without};
use crate::{ComponentField, Entity, EntityBundle, StateError, Store};

#[derive(
    Debug,
    Clone,
    Copy,
    PartialEq,
    Serialize,
    Deserialize,
    rkyv::Archive,
    rkyv::Serialize,
    rkyv::Deserialize,
)]
struct Pos {
    x: f32,
    y: f32,
}

#[derive(
    Debug,
    Clone,
    Copy,
    PartialEq,
    Serialize,
    Deserialize,
    rkyv::Archive,
    rkyv::Serialize,
    rkyv::Deserialize,
)]
struct Vel {
    dx: f32,
    dy: f32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
struct ChildOf;

struct PosXField;

impl ComponentField<Pos> for PosXField {
    type Value = f32;
    fn extract(component: &Pos) -> &Self::Value {
        &component.x
    }
    fn field_name() -> &'static str {
        "x"
    }
}

#[test]
fn al_state_space_entity_facade_and_commands_work_on_store() {
    let mut store = Store::new();
    let e1 = store.spawn_auto().unwrap();
    let e2 = store.spawn_auto().unwrap();
    assert!(!e1.is_null());
    assert_eq!(store.entity_count(), 2);

    // Immediate EntityMut facade chaining & get/try_get/get_mut
    {
        let mut em = store.entity_mut(e1).unwrap();
        em.set(Pos { x: 1.0, y: 2.0 }).unwrap();
        em.add(Vel { dx: 0.5, dy: 1.5 }).unwrap();
        em.set(Tag::<0>).unwrap();
        assert!(em.has::<Pos>());
        assert_eq!(em.get::<Pos>().unwrap().x, 1.0);
        assert_eq!(em.try_get::<Pos>().unwrap().y, 2.0);
        em.get_mut::<Pos>().unwrap().x = 1.25;
        assert_eq!(em.get::<Pos>().unwrap().x, 1.25);
        em.set(Pos { x: 1.0, y: 2.0 }).unwrap();
    }

    // Immediate EntityRef facade (get / try_get / get_tuple)
    {
        let er = store.entity(e1).unwrap();
        assert!(er.has::<Pos>());
        assert_eq!(er.get::<Vel>().unwrap().dy, 1.5);
        let (p, v) = er.get_tuple::<(&Pos, &Vel)>().unwrap();
        assert_eq!((p.x, v.dx), (1.0, 0.5));
    }

    // Formal two-stage diagnostics (try_entity / try_get / try_get_mut / try_has / try_pair_target)
    {
        let dead = <Store as StateSpace>::Entity::from_parts(9999, 1);
        assert!(matches!(
            store.try_entity(dead),
            Err(StateError::EntityNotFound(_))
        ));
        assert!(matches!(
            store.try_get::<Pos>(dead),
            Err(StateError::EntityNotFound(_))
        ));
        assert!(matches!(
            store.try_get::<Pos>(e2),
            Err(StateError::ComponentNotFoundInEntity { .. })
        ));
        assert!(matches!(
            store.try_has::<Pos>(dead),
            Err(StateError::EntityNotFound(_))
        ));
        assert!(store.try_has::<Pos>(e1).unwrap());
        assert!(!store.try_has::<Pos>(e2).unwrap());
    }

    // Two-phase CommandBuffer via GAT
    {
        let mut cmd = StateSpace::commands(&store);
        let e3 = <Store as StateSpace>::Entity::from_parts(99, 1);
        cmd.create_empty(e3);
        cmd.entity(e3).set(Pos { x: 10.0, y: 20.0 });
        cmd.entity(e2)
            .set(Pos { x: 3.0, y: 4.0 })
            .set(Pos { x: 7.0, y: 8.0 });
        cmd.apply_discard(&mut store).unwrap();
        assert_eq!(store.entity_count(), 3);
        assert_eq!(store.get::<Pos>(e3).unwrap().x, 10.0);
        assert_eq!(store.get::<Pos>(e2).unwrap().x, 7.0);
    }

    // Filter + Chunk iteration (unified With<T, F = All> / Without<T, F = All>)
    store.for_each_chunk_mut2::<Pos, Vel, With<Tag<0>>>(|_entities, pos, vel| {
        for i in 0..pos.len() {
            pos[i].x += vel[i].dx;
            pos[i].y += vel[i].dy;
        }
    });
    assert_eq!(store.get::<Pos>(e1).unwrap().x, 1.5);

    let mut without_tag_count = 0usize;
    store.for_each_chunk1::<Pos, Without<Tag<0>>>(|entities, _pos| {
        without_tag_count += entities.len();
    });
    assert_eq!(without_tag_count, 2);

    // First-class Pair<R, Entity>
    store.set_pair(e2, ChildOf, e1).unwrap();
    assert_eq!(store.pair_target::<ChildOf>(e2), Some(e1));
    assert_eq!(store.try_pair_target::<ChildOf>(e2).unwrap(), e1);
    assert_eq!(store.sources_with_pair::<ChildOf>(e1), vec![e2]);

    // SelectBuilder / QueryMutBuilder orthogonal filter + with_pair
    let matched: Vec<_> = store
        .query::<&Pos>()
        .with_pair::<ChildOf>(e1)
        .filter::<Pos>(|p| p.x > 5.0)
        .filter_eq::<Pos, PosXField>(7.0)
        .iter()
        .map(|(id, p)| (id, p.x))
        .collect();
    assert_eq!(matched, vec![(e2, 7.0)]);

    for (_, p) in store
        .query_mut::<&mut Pos>()
        .unwrap()
        .with_pair::<ChildOf>(e1)
        .filter_eq::<Pos, PosXField>(7.0)
        .iter_mut()
    {
        p.x = 14.0;
    }
    assert_eq!(store.get::<Pos>(e2).unwrap().x, 14.0);
}

#[test]
fn al_batch_and_commands_work_on_store() {
    let mut store = Store::with_capacity(16);
    store.register_component::<Pos>().unwrap();
    store.register_component::<Vel>().unwrap();

    let e1 = <Store as StateSpace>::Entity::from_parts(1, 1);
    let e2 = <Store as StateSpace>::Entity::from_parts(2, 1);
    let e3 = <Store as StateSpace>::Entity::from_parts(3, 1);

    // Entity fluent builder without intermediate .unwrap() + insert_batch / despawn_batch
    let built = Entity::new(e1)
        .with(Pos { x: 10.0, y: 20.0 })
        .with(Vel { dx: 1.0, dy: 2.0 })
        .with_tag::<1>();
    store.insert(built).unwrap();

    store
        .insert_batch([
            (e2, (Pos { x: 30.0, y: 40.0 }, Vel { dx: 3.0, dy: 4.0 })),
            (e3, (Pos { x: 50.0, y: 60.0 }, Vel { dx: 5.0, dy: 6.0 })),
        ])
        .unwrap();
    assert_eq!(store.populated_archetype_count(), 2);

    // Reorder entities in archetype
    let moved = store.reorder_entities(&[e3, e2]).unwrap();
    assert!(moved > 0);

    let mut cmd = store.commands();
    cmd.entity(e1).replace(Pos { x: 99.0, y: 99.0 });
    let receipt = cmd.apply(&mut store).unwrap();
    assert_eq!(receipt.updated_entities, vec![e1]);
    assert_eq!(store.get::<Pos>(e1), Some(&Pos { x: 99.0, y: 99.0 }));

    let removed_count = store.despawn_batch([e2, e3]).unwrap();
    assert_eq!(removed_count, 2);
    assert_eq!(store.len(), 1);
}

#[test]
fn al_spawn_bundle_polymorphic_and_batch_update_work() {
    assert_eq!(
        <Store as StateSpace>::BACKEND_NAME,
        <Store as StateSpace>::ENGINE_NAME
    );
    assert_eq!(<Store as StateSpace>::BACKEND_NAME, "StateFabric (Store)");

    let mut store = Store::new();

    // 1. store.spawn(())
    let e_empty = store.spawn(()).unwrap();
    assert!(!e_empty.is_null());
    assert!(store.contains_entity(e_empty));

    // 2. store.spawn((Pos, Vel))
    let e_tuple = store
        .spawn((Pos { x: 1.0, y: 2.0 }, Vel { dx: 3.0, dy: 4.0 }))
        .unwrap();
    assert_eq!(store.get::<Pos>(e_tuple), Some(&Pos { x: 1.0, y: 2.0 }));
    assert_eq!(store.get::<Vel>(e_tuple), Some(&Vel { dx: 3.0, dy: 4.0 }));

    // 3. store.spawn(EntityBundle)
    let bundle = EntityBundle::new()
        .with(Pos { x: 5.0, y: 6.0 })
        .unwrap()
        .with(Vel { dx: 7.0, dy: 8.0 })
        .unwrap();
    let e_bundle = store.spawn(bundle).unwrap();
    assert_eq!(store.get::<Pos>(e_bundle), Some(&Pos { x: 5.0, y: 6.0 }));
    assert_eq!(store.get::<Vel>(e_bundle), Some(&Vel { dx: 7.0, dy: 8.0 }));

    // 4. store.spawn(Entity::new(id).with(Pos))
    let explicit_id1 = <Store as StateSpace>::Entity::from_parts(500, 1);
    let e_builder = store
        .spawn(Entity::new(explicit_id1).with(Pos { x: 9.0, y: 10.0 }))
        .unwrap();
    assert_eq!(e_builder, explicit_id1);
    assert_eq!(store.get::<Pos>(explicit_id1), Some(&Pos { x: 9.0, y: 10.0 }));

    // 5. store.spawn(id)
    let explicit_id2 = <Store as StateSpace>::Entity::from_parts(501, 1);
    let e_id = store.spawn(explicit_id2).unwrap();
    assert_eq!(e_id, explicit_id2);
    assert!(store.contains_entity(explicit_id2));

    // 6. Store::update_batch with multi-component tuples (EntityId, Pos, Vel)
    let receipt = store
        .update_batch([
            (e_tuple, Pos { x: 11.0, y: 22.0 }, Vel { dx: 33.0, dy: 44.0 }),
            (e_bundle, Pos { x: 55.0, y: 66.0 }, Vel { dx: 77.0, dy: 88.0 }),
        ])
        .unwrap();
    assert_eq!(receipt.len(), 2);
    assert_eq!(store.get::<Pos>(e_tuple), Some(&Pos { x: 11.0, y: 22.0 }));
    assert_eq!(store.get::<Vel>(e_tuple), Some(&Vel { dx: 33.0, dy: 44.0 }));
    assert_eq!(store.get::<Pos>(e_bundle), Some(&Pos { x: 55.0, y: 66.0 }));
    assert_eq!(store.get::<Vel>(e_bundle), Some(&Vel { dx: 77.0, dy: 88.0 }));

    // 7. Store::update_batch_discard with multi-component tuples (EntityId, Pos, Vel)
    store
        .update_batch_discard([
            (e_tuple, Pos { x: 100.0, y: 200.0 }, Vel { dx: 300.0, dy: 400.0 }),
            (e_bundle, Pos { x: 500.0, y: 600.0 }, Vel { dx: 700.0, dy: 800.0 }),
        ])
        .unwrap();
    assert_eq!(store.get::<Pos>(e_tuple), Some(&Pos { x: 100.0, y: 200.0 }));
    assert_eq!(store.get::<Vel>(e_tuple), Some(&Vel { dx: 300.0, dy: 400.0 }));
    assert_eq!(store.get::<Pos>(e_bundle), Some(&Pos { x: 500.0, y: 600.0 }));
    assert_eq!(store.get::<Vel>(e_bundle), Some(&Vel { dx: 700.0, dy: 800.0 }));

    // 8. CommandBuffer::insert_batch & EntityCommands pair/tag
    let batch_id1 = <Store as StateSpace>::Entity::from_parts(600, 1);
    let batch_id2 = <Store as StateSpace>::Entity::from_parts(601, 1);
    let mut cmd = store.commands();
    cmd.insert_batch([
        (batch_id1, (Pos { x: 1.0, y: 1.0 },)),
        (batch_id2, (Pos { x: 2.0, y: 2.0 },)),
    ]);
    cmd.entity(e_tuple).set_pair(ChildOf, e_bundle).add_tag::<4>();
    cmd.apply_discard(&mut store).unwrap();
    assert!(store.contains_entity(batch_id1));
    assert!(store.contains_entity(batch_id2));
    assert_eq!(store.pair_target::<ChildOf>(e_tuple), Some(e_bundle));
    assert!(store.has_tag::<4>(e_tuple));

    // 9. Store::reorder_entities & StateSpace::for_each1 / for_each_mut1
    let moved = store.reorder_entities(&[batch_id2, batch_id1]).unwrap();
    assert!(moved > 0);

    StateSpace::for_each_mut1::<Pos, crate::With<crate::al::Tag<4>>>(&mut store, |_id, p| {
        p.x += 9.0;
    });
    let mut tagged_x = Vec::new();
    StateSpace::for_each1::<Pos, crate::With<crate::al::Tag<4>>>(&store, |_id, p| {
        tagged_x.push(p.x);
    });
    assert_eq!(tagged_x, vec![109.0]);
}

#[test]
fn al_singleton_and_entity_ref_mut_symmetry_work() {
    let mut store = Store::new();

    // Native O(1) singleton methods
    assert!(!store.has_singleton::<Pos>());
    assert_eq!(store.singleton::<Pos>(), None);
    assert!(store.try_singleton::<Pos>().is_err());
    assert_eq!(store.singleton_mut::<Pos>(), None);
    assert!(store.try_singleton_mut::<Pos>().is_err());

    let prev = store.set_singleton(Pos { x: 42.0, y: 84.0 }).unwrap();
    assert_eq!(prev, None);
    assert!(store.has_singleton::<Pos>());
    assert_eq!(store.singleton::<Pos>(), Some(&Pos { x: 42.0, y: 84.0 }));
    assert_eq!(store.try_singleton::<Pos>().unwrap(), &Pos { x: 42.0, y: 84.0 });

    store.singleton_mut::<Pos>().unwrap().x = 50.0;
    assert_eq!(store.singleton::<Pos>().unwrap().x, 50.0);

    store.try_singleton_mut::<Pos>().unwrap().y = 90.0;
    assert_eq!(store.try_singleton::<Pos>().unwrap(), &Pos { x: 50.0, y: 90.0 });

    let replaced = store.set_singleton(Pos { x: 100.0, y: 200.0 }).unwrap();
    assert_eq!(replaced, Some(Pos { x: 50.0, y: 90.0 }));

    // Verify store.query::<&Pos>() on normal entities does NOT match Singleton<Pos>
    let normal = store.spawn((Pos { x: 1.0, y: 2.0 },)).unwrap();
    let queried: Vec<_> = store.query::<&Pos>().iter().map(|(id, p)| (id, *p)).collect();
    assert_eq!(queried, vec![(normal, Pos { x: 1.0, y: 2.0 })]);

    let removed = store.remove_singleton::<Pos>().unwrap();
    assert_eq!(removed, Some(Pos { x: 100.0, y: 200.0 }));
    assert!(!store.has_singleton::<Pos>());
    assert_eq!(store.remove_singleton::<Pos>().unwrap(), None);

    // EntityRef and EntityMut symmetry
    let parent = store.spawn((Pos { x: 0.0, y: 0.0 },)).unwrap();
    let child = store
        .spawn((Pos { x: 3.0, y: 4.0 }, Vel { dx: 5.0, dy: 6.0 }))
        .unwrap();
    let dead_target = store.spawn(()).unwrap();
    store.despawn(dead_target).unwrap();

    {
        let mut em = store.entity_mut(child).unwrap();
        // EntityMut::set_pair rejecting despawned target
        assert!(matches!(
            em.set_pair(ChildOf, dead_target),
            Err(StateError::EntityNotFound(id)) if id == dead_target
        ));
        assert_eq!(em.pair_target::<ChildOf>(), None);

        // EntityMut::set_pair and pair_target on alive target
        assert_eq!(em.set_pair(ChildOf, parent).unwrap(), None);
        assert_eq!(em.pair_target::<ChildOf>(), Some(parent));

        // EntityMut::try_get_mut
        em.try_get_mut::<Pos>().unwrap().x = 30.0;
        assert_eq!(em.get::<Pos>().unwrap().x, 30.0);

        // EntityMut::get_tuple and EntityMut::extract
        let (p, v) = em.get_tuple::<(&Pos, &Vel)>().unwrap();
        assert_eq!((p.x, v.dx), (30.0, 5.0));
        let (p2, v2) = em.extract::<(&Pos, &Vel)>().unwrap();
        assert_eq!((p2.y, v2.dy), (4.0, 6.0));
    }

    // EntityRef::pair_target before and after target despawn
    assert_eq!(
        store.entity(child).unwrap().pair_target::<ChildOf>(),
        Some(parent)
    );
    store.despawn(parent).unwrap();
    assert_eq!(store.entity(child).unwrap().pair_target::<ChildOf>(), None);
    assert_eq!(
        store.entity_mut(child).unwrap().pair_target::<ChildOf>(),
        None
    );
}

#[test]
fn al_query_builder_plan_and_tag_pair_combinators_work() {
    let mut store = Store::new();
    let parent = store.spawn(()).unwrap();

    // e1: matches (has Pos, Vel, Tag<0>; no Tag<1>; no Pair<ChildOf>)
    let e1 = store
        .spawn((Pos { x: 1.0, y: 2.0 }, Vel { dx: 10.0, dy: 20.0 }, Tag::<0>))
        .unwrap();
    // e2: excluded by without_tag::<1>()
    let _e2 = store
        .spawn((
            Pos { x: 3.0, y: 4.0 },
            Vel { dx: 30.0, dy: 40.0 },
            Tag::<0>,
            Tag::<1>,
        ))
        .unwrap();
    // e3: excluded by without_pair::<ChildOf>()
    let e3 = store
        .spawn((Pos { x: 5.0, y: 6.0 }, Vel { dx: 50.0, dy: 60.0 }, Tag::<0>))
        .unwrap();
    store.set_pair(e3, ChildOf, parent).unwrap();
    // e4: excluded by with_tag::<0>()
    let _e4 = store
        .spawn((Pos { x: 7.0, y: 8.0 }, Vel { dx: 70.0, dy: 80.0 }))
        .unwrap();

    // 1. SelectBuilder: with_tag, without_tag, without_pair
    let selected: Vec<_> = store
        .query::<(&Pos, &Vel)>()
        .with_tag::<0>()
        .without_tag::<1>()
        .without_pair::<ChildOf>()
        .iter()
        .map(|(id, (p, v))| (id, p.x, v.dx))
        .collect();
    assert_eq!(selected, vec![(e1, 1.0, 10.0)]);

    // 2. QueryMutBuilder: with_tag, without_tag, without_pair
    for (id, (p, v)) in store
        .query_mut::<(&mut Pos, &Vel)>()
        .unwrap()
        .with_tag::<0>()
        .without_tag::<1>()
        .without_pair::<ChildOf>()
        .iter_mut()
    {
        assert_eq!(id, e1);
        p.x += v.dx;
    }
    assert_eq!(store.get::<Pos>(e1).unwrap().x, 11.0);

    // 3. QueryBuilder & Query<Q, F>: plan, for_each, for_each_chunk, for_each_mut, for_each_chunk_mut
    let query_ro = store
        .query_builder::<(&Pos, &Vel)>()
        .with_tag::<0>()
        .without_tag::<1>()
        .without_pair::<ChildOf>()
        .plan(&store)
        .unwrap();

    let mut visited_ro = Vec::new();
    query_ro
        .for_each(&store, |(id, (p, v))| {
            visited_ro.push((id, p.x, v.dx));
        })
        .unwrap();
    assert_eq!(visited_ro, vec![(e1, 11.0, 10.0)]);

    let mut chunk_ro_count = 0usize;
    query_ro
        .for_each_chunk(&store, |(ids, (pos, vel))| {
            assert_eq!(ids, &[e1]);
            assert_eq!(pos.len(), 1);
            assert_eq!(vel.len(), 1);
            chunk_ro_count += ids.len();
        })
        .unwrap();
    assert_eq!(chunk_ro_count, 1);

    let query_rw = store
        .query_builder::<(&mut Pos, &Vel)>()
        .with_tag::<0>()
        .without_tag::<1>()
        .without_pair::<ChildOf>()
        .plan(&store)
        .unwrap();

    query_rw
        .for_each_mut(&mut store, |(id, (p, v))| {
            assert_eq!(id, e1);
            p.x += v.dx;
        })
        .unwrap();
    assert_eq!(store.get::<Pos>(e1).unwrap().x, 21.0);

    query_rw
        .for_each_chunk_mut(&mut store, |(ids, (pos, vel))| {
            assert_eq!(ids, &[e1]);
            for (p, v) in pos.iter_mut().zip(vel.iter()) {
                p.y += v.dy;
            }
        })
        .unwrap();
    assert_eq!(store.get::<Pos>(e1).unwrap().y, 22.0);
}

