use crate::common::*;
use crate::elog;

// Singleton component
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
pub struct Gravity {
    pub value: f32,
}

pub fn main() {
    clear_log();
    let mut store = Store::new();

    // Set singleton component
    store.set_singleton(Gravity { value: 9.81 }).unwrap();

    // Set Velocity on entities
    let e1 = store.entity_named("e1");
    store.set(e1, Velocity { x: 0.0, y: 0.0 });

    let e2 = store.entity_named("e2");
    store.set(e2, Velocity { x: 0.0, y: 1.0 });

    let e3 = store.entity_named("e3");
    store.set(e3, Velocity { x: 0.0, y: 2.0 });

    // Query Velocity together with the Gravity singleton
    let g = *store.singleton::<Gravity>().unwrap();

    for (_e, (name, vel)) in store
        .query_mut::<(&EntityName, &mut Velocity)>()
        .unwrap()
        .iter()
    {
        vel.y += g.value;
        elog!("Entity ::{} has {:?}", name.0, vel);
    }
}

#[test]
fn test_query_singleton() {
    main();
    assert_eq!(
        take_log(),
        vec![
            "Entity ::e1 has Velocity { x: 0.0, y: 9.81 }",
            "Entity ::e2 has Velocity { x: 0.0, y: 10.81 }",
            "Entity ::e3 has Velocity { x: 0.0, y: 11.81 }",
        ]
    );
}
