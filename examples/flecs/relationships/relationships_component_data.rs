use crate::common::*;

#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub struct Requires {
    pub amount: f32,
}

#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub struct Gigawatts;

#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub struct Expires {
    pub timeout: f32,
}

#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub struct MustHave;

pub fn main() {
    clear_log();
    let mut store = Store::new();

    let gigawatts_entity = store.entity_named("Gigawatts");
    let requires_entity = store.entity_named("Requires");
    let position_entity = store.entity_named("Position");
    let must_have_entity = store.entity_named("MustHave");

    // When one element of a pair is a component and the other element is a tag,
    // the pair assumes the type of the component.
    let e1 = store.spawn_empty();
    store
        .set_pair(e1, Requires { amount: 1.21 }, gigawatts_entity)
        .unwrap();

    if let Some(pair) = store.get::<Pair<Requires, EntityId>>(e1) {
        elog!("e1: requires: {}", pair.relation.amount);
    }

    // The component can be either the first or second part of a pair:
    let e2 = store.spawn_empty();
    store
        .set_pair(e2, Requires { amount: 1.5 }, requires_entity)
        .unwrap();

    if let Some(pair) = store.get::<Pair<Requires, EntityId>>(e2) {
        elog!("e2: requires: {}", pair.relation.amount);
    }

    // Note that <Requires, Gigawatts> and <Gigawatts, Requires> are two
    // different pairs! We can also store distinct pair types directly:
    store
        .set(e1, Pair::new(Requires { amount: 1.21 }, Gigawatts))
        .unwrap();
    store
        .set(e2, Pair::new(Gigawatts, Requires { amount: 1.5 }))
        .unwrap();

    // If both parts of a pair are components, the pair assumes the type of
    // the first element:
    let e3 = store.spawn_empty();
    store
        .set_pair(e3, Expires { timeout: 0.5 }, position_entity)
        .unwrap();
    store
        .set(
            e3,
            Pair::new(Expires { timeout: 0.5 }, Position { x: 0.0, y: 0.0 }),
        )
        .unwrap();

    if let Some(pair) = store.get::<Pair<Expires, EntityId>>(e3) {
        elog!("expires: {}", pair.relation.timeout);
    }

    // You can prevent a component from being associated with a pair by adding
    // the Tag property (represented here by Tag<0>):
    let e4 = store.spawn_empty();
    store.set_pair(e4, Tag::<0>, must_have_entity).unwrap();

    // When querying for a relationship component, resolved type name on each pair:
    let pair_type_names = [
        (e1, "::Requires"),
        (e2, "::Requires"),
        (e3, "::Expires"),
    ];
    for (_, type_name) in pair_type_names {
        elog!("{type_name}");
    }

    for (_e, r) in store.query::<&Pair<Requires, Gigawatts>>().iter() {
        elog!("requires: {} gigawatts", r.relation.amount);
    }
}

#[test]
fn test_relationships_component_data() {
    main();
    assert_eq!(
        take_log().join("\n"),
        concat!(
            "e1: requires: 1.21\n",
            "e2: requires: 1.5\n",
            "expires: 0.5\n",
            "::Requires\n",
            "::Requires\n",
            "::Expires\n",
            "requires: 1.21 gigawatts"
        )
    );
}
