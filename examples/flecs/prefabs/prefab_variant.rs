use crate::common::*;
use super::{add_is_a, get_inherited};

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Attack {
    pub value: f32,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Defence {
    pub value: f32,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct FreightCapacity {
    pub value: f32,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct ImpulseSpeed {
    pub value: f32,
}

pub fn main() {
    clear_log();
    let mut store = Store::new();

    // Create a base prefab for SpaceShips.
    let spaceship = store.prefab_named("SpaceShip");
    store.set(spaceship, ImpulseSpeed { value: 50.0 }).unwrap();
    store.set(spaceship, Defence { value: 25.0 }).unwrap();

    // Create a Freighter variant which inherits from SpaceShip
    let freighter = store.prefab_named("Freighter");
    add_is_a(&mut store, freighter, spaceship);
    store.set(freighter, FreightCapacity { value: 100.0 }).unwrap();
    store.set(freighter, Defence { value: 50.0 }).unwrap();

    // Create a MammothFreighter variant which inherits from Freighter
    let mammoth_freighter = store.prefab_named("MammothFreighter");
    add_is_a(&mut store, mammoth_freighter, freighter);
    store
        .set(mammoth_freighter, FreightCapacity { value: 500.0 })
        .unwrap();

    // Create a Frigate variant which inherits from SpaceShip
    let frigate = store.prefab_named("Frigate");
    add_is_a(&mut store, frigate, spaceship);
    store.set(frigate, Attack { value: 100.0 }).unwrap();
    store.set(frigate, Defence { value: 75.0 }).unwrap();
    store.set(frigate, ImpulseSpeed { value: 125.0 }).unwrap();

    // Create an instance of the MammothFreighter. This entity will inherit the
    // ImpulseSpeed from SpaceShip, Defence from Freighter and FreightCapacity
    // from MammothFreighter.
    let inst = store.entity_named("my_freighter");
    add_is_a(&mut store, inst, mammoth_freighter);

    // Add a private Position component.
    store.set(inst, Position { x: 10.0, y: 20.0 }).unwrap();

    // Instances can override inherited components to give them a private copy
    // of the component. This freighter got an armor upgrade:
    store.set(inst, Defence { value: 100.0 }).unwrap();

    // Queries can match components from multiple levels of inheritance
    for (e, p) in store.query::<&Position>().iter() {
        if let (Some(s), Some(d), Some(c)) = (
            get_inherited::<ImpulseSpeed>(&store, e),
            get_inherited::<Defence>(&store, e),
            get_inherited::<FreightCapacity>(&store, e),
        ) {
            elog!("{}:", store.name_of(e));
            elog!(" - position: {}, {}", p.x, p.y);
            elog!(" - impulse speed: {}", s.value);
            elog!(" - defense: {}", d.value);
            elog!(" - capacity: {}", c.value);
        }
    }
}

#[test]
fn test_prefab_variant() {
    main();
    assert_eq!(
        take_log().join("\n"),
        concat!(
            "my_freighter:\n",
            " - position: 10, 20\n",
            " - impulse speed: 50\n",
            " - defense: 100\n",
            " - capacity: 500"
        )
    );
}
