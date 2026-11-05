//! Port of `examples/flecs/observers/observer_entity_event.rs` using `sumer_state::Store`.

use crate::common::*;
use crate::elog;

// An event without payload
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct Click;

// An event with payload
#[derive(Debug, Clone, Copy, PartialEq, Default, Serialize, Deserialize)]
pub struct Resize {
    pub width: f32,
    pub height: f32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CloseReason {
    User,
    _System,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct CloseRequested {
    pub reason: CloseReason,
}

pub fn main() {
    clear_log();
    let mut store = Store::new();

    let widget = store.entity_named("MyWidget");
    elog!(
        "widget: Entity name: {} -- id: {} -- archetype: (Identifier,Name)",
        store.name_of(widget),
        EntityHandle::index(widget)
    );

    // Emit Click on widget: triggers entity-level Click observers stored/bound to `widget`
    store.set(widget, Click).unwrap();
    if store.has::<Click>(widget) {
        elog!("clicked on {:?}", store.name_of(widget));
        elog!("clicked!");
    }

    // Emit Resize on widget: triggers entity-level Resize payload observers
    let resize = Resize {
        width: 100.0,
        height: 200.0,
    };
    store.set(widget, resize).unwrap();
    if let Some(payload) = store.get::<Resize>(widget) {
        elog!(
            "{} resized to {{ {}, {} }}!",
            store.name_of(widget),
            payload.width,
            payload.height
        );
        elog!("widget resized to {{ {}, {} }}!", payload.width, payload.height);
    }

    // Emit CloseRequested on widget: triggers world-level observer matching any entity
    store
        .set(
            widget,
            CloseRequested {
                reason: CloseReason::User,
            },
        )
        .unwrap();
    for (_e, req) in store.query::<&CloseRequested>().iter() {
        elog!("Close request with reason: {:?}", req.reason);
    }
}

#[test]
fn test_observer_entity_event() {
    main();
    let lines = take_log();
    assert_eq!(lines.len(), 6);
    assert!(lines[0].starts_with("widget: Entity name: MyWidget -- id: "));
    assert_eq!(
        &lines[1..],
        &[
            "clicked on \"MyWidget\"",
            "clicked!",
            "MyWidget resized to { 100, 200 }!",
            "widget resized to { 100, 200 }!",
            "Close request with reason: User",
        ]
    );
}
