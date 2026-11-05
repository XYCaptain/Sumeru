//! Port of `examples/flecs/reflection/reflection_basics_bitmask.rs` using `sumeru::Store`.

use crate::common::*;
use crate::elog;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct Toppings {
    pub value: u32,
}

impl Toppings {
    pub const BACON: u32 = 0x1;
    pub const LETTUCE: u32 = 0x2;
    pub const TOMATO: u32 = 0x4;

    pub fn new() -> Self {
        Self { value: 0 }
    }

    pub fn add(&mut self, topping: u32) {
        self.value |= topping;
    }

    pub fn remove(&mut self, topping: u32) {
        self.value &= !topping;
    }

    pub fn has(&self, topping: u32) -> bool {
        self.value & topping != 0
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct Sandwich {
    pub toppings: Toppings,
}

impl Sandwich {
    pub fn new() -> Self {
        Self {
            toppings: Toppings::new(),
        }
    }

    pub fn add_topping(&mut self, topping: u32) {
        self.toppings.add(topping);
    }

    pub fn remove_topping(&mut self, topping: u32) {
        self.toppings.remove(topping);
    }

    pub fn has_topping(&self, topping: u32) -> bool {
        self.toppings.has(topping)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BitmaskMeta {
    pub bits: Vec<(String, u32)>,
}

impl BitmaskMeta {
    pub fn format_mask(&self, value: u32) -> String {
        let mut names = Vec::new();
        for (name, bit) in self.bits.iter().rev() {
            if value & bit != 0 {
                names.push(name.as_str());
            }
        }
        names.join("|")
    }
}

pub struct ToppingsField;

impl ComponentField<Sandwich> for ToppingsField {
    type Value = Toppings;

    fn extract<'a>(component: &'a Sandwich) -> &'a Self::Value {
        &component.toppings
    }

    fn field_name() -> &'static str {
        "toppings"
    }
}

pub fn main() {
    clear_log();
    let mut store = Store::new();

    let toppings_meta_entity = store.entity_named("ToppingsMeta");
    store
        .set(
            toppings_meta_entity,
            BitmaskMeta {
                bits: vec![
                    ("bacon".to_string(), Toppings::BACON),
                    ("lettuce".to_string(), Toppings::LETTUCE),
                    ("tomato".to_string(), Toppings::TOMATO),
                ],
            },
        )
        .unwrap();

    // Create entity with Sandwich
    let e = store.spawn_auto().unwrap();
    store
        .set(
            e,
            Sandwich {
                toppings: Toppings {
                    value: Toppings::BACON | Toppings::LETTUCE,
                },
            },
        )
        .unwrap();

    // Convert Sandwich component to flecs expression string
    if let (Some(val), Some(meta)) = (
        store.get::<Sandwich>(e),
        store.get::<BitmaskMeta>(toppings_meta_entity),
    ) {
        let mask_str = meta.format_mask(ToppingsField::extract(val).value);
        elog!("{{{}: {}}}", ToppingsField::field_name(), mask_str);
    }
}

#[test]
fn test_reflection_basics_bitmask() {
    main();
    let lines = take_log();
    assert_eq!(lines, vec!["{toppings: lettuce|bacon}"]);
}
