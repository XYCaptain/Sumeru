//! Port of `examples/flecs/game_mechanics/inventory_system.rs` using `sumer_state::Store`.

use crate::common::*;
use crate::elog;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct Item;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct Container;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default, Serialize, Deserialize)]
pub struct Inventory;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default, Serialize, Deserialize)]
pub struct ContainedBy;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Default, Serialize, Deserialize)]
pub struct ContainedOrder(pub u32);

static NEXT_CONTAINED_ORDER: std::sync::atomic::AtomicU32 =
    std::sync::atomic::AtomicU32::new(1);

fn place_in_container(store: &mut Store, item: EntityId, container: EntityId) {
    let seq = NEXT_CONTAINED_ORDER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    store.set_pair(item, ContainedBy, container).unwrap();
    store.set(item, ContainedOrder(seq)).unwrap();
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default, Serialize, Deserialize)]
pub struct IsA;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct Active;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct Amount {
    pub amount: i32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct Health {
    pub value: i32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct Attack {
    pub value: i32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct Sword;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct Armor;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct Coin;

fn get_inherited<T: Component>(store: &Store, entity: EntityId) -> Option<&T> {
    let mut cur = Some(entity);
    while let Some(e) = cur {
        if let Some(v) = store.get::<T>(e) {
            return Some(v);
        }
        cur = store.pair_target::<IsA>(e);
    }
    None
}

fn instantiate_item(store: &mut Store, prefab: EntityId) -> EntityId {
    let inst = store.spawn_auto().unwrap();
    store.set_pair(inst, IsA, prefab).unwrap();
    // Health defaults to OnInstantiate::Override (copied to instance)
    if let Some(&hp) = store.get::<Health>(prefab) {
        store.set(inst, hp).unwrap();
    }
    inst
}

fn item_kind_name(store: &Store, item: EntityId) -> Option<&'static str> {
    let mut cur = Some(item);
    while let Some(e) = cur {
        if store.has::<Sword>(e) {
            return Some("Sword");
        }
        if store.has::<Armor>(e) {
            return Some("Armor");
        }
        if store.has::<Coin>(e) {
            return Some("Coin");
        }
        cur = store.pair_target::<IsA>(e);
    }
    None
}

fn item_name(store: &Store, item: EntityId) -> String {
    if let Some(prefab) = store.pair_target::<IsA>(item) {
        return store.name_of(prefab);
    }
    item_kind_name(store, item)
        .unwrap_or("UnknownItem")
        .to_string()
}

fn get_container(store: &Store, entity: EntityId) -> EntityId {
    if store.has::<Container>(entity) {
        return entity;
    }
    store
        .pair_target::<Inventory>(entity)
        .expect("entity has Inventory target")
}

fn items_in_container(store: &Store, container: EntityId) -> Vec<EntityId> {
    let actual = get_container(store, container);
    let mut items = store.sources_with_pair::<ContainedBy>(actual);
    items.sort_by_key(|e| {
        (
            store.get::<ContainedOrder>(*e).map(|o| o.0).unwrap_or(0),
            EntityHandle::index(*e),
        )
    });
    items
}

fn find_item_w_kind(
    store: &Store,
    container: EntityId,
    kind: &str,
    active_required: bool,
) -> Option<EntityId> {
    for item in items_in_container(store, container) {
        if active_required && !store.has::<Active>(item) {
            continue;
        }
        if item_kind_name(store, item) == Some(kind) {
            return Some(item);
        }
    }
    None
}

fn transfer_item(store: &mut Store, container: EntityId, item: EntityId) {
    let amt = store
        .get::<Amount>(item)
        .copied()
        .unwrap_or(Amount { amount: 1 });
    if amt.amount > 0 {
        let ik = item_kind_name(store, item).unwrap();
        if let Some(dst_item) = find_item_w_kind(store, container, ik, false) {
            if let Some(dst_amt) = store.get_mut::<Amount>(dst_item) {
                dst_amt.amount += amt.amount;
                store.despawn(item).unwrap();
                return;
            }
        }
    }
    place_in_container(store, item, container);
}

fn transfer_items(store: &mut Store, dst: EntityId, src: EntityId) {
    elog!(
        ">> Transfer items from {} to {}\n",
        store.name_of(src),
        store.name_of(dst)
    );
    let dst_container = get_container(store, dst);
    let src_container = get_container(store, src);
    let items = items_in_container(store, src_container);
    for item in items {
        transfer_item(store, dst_container, item);
    }
}

fn attack(store: &mut Store, player: EntityId, weapon: EntityId) {
    let player_name = store.name_of(player);
    let wep_name = item_name(store, weapon);
    elog!(">> {player_name} is attacked with a {wep_name}!");

    let Some(&atk) = get_inherited::<Attack>(store, weapon) else {
        elog!(" - the weapon is a dud");
        return;
    };
    let mut att_value = atk.value;

    if let Some(armor_e) = find_item_w_kind(store, player, "Armor", true) {
        let armor_label = item_name(store, armor_e);
        if let Some(armor_hp) = store.get::<Health>(armor_e).map(|h| h.value) {
            elog!(" - {player_name} defends with {armor_label} ({armor_hp} health)");
            let next_hp = armor_hp - att_value;
            if next_hp <= 0 {
                att_value = -next_hp;
                store.despawn(armor_e).unwrap();
                elog!(" - {armor_label} is destroyed!");
            } else {
                store.set(armor_e, Health { value: next_hp }).unwrap();
                elog!(
                    " - {armor_label} has {next_hp} health left after taking {att_value} damage"
                );
                att_value = 0;
            }
        } else {
            elog!(" - the {armor_label} armor is a dud");
        }
    } else {
        elog!(" - {player_name} fights without armor!");
    }

    if let Some(wep_hp) = store.get::<Health>(weapon).map(|h| h.value) {
        if wep_hp > 0 {
            let next_wep_hp = wep_hp - 1;
            if next_wep_hp == 0 {
                elog!(" - {wep_name} is destroyed!");
                store.despawn(weapon).unwrap();
            } else {
                store.set(weapon, Health { value: next_wep_hp }).unwrap();
                elog!(" - {wep_name} has {next_wep_hp} uses left");
            }
        }
    }

    if att_value > 0 {
        if let Some(player_hp) = store.get::<Health>(player).map(|h| h.value) {
            let next_player_hp = player_hp - att_value;
            if next_player_hp <= 0 {
                elog!(" - {player_name} died!");
                store.despawn(player).unwrap();
            } else {
                store.set(player, Health { value: next_player_hp }).unwrap();
                elog!(
                    " - {player_name} has {next_player_hp} health left after taking {att_value} damage"
                );
            }
        }
    }

    elog!();
}

fn print_items(store: &Store, container: EntityId) {
    elog!("-- {}'s inventory:", store.name_of(container));
    let items = items_in_container(store, container);
    if items.is_empty() {
        elog!(" - << empty >>");
    } else {
        for item in items {
            let amount = store
                .get::<Amount>(item)
                .map(|a| a.amount)
                .unwrap_or(1);
            let name = item_name(store, item);
            let kind = item_kind_name(store, item).unwrap_or("Item");
            elog!(" - {amount} {name} ({kind})");
        }
    }
    elog!();
}

pub fn main() {
    clear_log();
    let mut store = Store::new();

    // Register item prefabs in Store
    let wooden_sword = store.prefab_named("WoodenSword");
    store.set(wooden_sword, Sword).unwrap();
    store.set(wooden_sword, Attack { value: 1 }).unwrap();
    store.set(wooden_sword, Health { value: 5 }).unwrap();

    let iron_sword = store.prefab_named("IronSword");
    store.set(iron_sword, Sword).unwrap();
    store.set(iron_sword, Attack { value: 4 }).unwrap();
    store.set(iron_sword, Health { value: 10 }).unwrap();

    let wooden_armor = store.prefab_named("WoodenArmor");
    store.set(wooden_armor, Armor).unwrap();
    store.set(wooden_armor, Health { value: 10 }).unwrap();

    let iron_armor = store.prefab_named("IronArmor");
    store.set(iron_armor, Armor).unwrap();
    store.set(iron_armor, Health { value: 20 }).unwrap();

    // Create a loot box with items
    let loot_box = store.entity_named("Chest");
    store.set(loot_box, Container).unwrap();

    let chest_sword = instantiate_item(&mut store, iron_sword);
    place_in_container(&mut store, chest_sword, loot_box);

    let chest_armor = instantiate_item(&mut store, wooden_armor);
    place_in_container(&mut store, chest_armor, loot_box);

    let chest_coins = store.spawn_auto().unwrap();
    store.set(chest_coins, Coin).unwrap();
    store.set(chest_coins, Amount { amount: 30 }).unwrap();
    place_in_container(&mut store, chest_coins, loot_box);

    // Create a player entity with an inventory
    let player = store.entity_named("Player");
    store.set(player, Health { value: 10 }).unwrap();
    let player_inv = store.spawn_auto().unwrap();
    store.set(player_inv, Container).unwrap();
    store.set_pair(player, Inventory, player_inv).unwrap();

    let player_coins = store.spawn_auto().unwrap();
    store.set(player_coins, Coin).unwrap();
    store.set(player_coins, Amount { amount: 20 }).unwrap();
    place_in_container(&mut store, player_coins, player_inv);

    print_items(&store, loot_box);
    print_items(&store, player);

    transfer_items(&mut store, player, loot_box);

    print_items(&store, player);
    print_items(&store, loot_box);

    if let Some(armor) = find_item_w_kind(&store, player, "Armor", false) {
        store.set(armor, Active).unwrap();
    }

    let my_sword = instantiate_item(&mut store, iron_sword);

    for _ in 0..5 {
        attack(&mut store, player, my_sword);
    }
}

#[test]
fn test_inventory_system() {
    main();
    let lines = take_log();
    assert_eq!(
        lines,
        vec![
            "-- Chest's inventory:",
            " - 1 IronSword (Sword)",
            " - 1 WoodenArmor (Armor)",
            " - 30 Coin (Coin)",
            "",
            "-- Player's inventory:",
            " - 20 Coin (Coin)",
            "",
            ">> Transfer items from Chest to Player\n",
            "-- Player's inventory:",
            " - 50 Coin (Coin)",
            " - 1 IronSword (Sword)",
            " - 1 WoodenArmor (Armor)",
            "",
            "-- Chest's inventory:",
            " - << empty >>",
            "",
            ">> Player is attacked with a IronSword!",
            " - Player defends with WoodenArmor (10 health)",
            " - WoodenArmor has 6 health left after taking 4 damage",
            " - IronSword has 9 uses left",
            "",
            ">> Player is attacked with a IronSword!",
            " - Player defends with WoodenArmor (6 health)",
            " - WoodenArmor has 2 health left after taking 4 damage",
            " - IronSword has 8 uses left",
            "",
            ">> Player is attacked with a IronSword!",
            " - Player defends with WoodenArmor (2 health)",
            " - WoodenArmor is destroyed!",
            " - IronSword has 7 uses left",
            " - Player has 8 health left after taking 2 damage",
            "",
            ">> Player is attacked with a IronSword!",
            " - Player fights without armor!",
            " - IronSword has 6 uses left",
            " - Player has 4 health left after taking 4 damage",
            "",
            ">> Player is attacked with a IronSword!",
            " - Player fights without armor!",
            " - IronSword has 5 uses left",
            " - Player died!",
            "",
        ]
    );
}
