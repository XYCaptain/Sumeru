use crate::common::*;

// Regular enumerations can be added to entities as (Relationship, Target) pairs
// where the enum type is the exclusive relationship and the variant is the target.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum Tile {
    Grass,
    Sand,
    Stone,
}

impl Tile {
    fn variant_name(self) -> &'static str {
        match self {
            Tile::Grass => "Grass",
            Tile::Sand => "Sand",
            Tile::Stone => "Stone",
        }
    }

    fn full_path(self) -> String {
        format!("::relationships_enum::Tile::{}", self.variant_name())
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum TileStatus {
    Free,
    Occupied,
}

impl TileStatus {
    fn variant_name(self) -> &'static str {
        match self {
            TileStatus::Free => "Free",
            TileStatus::Occupied => "Occupied",
        }
    }
}

fn format_enum_type(store: &Store, entity: EntityId) -> String {
    let mut parts = Vec::new();
    if let Some(tile) = store.get::<Tile>(entity) {
        parts.push(format!(
            "(relationships_enum.Tile,relationships_enum.Tile.{})",
            tile.variant_name()
        ));
    }
    if let Some(status) = store.get::<TileStatus>(entity) {
        parts.push(format!(
            "(relationships_enum.TileStatus,relationships_enum.TileStatus.{})",
            status.variant_name()
        ));
    }
    parts.join(", ")
}

pub fn main() {
    clear_log();
    let mut store = Store::new();

    // Create an entity with (Tile, Stone) and (TileStatus, Free) relationships
    let tile = store.spawn_empty();
    store.set(tile, Tile::Stone).unwrap();
    store.set(tile, TileStatus::Free).unwrap();

    // (Tile, Tile.Stone), (TileStatus, TileStatus.Free)
    elog!("{}", format_enum_type(&store, tile));

    // Replace (TileStatus, Free) with (TileStatus, Occupied)
    store.set(tile, TileStatus::Occupied).unwrap();

    // (Tile, Tile.Stone), (TileStatus, TileStatus.Occupied)
    elog!("{}", format_enum_type(&store, tile));
    elog!();

    // Check if the entity has the Tile relationship and the Tile::Stone pair
    elog!("has tile enum: {}", store.has::<Tile>(tile));
    elog!(
        "is the enum from tile stone?: {}",
        store.get::<Tile>(tile) == Some(&Tile::Stone)
    );

    // Get the current value of the enum
    if let Some(v) = store.get::<Tile>(tile) {
        elog!("is tile stone: {}", *v == Tile::Stone);
    }
    elog!();

    // Create a few more entities that we can query
    let t2 = store.spawn_empty();
    store.set(t2, Tile::Grass).unwrap();
    store.set(t2, TileStatus::Free).unwrap();

    let t3 = store.spawn_empty();
    store.set(t3, Tile::Sand).unwrap();
    store.set(t3, TileStatus::Occupied).unwrap();

    // Iterate all entities with a Tile relationship
    for (_e, t) in store.query::<&Tile>().iter() {
        elog!("{}", t.full_path());
    }
    elog!();

    // Iterate only occupied tiles
    for (_e, (t, status)) in store.query::<(&Tile, &TileStatus)>().iter() {
        if *status == TileStatus::Occupied {
            elog!("{}", t.full_path());
        }
    }
    elog!();

    // Remove any instance of the TileStatus relationship
    store.remove::<TileStatus>(tile).unwrap();

    // (Tile, Tile.Stone)
    elog!("{}", format_enum_type(&store, tile));
}

#[test]
fn test_relationships_enum() {
    main();
    assert_eq!(
        take_log().join("\n"),
        concat!(
            "(relationships_enum.Tile,relationships_enum.Tile.Stone), (relationships_enum.TileStatus,relationships_enum.TileStatus.Free)\n",
            "(relationships_enum.Tile,relationships_enum.Tile.Stone), (relationships_enum.TileStatus,relationships_enum.TileStatus.Occupied)\n",
            "\n",
            "has tile enum: true\n",
            "is the enum from tile stone?: true\n",
            "is tile stone: true\n",
            "\n",
            "::relationships_enum::Tile::Stone\n",
            "::relationships_enum::Tile::Grass\n",
            "::relationships_enum::Tile::Sand\n",
            "\n",
            "::relationships_enum::Tile::Stone\n",
            "::relationships_enum::Tile::Sand\n",
            "\n",
            "(relationships_enum.Tile,relationships_enum.Tile.Stone)"
        )
    );
}
