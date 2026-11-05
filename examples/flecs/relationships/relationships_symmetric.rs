use crate::common::*;

// Type for TradesWith relationship
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub struct TradesWith;

fn add_symmetric_trade(store: &mut Store, a: EntityId, b: EntityId) {
    store.set_pair(a, TradesWith, b).unwrap();
    store.set_pair(b, TradesWith, a).unwrap();
}

fn trades_with(store: &Store, a: EntityId, b: EntityId) -> bool {
    store.pair_target::<TradesWith>(a) == Some(b)
}

pub fn main() {
    clear_log();
    let mut store = Store::new();

    // Create two players
    let player_1 = store.spawn_empty();
    let player_2 = store.spawn_empty();

    // Add (TradesWith, player_2) to player_1. Because the relationship is symmetric,
    // this also adds (TradesWith, player_1) to player_2.
    add_symmetric_trade(&mut store, player_1, player_2);

    // Log platoon of unit
    elog!(
        "Player 1 trades with Player 2: {}",
        trades_with(&store, player_1, player_2)
    );
    elog!(
        "Player 2 trades with Player 1: {}",
        trades_with(&store, player_2, player_1)
    );
}

#[test]
fn test_relationships_symmetric() {
    main();
    assert_eq!(
        take_log().join("\n"),
        concat!(
            "Player 1 trades with Player 2: true\n",
            "Player 2 trades with Player 1: true"
        )
    );
}
