//! Quick Start runnable example & test matching `README.md` / `README_zh-CN.md`.

use serde::{Deserialize, Serialize};
use sumeru::{EntityId, Pair, Store};

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
struct Position {
    x: f32,
    y: f32,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
struct Velocity {
    vx: f32,
    vy: f32,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
struct Active;

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
struct Frozen;

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
struct ChildOf;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut store = Store::new();
    let root = EntityId::from_u128(1);
    let entity = EntityId::from_u128(2);

    store.insert((
        entity,
        (
            Position { x: 0.0, y: 0.0 },
            Velocity { vx: 1.5, vy: 2.0 },
            Active,
            Pair::<ChildOf>::new(ChildOf, root),
        ),
    ))?;

    let baseline_epoch = store.epoch();

    // 设置状态位
    store.set_flag::<0>(entity, true)?;

    // 链式过滤并批量更新
    for (_id, (pos, vel)) in store
        .query_mut::<(&mut Position, &Velocity)>()?
        .with::<Active>()
        .without::<Frozen>()
        .with_pair::<ChildOf>(root)
        .with_flag::<0>()
        .iter()
    {
        pos.x += vel.vx;
        pos.y += vel.vy;
    }

    assert_eq!(
        store.get::<Position>(entity),
        Some(&Position { x: 1.5, y: 2.0 })
    );

    // 查询 baseline_epoch 之后发生变更的数据
    let dirty_pos = store
        .query::<&Position>()
        .changed_since::<Position>(baseline_epoch)
        .iter()
        .count();
    let dirty_vel = store
        .query::<&Velocity>()
        .changed_since::<Velocity>(baseline_epoch)
        .iter()
        .count();

    assert_eq!(dirty_pos, 1);
    assert_eq!(dirty_vel, 0);
    Ok(())
}

#[test]
fn test_quick_start() {
    main().expect("Quick Start example should execute and pass all assertions");
}
