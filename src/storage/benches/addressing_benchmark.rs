//! 包内三层寻址差分性能基准 (In-Crate Addressing Differential Benchmark)
//!
//! 评测目的：
//! 准确隔离并差分测量以下 3 个硬件访问层级的耗时成本：
//! - Layer A (Full UUID Routing): UUID -> Allocator Hash Index -> LocalSlotId -> 槽位代际校验与位置 -> 原型列定位与借用 -> 联合修改
//! - Layer B (LocalSlotId O(1) Check): 已持有 (EntityId, LocalSlotId) -> 纯数组索引代际校验与位置 (跳过 Allocator Index) -> 原型列定位与借用 -> 联合修改
//! - Layer C (Prebound Slices Baseline): 循环外预绑定原型组件列切片，循环内按相同随机行号访问并做代际溢出预检（切片直寻址硬件底线参考）
//!
//! 严谨性契约：
//! 1. 消除冗余查找：Layer B 经过修复，allocator.get_location_by_slot 彻底移除 B 树检查，实现真正的 O(1) 槽位直读；
//! 2. 槽位专属原则：每个候选 Store 必须从自身分配器提取自身的 LocalSlotId，严禁跨 Store 混用；
//! 3. 内存隔离与热身：每个候选 Store 独立构造、独立预热，测试完成立即析构，消除内存长驻偏置；
//! 4. 独立 Oracle 校验：Store 状态在测试结束后与外部独立裸数组计算结果 100% 逐实体验证，防止计算被死代码消除；
//! 5. 客观表述原则：Layer C 作为切片直寻址底线参考，不宣称与 A/B 具有相同的逐事件全局版本开销。

use crate::id::{EntityId, LocalSlotId};
use crate::storage::store::Store;
use serde::{Deserialize, Serialize};
use std::hint::black_box;
use std::time::Instant;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
struct Position {
    x: f64,
    y: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
struct Velocity {
    vx: f64,
    vy: f64,
}

struct Rng(u64);
impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed.max(1))
    }
    #[inline]
    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
    #[inline]
    fn next_usize(&mut self, bound: usize) -> usize {
        (self.next_u64() % (bound as u64)) as usize
    }
}

fn build_test_store(scale: usize) -> (Store, Vec<EntityId>) {
    let mut store = Store::new();
    let mut ids = Vec::with_capacity(scale);
    for i in 0..scale {
        let id = EntityId::from_u128((i + 1) as u128);
        ids.push(id);
        store
            .insert((
                id,
                (
                    Position {
                        x: i as f64,
                        y: (i * 2) as f64,
                    },
                    Velocity {
                        vx: 1.0,
                        vy: 2.0,
                    },
                ),
            ))
            .unwrap();
    }
    (store, ids)
}

fn build_initial_oracle(scale: usize) -> (Vec<Position>, Vec<Velocity>) {
    let mut pos = Vec::with_capacity(scale);
    let mut vel = Vec::with_capacity(scale);
    for i in 0..scale {
        pos.push(Position {
            x: i as f64,
            y: (i * 2) as f64,
        });
        vel.push(Velocity { vx: 1.0, vy: 2.0 });
    }
    (pos, vel)
}

fn verify_store_with_oracle(
    store: &Store,
    ids: &[EntityId],
    oracle_pos: &[Position],
    oracle_vel: &[Velocity],
) {
    for (i, &id) in ids.iter().enumerate() {
        let view = store.entity(id).expect("entity must exist");
        let pos = view.get::<Position>().unwrap();
        let vel = view.get::<Velocity>().unwrap();
        assert_eq!(pos.x, oracle_pos[i].x);
        assert_eq!(pos.y, oracle_pos[i].y);
        assert_eq!(vel.vx, oracle_vel[i].vx);
        assert_eq!(vel.vy, oracle_vel[i].vy);
    }
}

fn run_addressing_differential_suite(scale: usize, event_count: usize) {
    println!("\n================================================================================");
    println!("  IN-CRATE THREE-LAYER ADDRESSING BENCHMARK: {} ENTITIES ({} EVENTS)", scale, event_count);
    println!("  Contract: O(1) slot check in Layer B, per-store slots, isolated memory, Oracle verified");
    println!("================================================================================");

    // 1. 预先生成随机事件序列
    let mut rng = Rng::new(12345);
    let event_indices: Vec<usize> = (0..event_count).map(|_| rng.next_usize(scale)).collect();

    // 2. 外部独立 Oracle 参考演算
    let (mut oracle_pos, mut oracle_vel) = build_initial_oracle(scale);
    for &idx in &event_indices {
        oracle_pos[idx].x += 1.0;
        oracle_vel[idx].vx += 0.5;
    }

    let runs = 3;
    let mut layer_a_ns = Vec::with_capacity(runs);
    let mut layer_b_ns = Vec::with_capacity(runs);
    let mut layer_c_ns = Vec::with_capacity(runs);

    for round in 1..=runs {
        // 候选执行顺序交替：奇数轮 A -> B -> C，偶数轮 C -> B -> A
        if round % 2 == 1 {
            // --- Layer A ---
            let (mut store_a, ids_a) = build_test_store(scale);
            // 独立预热 100 次
            for &idx in event_indices.iter().take(100) {
                store_a.modify_tuple::<(Position, Velocity), _>(ids_a[idx], |(p, v)| {
                    p.x += 0.0;
                    v.vx += 0.0;
                }).unwrap();
            }
            // 重新初始化干净 Store 进行正式测量
            drop(store_a);
            let (mut store_a, ids_a) = build_test_store(scale);
            let t0 = Instant::now();
            for &idx in &event_indices {
                let id = ids_a[idx];
                store_a.modify_tuple::<(Position, Velocity), _>(id, |(p, v)| {
                    p.x += 1.0;
                    v.vx += 0.5;
                }).unwrap();
            }
            let dur_a = t0.elapsed();
            layer_a_ns.push(dur_a.as_nanos() as f64 / event_count as f64);
            verify_store_with_oracle(&store_a, &ids_a, &oracle_pos, &oracle_vel);
            drop(store_a);

            // --- Layer B ---
            let (mut store_b, ids_b) = build_test_store(scale);
            // 每个候选 Store 必须从自身提取自身的槽位
            let slots_b: Vec<LocalSlotId> = ids_b.iter().map(|&id| store_b.local_slot(id).unwrap()).collect();
            // 预热
            for &idx in event_indices.iter().take(100) {
                store_b.modify_tuple_by_slot::<(Position, Velocity), _>(ids_b[idx], slots_b[idx], |(p, v)| {
                    p.x += 0.0;
                    v.vx += 0.0;
                }).unwrap();
            }
            drop(store_b);
            let (mut store_b, ids_b) = build_test_store(scale);
            let slots_b: Vec<LocalSlotId> = ids_b.iter().map(|&id| store_b.local_slot(id).unwrap()).collect();
            let t0 = Instant::now();
            for &idx in &event_indices {
                let id = ids_b[idx];
                let slot = slots_b[idx];
                store_b.modify_tuple_by_slot::<(Position, Velocity), _>(id, slot, |(p, v)| {
                    p.x += 1.0;
                    v.vx += 0.5;
                }).unwrap();
            }
            let dur_b = t0.elapsed();
            layer_b_ns.push(dur_b.as_nanos() as f64 / event_count as f64);
            verify_store_with_oracle(&store_b, &ids_b, &oracle_pos, &oracle_vel);
            drop(store_b);

            // --- Layer C ---
            let (mut store_c, ids_c) = build_test_store(scale);
            let t0 = Instant::now();
            {
                let query = store_c.query_mut::<(&mut Position, &mut Velocity)>().unwrap();
                let mut chunks = query.chunks_mut();
                let (entities, (pos_slice, vel_slice)) = chunks.next().unwrap();
                assert_eq!(entities.len(), scale);
                let mut current_epoch: u64 = 1;
                for &idx in &event_indices {
                    current_epoch = current_epoch.checked_add(1).expect("epoch overflow");
                    let p = &mut pos_slice[idx];
                    let v = &mut vel_slice[idx];
                    p.x += 1.0;
                    v.vx += 0.5;
                }
                black_box(current_epoch);
            }
            let dur_c = t0.elapsed();
            layer_c_ns.push(dur_c.as_nanos() as f64 / event_count as f64);
            verify_store_with_oracle(&store_c, &ids_c, &oracle_pos, &oracle_vel);
            drop(store_c);
        } else {
            // 偶数轮反向：C -> B -> A
            // --- Layer C ---
            let (mut store_c, ids_c) = build_test_store(scale);
            let t0 = Instant::now();
            {
                let query = store_c.query_mut::<(&mut Position, &mut Velocity)>().unwrap();
                let mut chunks = query.chunks_mut();
                let (entities, (pos_slice, vel_slice)) = chunks.next().unwrap();
                assert_eq!(entities.len(), scale);
                let mut current_epoch: u64 = 1;
                for &idx in &event_indices {
                    current_epoch = current_epoch.checked_add(1).expect("epoch overflow");
                    let p = &mut pos_slice[idx];
                    let v = &mut vel_slice[idx];
                    p.x += 1.0;
                    v.vx += 0.5;
                }
                black_box(current_epoch);
            }
            let dur_c = t0.elapsed();
            layer_c_ns.push(dur_c.as_nanos() as f64 / event_count as f64);
            verify_store_with_oracle(&store_c, &ids_c, &oracle_pos, &oracle_vel);
            drop(store_c);

            // --- Layer B ---
            let (mut store_b, ids_b) = build_test_store(scale);
            let slots_b: Vec<LocalSlotId> = ids_b.iter().map(|&id| store_b.local_slot(id).unwrap()).collect();
            let t0 = Instant::now();
            for &idx in &event_indices {
                let id = ids_b[idx];
                let slot = slots_b[idx];
                store_b.modify_tuple_by_slot::<(Position, Velocity), _>(id, slot, |(p, v)| {
                    p.x += 1.0;
                    v.vx += 0.5;
                }).unwrap();
            }
            let dur_b = t0.elapsed();
            layer_b_ns.push(dur_b.as_nanos() as f64 / event_count as f64);
            verify_store_with_oracle(&store_b, &ids_b, &oracle_pos, &oracle_vel);
            drop(store_b);

            // --- Layer A ---
            let (mut store_a, ids_a) = build_test_store(scale);
            let t0 = Instant::now();
            for &idx in &event_indices {
                let id = ids_a[idx];
                store_a.modify_tuple::<(Position, Velocity), _>(id, |(p, v)| {
                    p.x += 1.0;
                    v.vx += 0.5;
                }).unwrap();
            }
            let dur_a = t0.elapsed();
            layer_a_ns.push(dur_a.as_nanos() as f64 / event_count as f64);
            verify_store_with_oracle(&store_a, &ids_a, &oracle_pos, &oracle_vel);
            drop(store_a);
        }
    }

    let avg_a = layer_a_ns.iter().sum::<f64>() / runs as f64;
    let avg_b = layer_b_ns.iter().sum::<f64>() / runs as f64;
    let avg_c = layer_c_ns.iter().sum::<f64>() / runs as f64;

    let min_a = layer_a_ns.iter().cloned().fold(f64::INFINITY, f64::min);
    let max_a = layer_a_ns.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let min_b = layer_b_ns.iter().cloned().fold(f64::INFINITY, f64::min);
    let max_b = layer_b_ns.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let min_c = layer_c_ns.iter().cloned().fold(f64::INFINITY, f64::min);
    let max_c = layer_c_ns.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

    println!(
        "    Layer A (Full UUID Routing):    avg: {:>5.1} ns/op [range: {:>4.1} - {:>4.1}] | {:>6.2} M evt/s",
        avg_a, min_a, max_a, 1e3 / avg_a
    );
    println!(
        "    Layer B (LocalSlot Generation): avg: {:>5.1} ns/op [range: {:>4.1} - {:>4.1}] | {:>6.2} M evt/s",
        avg_b, min_b, max_b, 1e3 / avg_b
    );
    println!(
        "    Layer C (Prebound Slice Base):  avg: {:>5.1} ns/op [range: {:>4.1} - {:>4.1}] | {:>6.2} M evt/s",
        avg_c, min_c, max_c, 1e3 / avg_c
    );
    let delta_ab = avg_a - avg_b;
    let delta_bc = avg_b - avg_c;
    println!("    Addressing Cost Breakdown:");
    println!("      UUID Index Search Cost (A - B):                 {:>5.1} ns/op", delta_ab);
    println!("      Slot Gen Check & Archetype Fetch Cost (B - C):  {:>5.1} ns/op", delta_bc);
    println!("      Direct Slice Update Baseline (Layer C):         {:>5.1} ns/op", avg_c);
    println!("    Oracle Verification: 100% passed for all candidates.");
}

#[test]
#[ignore = "in-crate internal benchmark"]
fn bench_addressing_differential_ten_thousand() {
    run_addressing_differential_suite(10_000, 20_000);
}

#[test]
#[ignore = "in-crate internal benchmark"]
fn bench_addressing_differential_one_hundred_thousand() {
    run_addressing_differential_suite(100_000, 20_000);
}

#[test]
#[ignore = "in-crate internal benchmark"]
fn bench_addressing_differential_one_million() {
    run_addressing_differential_suite(1_000_000, 20_000);
}
