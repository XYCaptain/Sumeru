use super::*;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
struct CompA(pub u64);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
struct CompB(pub u64);

#[derive(Debug, Clone, Serialize, Deserialize)]
struct PanicOnDrop(pub u64, #[serde(skip)] pub Arc<AtomicBool>);

impl Drop for PanicOnDrop {
    fn drop(&mut self) {
        if self.1.load(Ordering::SeqCst) {
            panic!("simulated panic inside Drop");
        }
    }
}

#[test]
fn test_modify_closure_panic_preserves_store_health() {
    // 验证闭包 panic 时遵循弱异常安全：代际推进，原地部分写入保留，Store 保持健康不中毒
    let mut store = Store::new();
    let id = EntityId::from_u128(100);
    store.insert((id, (CompA(10),))).unwrap();
    let initial_epoch = store.epoch();

    let res = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _ = store.modify::<CompA, ()>(id, |comp| {
            comp.0 = 999;
            panic!("simulated panic inside modify closure");
        });
    }));
    assert!(res.is_err());
    assert!(!store.is_poisoned(), "store must NOT be poisoned on in-place modify closure panic");
    assert_eq!(store.epoch(), initial_epoch + 1);
    assert_eq!(store.extract::<&CompA>(id).unwrap(), &CompA(999));

    // Store 结构未损毁，后续写操作正常运作
    store.modify::<CompA, ()>(id, |c| c.0 += 1).unwrap();
    assert_eq!(store.extract::<&CompA>(id).unwrap(), &CompA(1000));
}

#[test]
fn test_modify_tuple_closure_panic_preserves_store_health() {
    let mut store = Store::new();
    let id = EntityId::from_u128(101);
    store.insert((id, (CompA(10), CompB(20)))).unwrap();
    let initial_epoch = store.epoch();

    let res = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _ = store.modify_tuple::<(CompA, CompB), ()>(id, |(a, _b)| {
            a.0 = 888;
            panic!("simulated panic inside modify_tuple closure");
        });
    }));
    assert!(res.is_err());
    assert!(!store.is_poisoned(), "store must NOT be poisoned on in-place modify_tuple closure panic");
    assert_eq!(store.epoch(), initial_epoch + 1);
    assert_eq!(store.extract::<&CompA>(id).unwrap(), &CompA(888));
}

#[test]
fn test_poison_guard_on_delete_entity_drop_panic() {
    let should_panic = Arc::new(AtomicBool::new(false));
    let mut store = Store::new();
    let id = EntityId::from_u128(102);
    store
        .insert((id, (PanicOnDrop(1, should_panic.clone()),)))
        .unwrap();

    should_panic.store(true, Ordering::SeqCst);
    let res = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _ = store.delete_entity(id);
    }));
    assert!(res.is_err());
    assert!(
        store.is_poisoned(),
        "store must be marked poisoned after delete_entity drop panic"
    );
    assert_eq!(store.check_poisoned(), Err(StateError::SessionPoisoned));
}

