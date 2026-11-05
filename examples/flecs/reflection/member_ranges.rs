//! Port of `examples/flecs/reflection/member_ranges.rs` using `sumer_state::Store`.

use crate::common::*;
use crate::elog;
use sumer_state::SchemaVersion;

#[derive(Debug, Clone, Copy, PartialEq, Default, Serialize, Deserialize)]
pub struct CpuUtilization {
    pub value: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct MemberRanges {
    pub value_range: (f64, f64),
    pub warning_range: (f64, f64),
    pub error_range: (f64, f64),
}

impl MemberRanges {
    pub fn classify(&self, v: f64) -> &'static str {
        if v < self.error_range.0 || v > self.error_range.1 {
            "error"
        } else if v < self.warning_range.0 || v > self.warning_range.1 {
            "warning"
        } else {
            "normal"
        }
    }
}

pub struct ValueField;

impl ComponentField<CpuUtilization> for ValueField {
    type Value = f64;

    fn extract<'a>(component: &'a CpuUtilization) -> &'a Self::Value {
        &component.value
    }

    fn field_name() -> &'static str {
        "value"
    }
}

pub fn main() {
    clear_log();
    let mut store = Store::new();
    store
        .register_components([ComponentRegistration::of_unsupported::<CpuUtilization>()])
        .unwrap();
    let _schema = ComponentSchema::of::<CpuUtilization>(SchemaVersion::new(1));

    let meta_entity = store.entity_named("CpuUtilizationMeta");
    let ranges = MemberRanges {
        value_range: (0.0, 100.0),
        warning_range: (0.0, 60.0),
        error_range: (0.0, 80.0),
    };
    store.set(meta_entity, ranges).unwrap();

    let machine_a = store.entity_named("MachineA");
    store.set(machine_a, CpuUtilization { value: 50.0 }).unwrap();

    let machine_b = store.entity_named("MachineB");
    store.set(machine_b, CpuUtilization { value: 75.0 }).unwrap();

    let machine_c = store.entity_named("MachineC");
    store.set(machine_c, CpuUtilization { value: 90.0 }).unwrap();

    let meta = *store.get::<MemberRanges>(meta_entity).unwrap();
    for machine in [machine_a, machine_b, machine_c] {
        let cpu = store.get::<CpuUtilization>(machine).unwrap();
        let val = *ValueField::extract(cpu);
        elog!(
            "{}: {}.{} = {} ({})",
            store.name_of(machine),
            "CpuUtilization",
            ValueField::field_name(),
            val,
            meta.classify(val)
        );
    }
}

#[test]
fn test_member_ranges() {
    main();
    let lines = take_log();
    assert_eq!(
        lines,
        vec![
            "MachineA: CpuUtilization.value = 50 (normal)",
            "MachineB: CpuUtilization.value = 75 (warning)",
            "MachineC: CpuUtilization.value = 90 (error)",
        ]
    );
}
