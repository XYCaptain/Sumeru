//! All 18 Flecs reflection examples implemented on `sumeru::Store`.

pub mod entity_type;
pub mod member_ranges;
pub mod reflection_basics;
pub mod reflection_basics_bitmask;
pub mod reflection_basics_deserialize;
pub mod reflection_basics_json;
pub mod reflection_basics_simple_enum;
pub mod reflection_nested_set_member;
pub mod reflection_nested_struct;
pub mod reflection_query_to_custom_json;
pub mod reflection_query_to_json;
pub mod reflection_runtime_component;
pub mod reflection_runtime_nested_component;
pub mod reflection_ser_opaque_type;
pub mod reflection_ser_std_string;
pub mod reflection_ser_std_vector_builtin_types;
pub mod reflection_ser_std_vector_custom_types;
pub mod reflection_world_ser_deser;

pub fn run_all() {
    entity_type::main();
    member_ranges::main();
    reflection_basics::main();
    reflection_basics_bitmask::main();
    reflection_basics_deserialize::main();
    reflection_basics_json::main();
    reflection_basics_simple_enum::main();
    reflection_nested_set_member::main();
    reflection_nested_struct::main();
    reflection_query_to_custom_json::main();
    reflection_query_to_json::main();
    reflection_runtime_component::main();
    reflection_runtime_nested_component::main();
    reflection_ser_opaque_type::main();
    reflection_ser_std_string::main();
    reflection_ser_std_vector_builtin_types::main();
    reflection_ser_std_vector_custom_types::main();
    reflection_world_ser_deser::main();
}
