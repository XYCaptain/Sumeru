pub mod relationships_basics;
pub mod relationships_component_data;
pub mod relationships_enum;
pub mod relationships_exclusive;
pub mod relationships_symmetric;

pub fn run_all() {
    relationships_basics::main();
    relationships_component_data::main();
    relationships_enum::main();
    relationships_exclusive::main();
    relationships_symmetric::main();
}
