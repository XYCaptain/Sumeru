use std::fs;
use std::path::{Path, PathBuf};

fn collect_rs_files(dir: &Path, out: &mut Vec<PathBuf>) {
    if let Ok(entries) = fs::read_dir(dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                collect_rs_files(&path, out);
            } else if path.extension().and_then(|e| e.to_str()) == Some("rs") {
                out.push(path);
            }
        }
    }
}

#[test]
fn verify_zero_pro_and_internal_identifier_leakage() {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let mut files = Vec::new();
    collect_rs_files(&manifest_dir.join("src"), &mut files);
    collect_rs_files(&manifest_dir.join("examples"), &mut files);

    let forbidden = [
        "Segmented",
        "SegmentList",
        "SegmentSlice",
        "BufferPool",
        "BorrowTracker",
        "LeaseHandle",
        "StoreShared",
        "SharedGroupedQuery",
        "ContiguousView",
        "SharedRangeSession",
        "DeltaReceipt",
        "CompactOldValues",
        "HistoryManager",
        "StagedSession",
        "PartitionSpace",
        "SharedSpace",
        "ens_compute_",
    ];

    for file in files {
        let content = fs::read_to_string(&file).expect("failed to read source file");
        for term in &forbidden {
            assert!(
                !content.contains(term),
                "Forbidden Pro/Internal identifier `{term}` found in open-source file {}",
                file.display()
            );
        }
    }

    let cargo_toml =
        fs::read_to_string(manifest_dir.join("Cargo.toml")).expect("failed to read Cargo.toml");
    assert!(
        !cargo_toml.contains("workspace = true"),
        "Open-source Cargo.toml must be standalone and must not contain `workspace = true`"
    );
    assert!(
        !cargo_toml.contains("ens_"),
        "Open-source Cargo.toml must not depend on any internal `ens_*` crate"
    );
}
