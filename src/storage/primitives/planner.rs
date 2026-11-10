//! 缓存感知自适应子块规划器 (`SubchunkPlanner`)
//!
//! # 性能设计与文献支撑
//!
//! 1. **MonetDB/X100 向量化缓存驻留模型 (Boncz, Zukowski, Nes, CIDR 2005 & Zukowski Ph.D. 2009)**:
//!    超标量现代 CPU 处理列式数据时，若批次（Vector / Chunk）过大超出 L1/L2 缓存容量，
//!    CPU 硬件流水线将因访存停顿（Memory Stall）而陷入极低 IPC（Instructions Per Cycle）；
//!    若批次过小，则函数调用与循环调度开销占据主导。
//!    最佳吞吐平衡点是将单核每次处理的活跃列内存工作集（Working Set）严格约束在 CPU L1/L2 缓存之内（通常 16KB~64KB）。
//!
//! 2. **行宽感知动态对齐公式**:
//!    设当前查询所触达的所有组件列宽度之和为：
//!    $$W = \sum_{i=1}^k \text{sizeof}(C_i) \quad (\text{bytes})$$
//!    目标缓存容量为 $\text{TargetBytes}$（默认 32KB = 32768 字节）。
//!    子块行数 $N_{\text{rows}}$ 必须按 64 行对齐量化（与 `RowBitMask` 机器字和 `ChunkEpochs` 几何对齐）：
//!    $$N_{\text{ideal}} = \left\lfloor \frac{\text{TargetBytes}}{W \times 64} \right\rfloor \times 64$$
//!    $$N_{\text{rows}} = \text{clamp}(N_{\text{ideal}},\ 64,\ 32768)$$
//!
//! 3. **极值场景保障**:
//!    - **极窄表 / 纯标签 (ZST)**：$W = 0$，自适应采用大子块（16384 行），极大摊薄 Rayon 多核调度开销；
//!    - **极宽表 (如 $W = 512\text{B}$)**：自适应收缩至 64 行（$64 \times 512\text{B} = 32\text{KB}$），
//!      彻底避免原硬编码 4096 行占用 2MB 导致 L1/L2 缓存抖动（Cache Thrashing）。

use std::ops::Range;
use std::sync::OnceLock;

/// 回退默认 CPU L1/L2 缓存驻留目标工作集容量：32KB（32768 字节）
pub const DEFAULT_TARGET_BYTES: usize = 32 * 1024;

/// 子块最小行数：64 行（对应单个 64 位机器字与 64B/128B 缓存行整数倍）
pub const MIN_SUBCHUNK_ROWS: usize = 64;

/// 子块最大行数：32768 行（防止过大造成任务粗粒度负载不均衡）
pub const MAX_SUBCHUNK_ROWS: usize = 32768;

static DETECTED_L1D_BYTES: OnceLock<usize> = OnceLock::new();

/// 解析操作系统缓存容量字符串（如 `"32K"`、`"48K"`、`"128K"`、`"1M"` 或纯字节数）
#[inline]
pub(crate) fn parse_cache_size_str(raw: &str) -> Option<usize> {
    let s = raw.trim();
    if s.is_empty() {
        return None;
    }
    if let Some(num) = s.strip_suffix(['K', 'k']) {
        num.trim().parse::<usize>().ok()?.checked_mul(1024)
    } else if let Some(num) = s.strip_suffix(['M', 'm']) {
        num.trim().parse::<usize>().ok()?.checked_mul(1024 * 1024)
    } else if let Some(num) = s.strip_suffix(['B', 'b']) {
        num.trim().parse::<usize>().ok()
    } else {
        s.parse::<usize>().ok()
    }
}

fn probe_hardware_l1d_bytes() -> Option<usize> {
    #[cfg(target_os = "linux")]
    {
        for idx in 0..4 {
            let base = format!("/sys/devices/system/cpu/cpu0/cache/index{idx}");
            let level = std::fs::read_to_string(format!("{base}/level")).ok()?;
            if level.trim() != "1" {
                continue;
            }
            let kind = std::fs::read_to_string(format!("{base}/type")).unwrap_or_default();
            let kind_trimmed = kind.trim();
            if kind_trimmed.eq_ignore_ascii_case("Data")
                || kind_trimmed.eq_ignore_ascii_case("Unified")
            {
                if let Ok(size_str) = std::fs::read_to_string(format!("{base}/size")) {
                    if let Some(bytes) = parse_cache_size_str(&size_str) {
                        if bytes >= 4096 {
                            return Some(bytes);
                        }
                    }
                }
            }
        }
    }
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    {
        // Apple Silicon 统一内存架构：P 核 L1D 为 128KB，E 核 L1D 为 64KB，取 64KB 保守下界
        return Some(64 * 1024);
    }
    None
}

/// 探测当前主机 CPU 的 L1 数据缓存容量（字节），结果通过 `OnceLock` 进程级缓存（首次探测后 0 系统调用开销）。
/// 若当前平台不支持探测或处于受限沙箱环境，自动回退至 [`DEFAULT_TARGET_BYTES`]（32KB）。
#[inline]
pub fn detect_l1d_cache_bytes() -> usize {
    *DETECTED_L1D_BYTES.get_or_init(|| {
        probe_hardware_l1d_bytes()
            .map(|b| b.clamp(16 * 1024, 256 * 1024))
            .unwrap_or(DEFAULT_TARGET_BYTES)
    })
}

/// 缓存感知自适应子块规划器
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SubchunkPlanner {
    target_bytes: usize,
}

impl Default for SubchunkPlanner {
    #[inline]
    fn default() -> Self {
        Self::hardware_detected()
    }
}

impl SubchunkPlanner {
    /// 使用硬件自动探测的 L1D 缓存容量（[`detect_l1d_cache_bytes`]）创建规划器
    #[inline]
    pub fn hardware_detected() -> Self {
        Self {
            target_bytes: detect_l1d_cache_bytes(),
        }
    }

    /// 使用指定的单核目标缓存字节容量（例如 16KB、32KB 或 64KB）创建规划器
    #[inline]
    pub fn new(target_bytes: usize) -> Self {
        Self {
            target_bytes: target_bytes.clamp(4096, 256 * 1024),
        }
    }

    /// 当前规划器的目标缓存容量（字节）
    #[inline(always)]
    pub fn target_bytes(&self) -> usize {
        self.target_bytes
    }

    /// 根据访问列的总行宽 $W$ 计算 64 行对齐的最佳子块行数
    #[inline]
    pub fn subchunk_rows(&self, row_width: usize) -> usize {
        if row_width == 0 {
            // 纯 Tag 或 ZST 组件，无需载荷内存，使用较大块分摊调度开销
            return 16384;
        }

        let chunk_bytes_64 = row_width.saturating_mul(MIN_SUBCHUNK_ROWS);
        if chunk_bytes_64 >= self.target_bytes {
            // 宽表：单个 64 行块已达到或超过目标缓存大小，返回最小粒度 64 行
            return MIN_SUBCHUNK_ROWS;
        }

        let num_64_blocks = self.target_bytes / chunk_bytes_64;
        let rows = num_64_blocks * MIN_SUBCHUNK_ROWS;
        rows.clamp(MIN_SUBCHUNK_ROWS, MAX_SUBCHUNK_ROWS)
    }

    /// 为指定行宽与总行数生成切块迭代器
    #[inline]
    pub fn plan_subchunks(&self, row_width: usize, total_rows: usize) -> SubchunkRangeIter {
        let chunk_size = self.subchunk_rows(row_width);
        SubchunkRangeIter {
            total_rows,
            chunk_size,
            cur: 0,
        }
    }
}

/// 自适应切块区间迭代器
#[derive(Debug, Clone)]
pub struct SubchunkRangeIter {
    total_rows: usize,
    chunk_size: usize,
    cur: usize,
}

impl Iterator for SubchunkRangeIter {
    type Item = Range<usize>;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.cur >= self.total_rows {
            return None;
        }
        let start = self.cur;
        let end = (start + self.chunk_size).min(self.total_rows);
        self.cur = end;
        Some(start..end)
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        if self.cur >= self.total_rows {
            (0, Some(0))
        } else {
            let rem = self.total_rows - self.cur;
            let count = rem.div_ceil(self.chunk_size);
            (count, Some(count))
        }
    }
}

impl ExactSizeIterator for SubchunkRangeIter {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_planner_subchunk_rows_formula() {
        let planner = SubchunkPlanner::new(DEFAULT_TARGET_BYTES); // 32KB 基准验证

        // 1. ZST (0 字节)
        assert_eq!(planner.subchunk_rows(0), 16384);

        // 2. 窄表: 4 字节 (如单一 f32)
        // 32KB / 4 = 8192 行 (8192 是 64 的倍数)
        assert_eq!(planner.subchunk_rows(4), 8192);

        // 3. 中等表: 16 字节 (如 4 个 f32 / Vec4)
        // 32KB / 16 = 2048 行
        assert_eq!(planner.subchunk_rows(16), 2048);

        // 4. 常见物理表: 64 字节 (如 Transform 矩阵 + 刚体线速度/角速度)
        // 32KB / 64 = 512 行
        assert_eq!(planner.subchunk_rows(64), 512);

        // 5. 宽表: 256 字节
        // 32KB / 256 = 128 行
        assert_eq!(planner.subchunk_rows(256), 128);

        // 6. 极宽表: 1024 字节
        // 单个 64 行块需要 64KB，已超过 32KB -> 钳位至最小 64 行
        assert_eq!(planner.subchunk_rows(1024), 64);

        // 7. 硬件探测与解析验证
        assert_eq!(parse_cache_size_str("32K\n"), Some(32 * 1024));
        assert_eq!(parse_cache_size_str("48K"), Some(48 * 1024));
        assert_eq!(parse_cache_size_str("128K"), Some(128 * 1024));
        let hw = SubchunkPlanner::default();
        assert!(hw.target_bytes() >= 16 * 1024 && hw.target_bytes() <= 256 * 1024);
        assert_eq!(hw.subchunk_rows(32) % 64, 0);
    }

    #[test]
    fn test_planner_range_iter() {
        let planner = SubchunkPlanner::new(DEFAULT_TARGET_BYTES);
        // 64 字节/行 -> chunk_size = 512
        let iter = planner.plan_subchunks(64, 1200);
        let ranges: Vec<Range<usize>> = iter.collect();

        assert_eq!(ranges.len(), 3);
        assert_eq!(ranges[0], 0..512);
        assert_eq!(ranges[1], 512..1024);
        assert_eq!(ranges[2], 1024..1200);
    }
}
