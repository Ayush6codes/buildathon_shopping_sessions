// Diagnostic probe for the scoring machine. Prints environment + micro-benchmarks to stderr.
use rayon::prelude::*;
use std::time::Instant;

#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

fn cat(p: &str) -> String {
    std::fs::read_to_string(p).unwrap_or_else(|e| format!("<{}>", e))
}

fn main() {
    eprintln!("=== DIAG ===");
    let cpuinfo = cat("/proc/cpuinfo");
    let model = cpuinfo.lines().find(|l| l.starts_with("model name")).unwrap_or("");
    let ncpu_info = cpuinfo.lines().filter(|l| l.starts_with("processor")).count();
    eprintln!("{} | /proc/cpuinfo processors={}", model, ncpu_info);
    let flags = cpuinfo.lines().find(|l| l.starts_with("flags")).unwrap_or("");
    eprintln!("avx512bw={} avx2={} bmi2={}", flags.contains("avx512bw"), flags.contains("avx2"), flags.contains("bmi2"));
    eprintln!("available_parallelism={:?} NCPU={:?}", std::thread::available_parallelism(), std::env::var("NCPU"));
    eprintln!("cgroup cpu.max={} memory.max={}", cat("/sys/fs/cgroup/cpu.max").trim(), cat("/sys/fs/cgroup/memory.max").trim());
    eprintln!("THP enabled={} defrag={}", cat("/sys/kernel/mm/transparent_hugepage/enabled").trim(), cat("/sys/kernel/mm/transparent_hugepage/defrag").trim());
    let mi = cat("/proc/meminfo");
    for l in mi.lines().filter(|l| l.starts_with("MemTotal") || l.starts_with("Cached") || l.starts_with("AnonHuge") || l.starts_with("MemAvailable")) {
        eprintln!("  {}", l);
    }
    let mounts = cat("/proc/mounts");
    for l in mounts.lines().filter(|l| l.contains(" /data ") || l.contains(" /scratch ") || l.contains(" /output ")) {
        eprintln!("  mount: {}", l);
    }
    let nt: usize = std::env::var("NCPU").ok().and_then(|s| s.parse().ok()).unwrap_or(8);
    rayon::ThreadPoolBuilder::new().num_threads(nt).build_global().ok();

    // fresh-memory first-touch (page fault) throughput, 8 threads, 2 GB total
    for round in 0..2 {
        let t = Instant::now();
        let bufs: Vec<Vec<u8>> = (0..nt)
            .into_par_iter()
            .map(|_| {
                let mut v: Vec<u8> = Vec::with_capacity(256 << 20);
                unsafe { v.set_len(256 << 20) };
                for i in (0..v.len()).step_by(4096) {
                    v[i] = 1;
                }
                v
            })
            .collect();
        let s: usize = bufs.iter().map(|b| b[4096] as usize).sum();
        eprintln!("first-touch {} x 256MB round {}: {:?} ({})", nt, round, t.elapsed(), s);
        drop(bufs);
    }
    // memory bandwidth: parallel copy 2 GB
    {
        let src: Vec<Vec<u8>> = (0..nt).into_par_iter().map(|_| vec![7u8; 256 << 20]).collect();
        let t = Instant::now();
        let dst: Vec<Vec<u8>> = src.par_iter().map(|s| s.clone()).collect();
        eprintln!("copy {} x 256MB: {:?} ({})", nt, t.elapsed(), dst[0][5]);
        let t = Instant::now();
        let x: u64 = src.par_iter().map(|s| s.iter().step_by(64).map(|&b| b as u64).sum::<u64>()).sum();
        eprintln!("read-scan {} x 256MB (stride 64): {:?} ({})", nt, t.elapsed(), x);
    }
    // single vs multi-thread compute scaling
    for th in [1usize, 2, 4, 8] {
        let t = Instant::now();
        let r: u64 = (0..th)
            .into_par_iter()
            .map(|i| {
                let mut x = i as u64 + 1;
                for _ in 0..200_000_000u64 {
                    x = (x ^ (x >> 29)).wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                }
                x
            })
            .sum();
        eprintln!("compute {} threads x 200M mul: {:?} ({})", th, t.elapsed(), r & 1);
    }
    // read speed of /data files (page cache?) via mmap populate and via read()
    let data = std::env::var("DATA_DIR").unwrap_or("/data".into());
    let mut files = Vec::new();
    for sub in ["events", "clicks", "links"] {
        if let Ok(rd) = std::fs::read_dir(format!("{}/{}", data, sub)) {
            for e in rd.flatten() {
                files.push(e.path());
            }
        }
    }
    let t = Instant::now();
    let tot: usize = files.par_iter().map(|p| std::fs::read(p).map(|v| v.len()).unwrap_or(0)).sum();
    eprintln!("read() {} files {} MB: {:?}", files.len(), tot >> 20, t.elapsed());
    eprintln!("=== END DIAG ===");
}
