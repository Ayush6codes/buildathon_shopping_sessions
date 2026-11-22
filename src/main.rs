// Shopping Sessions pipeline: exact implementation of rules.md, parallel Rust.
use hashbrown::HashTable;
use memchr::memchr;
use rayon::prelude::*;
use std::fs::File;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};
use std::time::Instant;
use xxhash_rust::xxh3::xxh3_64;

#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

type S = &'static [u8];

const DAY: i64 = 86_400_000;
const HOUR24: i64 = 86_400_000;
const GAP: i64 = 1_800_000;
const WEEK: i64 = 604_800_000;
const NP: usize = 256; // partitions
const NP_SHIFT: u32 = 56;

macro_rules! tlog {
    ($t:expr, $($arg:tt)*) => {
        if std::env::var_os("SS_QUIET").is_none() {
            eprintln!("[{:>8.1} ms | cpu {:>7.0} ms] {}", $t.elapsed().as_secs_f64() * 1000.0, cpu_ms(), format!($($arg)*));
        }
    };
}

fn cpu_ms() -> f64 {
    let s = std::fs::read_to_string("/proc/self/stat").unwrap_or_default();
    let r = s.rsplit(')').next().unwrap_or("");
    let f: Vec<&str> = r.split_whitespace().collect();
    if f.len() > 13 {
        let u: f64 = f[11].parse().unwrap_or(0.0);
        let st: f64 = f[12].parse().unwrap_or(0.0);
        (u + st) * 10.0
    } else {
        0.0
    }
}

// ---------------------------------------------------------------- files

fn map_file(p: &Path) -> S {
    let f = File::open(p).expect("open");
    let len = f.metadata().map(|m| m.len()).unwrap_or(0);
    if len == 0 {
        return &[];
    }
    let m = unsafe { memmap2::MmapOptions::new().populate().map(&f).expect("mmap") };
    #[cfg(unix)]
    {
        let _ = m.advise(memmap2::Advice::Sequential);
    }
    let m: &'static memmap2::Mmap = Box::leak(Box::new(m));
    &m[..]
}

fn list_csv(dir: &Path) -> Vec<PathBuf> {
    let mut v = Vec::new();
    if let Ok(rd) = std::fs::read_dir(dir) {
        for e in rd.flatten() {
            let name = e.file_name();
            let n = name.to_string_lossy();
            if n.starts_with('.') || !n.ends_with(".csv") {
                continue;
            }
            let p = e.path();
            if p.is_file() {
                v.push(p);
            }
        }
    }
    v.sort();
    v
}

/// Split all files into newline-aligned chunks; the header line of each file is skipped.
fn chunks_of(bufs: &[S], target: usize) -> Vec<S> {
    let mut out = Vec::new();
    for &b in bufs {
        // skip header
        let start = match memchr(b'\n', b) {
            Some(p) => p + 1,
            None => continue,
        };
        let mut s = start;
        while s < b.len() {
            let mut e = s.saturating_add(target).min(b.len());
            if e < b.len() {
                match memchr(b'\n', &b[e..]) {
                    Some(p) => e = e + p + 1,
                    None => e = b.len(),
                }
            }
            out.push(&b[s..e]);
            s = e;
        }
    }
    out
}

/// Iterate lines of a chunk (split on '\n' only). A trailing empty piece is not a line.
#[inline]
fn for_lines(c: S, mut f: impl FnMut(S)) {
    let mut s = 0;
    while s < c.len() {
        match memchr(b'\n', &c[s..]) {
            Some(p) => {
                f(&c[s..s + p]);
                s += p + 1;
            }
            None => {
                f(&c[s..]);
                break;
            }
        }
    }
}

/// Split a line into exactly N fields following the CSV dialect; None if malformed.
#[inline]
fn split<const N: usize>(line: S) -> Option<[S; N]> {
    let mut out: [S; N] = [&line[0..0]; N];
    let mut start = 0;
    let mut i = 0;
    loop {
        match memchr(b',', &line[start..]) {
            Some(p) => {
                if i >= N - 1 {
                    return None;
                }
                out[i] = &line[start..start + p];
                i += 1;
                start += p + 1;
            }
            None => {
                if i != N - 1 {
                    return None;
                }
                out[i] = &line[start..];
                break;
            }
        }
    }
    if memchr(b'"', line).is_some() {
        for f in out.iter_mut() {
            if memchr(b'"', f).is_some() {
                let n = f.len();
                if n < 2 || f[0] != b'"' || f[n - 1] != b'"' || memchr(b'"', &f[1..n - 1]).is_some() {
                    return None;
                }
                *f = &f[1..n - 1];
            }
        }
    }
    Some(out)
}


// ---------------------------------------------------------------- SIMD line/field scanner

#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn masks64(p: *const u8) -> (u64, u64, u64) {
    use std::arch::x86_64::*;
    let a = _mm256_loadu_si256(p as *const __m256i);
    let b = _mm256_loadu_si256(p.add(32) as *const __m256i);
    let comma = _mm256_set1_epi8(b',' as i8);
    let nl = _mm256_set1_epi8(b'\n' as i8);
    let qt = _mm256_set1_epi8(b'"' as i8);
    let m = |x: __m256i, y: __m256i| -> u64 {
        let lo = _mm256_movemask_epi8(_mm256_cmpeq_epi8(x, y)) as u32 as u64;
        lo
    };
    let c = m(a, comma) | (m(b, comma) << 32);
    let n = m(a, nl) | (m(b, nl) << 32);
    let q = m(a, qt) | (m(b, qt) << 32);
    (c, n, q)
}

#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn m128(p: *const u8) -> (u128, u128, u128) {
    use std::arch::x86_64::*;
    let comma = _mm256_set1_epi8(b',' as i8);
    let nl = _mm256_set1_epi8(b'\n' as i8);
    let qt = _mm256_set1_epi8(b'"' as i8);
    let mut c: u128 = 0;
    let mut n: u128 = 0;
    let mut q: u128 = 0;
    for i in 0..4 {
        let v = _mm256_loadu_si256(p.add(32 * i) as *const __m256i);
        c |= (_mm256_movemask_epi8(_mm256_cmpeq_epi8(v, comma)) as u32 as u128) << (32 * i);
        n |= (_mm256_movemask_epi8(_mm256_cmpeq_epi8(v, nl)) as u32 as u128) << (32 * i);
        q |= (_mm256_movemask_epi8(_mm256_cmpeq_epi8(v, qt)) as u32 as u128) << (32 * i);
    }
    (c, n, q)
}

#[inline(always)]
fn emit_line<const N: usize>(line: S, commas: &[usize; N], ncomma: usize, qc: u32, f: &mut impl FnMut([S; N])) {
    if ncomma != N - 1 {
        return;
    }
    let mut out: [S; N] = [&line[0..0]; N];
    let mut st = 0usize;
    for j in 0..N {
        let en = if j < N - 1 { commas[j] } else { line.len() };
        out[j] = unsafe { line.get_unchecked(st..en) };
        st = en + 1;
    }
    if qc != 0 {
        let mut g = 0u32;
        for fld in out.iter_mut() {
            let l = fld.len();
            if l >= 2 && fld[0] == b'"' && fld[l - 1] == b'"' {
                g += 2;
                *fld = &fld[1..l - 1];
            }
        }
        if g != qc {
            return;
        }
    }
    f(out);
}

/// Scan a chunk; for every well-formed line with exactly N fields call f with unquoted fields.
#[inline(always)]
fn scan<const N: usize>(c: S, mut f: impl FnMut([S; N])) {
    let n = c.len();
    let base = c.as_ptr();
    let mut pos = 0usize;
    let mut pad = [0u8; 128];
    let mut commas = [0usize; N];
    while pos < n {
        let rem = n - pos;
        let (cm, nm, qm) = unsafe {
            if rem >= 128 {
                m128(base.add(pos))
            } else {
                pad = [0u8; 128];
                std::ptr::copy_nonoverlapping(base.add(pos), pad.as_mut_ptr(), rem);
                m128(pad.as_ptr())
            }
        };
        if nm != 0 || rem <= 128 {
            // line end within window (or at buffer end)
            let le = if nm != 0 { nm.trailing_zeros() as usize } else { rem };
            if le > rem {
                // newline bit beyond data can't happen (pad zero), defensive
            }
            let below: u128 = if le >= 128 { !0 } else { (1u128 << le) - 1 };
            let mut cb = cm & below;
            let nc = cb.count_ones() as usize;
            let qc = (qm & below).count_ones();
            if nc == N - 1 {
                for j in 0..N - 1 {
                    commas[j] = cb.trailing_zeros() as usize;
                    cb &= cb - 1;
                }
            }
            let line = unsafe { std::slice::from_raw_parts(base.add(pos), le) };
            emit_line::<N>(line, &commas, nc, qc, &mut f);
            pos += le + 1;
        } else {
            // long line: scalar fallback
            let le = match memchr(b'\n', &c[pos..]) {
                Some(p) => p,
                None => n - pos,
            };
            let line = &c[pos..pos + le];
            let mut nc = 0usize;
            let mut qc = 0u32;
            for (i, &b) in line.iter().enumerate() {
                if b == b',' {
                    if nc < N - 1 {
                        commas[nc] = i;
                    }
                    nc += 1;
                } else if b == b'"' {
                    qc += 1;
                }
            }
            emit_line::<N>(line, &commas, nc, qc, &mut f);
            pos += le + 1;
        }
    }
}

// ---------------------------------------------------------------- timestamps

#[inline]
fn dig(c: u8) -> Option<i64> {
    if c.is_ascii_digit() {
        Some((c - b'0') as i64)
    } else {
        None
    }
}
#[inline]
fn d2(b: &[u8], i: usize) -> Option<i64> {
    Some(dig(b[i])? * 10 + dig(b[i + 1])?)
}

fn days_from_civil(y: i64, m: i64, d: i64) -> i64 {
    let y = if m <= 2 { y - 1 } else { y };
    let era = if y >= 0 { y } else { y - 399 } / 400;
    let yoe = y - era * 400;
    let mp = (m + 9) % 12;
    let doy = (153 * mp + 2) / 5 + d - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146097 + doe - 719468
}

fn is_leap(y: i64) -> bool {
    (y % 4 == 0 && y % 100 != 0) || y % 400 == 0
}

static MONTH_DAYS: std::sync::OnceLock<Vec<i64>> = std::sync::OnceLock::new();
const Y0: i64 = 1900;
const NY: i64 = 400;
fn month_table() -> &'static Vec<i64> {
    MONTH_DAYS.get_or_init(|| {
        let mut v = Vec::with_capacity((NY * 12) as usize);
        for y in Y0..Y0 + NY {
            for m in 1..=12 {
                v.push(days_from_civil(y, m, 1));
            }
        }
        v
    })
}

#[inline(always)]
fn dg(c: u8) -> u32 {
    c.wrapping_sub(b'0') as u32
}

/// Parse ts into epoch ms. None = bad (including an F3 value too large for i64).
#[inline]
fn parse_ts(b: &[u8]) -> Option<i64> {
    let n = b.len();
    if n == 20 || n == 25 {
        if b[4] == b'-' {
            return parse_iso(b);
        }
    }
    parse_f3(b)
}

#[inline(always)]
fn parse_iso(b: &[u8]) -> Option<i64> {
    let n = b.len();
    let a = u64::from_le_bytes(b[0..8].try_into().unwrap()); // YYYY-MM-
    let c = u64::from_le_bytes(b[8..16].try_into().unwrap()); // DDTHH:MM
    let e = u32::from_le_bytes(b[16..20].try_into().unwrap()); // :SSZ or :SS+
    // separators
    const SA_MASK: u64 = 0xFF00_00FF_0000_0000;
    const SA_VAL: u64 = (b'-' as u64) << 32 | (b'-' as u64) << 56;
    const SC_MASK: u64 = 0x0000_FF00_00FF_0000;
    const SC_VAL: u64 = (b'T' as u64) << 16 | (b':' as u64) << 40;
    if (a & SA_MASK) != SA_VAL || (c & SC_MASK) != SC_VAL || (e & 0xFF) != b':' as u32 {
        return None;
    }
    let ta = a ^ 0x3030_3030_3030_3030;
    let tc = c ^ 0x3030_3030_3030_3030;
    let te = e ^ 0x3030_3030;
    const DA: u64 = 0x00FF_FF00_FFFF_FFFF; // digit bytes in a: 0,1,2,3,5,6
    const DC: u64 = 0xFFFF_00FF_FF00_FFFF; // digit bytes in c: 0,1,3,4,6,7
    const DE: u32 = 0x00FF_FF00; // bytes 1,2
    let bad_a = ((ta & 0xF0F0_F0F0_F0F0_F0F0) | (ta.wrapping_add(0x0606_0606_0606_0606) & 0x1010_1010_1010_1010)) & DA;
    let bad_c = ((tc & 0xF0F0_F0F0_F0F0_F0F0) | (tc.wrapping_add(0x0606_0606_0606_0606) & 0x1010_1010_1010_1010)) & DC;
    let bad_e = ((te & 0xF0F0_F0F0) | (te.wrapping_add(0x0606_0606) & 0x1010_1010)) & DE;
    if (bad_a | bad_c) != 0 || bad_e != 0 {
        return None;
    }
    let g = |x: u64, i: u32| -> i64 { ((x >> (8 * i)) & 0xFF) as i64 };
    let y = g(ta, 0) * 1000 + g(ta, 1) * 100 + g(ta, 2) * 10 + g(ta, 3);
    let mo = g(ta, 5) * 10 + g(ta, 6);
    let d = g(tc, 0) * 10 + g(tc, 1);
    let h = g(tc, 3) * 10 + g(tc, 4);
    let mi = g(tc, 6) * 10 + g(tc, 7);
    let s = (((te >> 8) & 0xFF) * 10 + ((te >> 16) & 0xFF)) as i64;
    if mo < 1 || mo > 12 || h > 23 || mi > 59 || s > 59 || d < 1 {
        return None;
    }
    let dim = match mo {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        4 | 6 | 9 | 11 => 30,
        _ => {
            if is_leap(y) {
                29
            } else {
                28
            }
        }
    };
    if d > dim {
        return None;
    }
    let days = if y >= Y0 && y < Y0 + NY { month_table()[((y - Y0) * 12 + mo - 1) as usize] + d - 1 } else { days_from_civil(y, mo, d) };
    let mut ms = (days * 86400 + h * 3600 + mi * 60 + s) * 1000;
    if n == 20 {
        if b[19] != b'Z' {
            return None;
        }
    } else {
        let sign = match b[19] {
            b'+' => 1,
            b'-' => -1,
            _ => return None,
        };
        if b[22] != b':' {
            return None;
        }
        let (p, q, r, t) = (dg(b[20]), dg(b[21]), dg(b[23]), dg(b[24]));
        if p > 9 || q > 9 || r > 9 || t > 9 {
            return None;
        }
        let oh = (p * 10 + q) as i64;
        let om = (r * 10 + t) as i64;
        if oh > 23 || om > 59 {
            return None;
        }
        ms -= sign * (oh * 3600 + om * 60) * 1000;
    }
    Some(ms)
}

#[inline(always)]
fn parse_f3(b: &[u8]) -> Option<i64> {
    if b.is_empty() {
        return None;
    }
    let mut i = 0;
    while i < b.len() && b[i] == b'0' {
        i += 1;
    }
    let r = &b[i..];
    let mut v: u64 = 0;
    for &c in r {
        let d = dg(c);
        if d > 9 {
            return None;
        }
        v = v.wrapping_mul(10).wrapping_add(d as u64);
    }
    if r.len() > 19 || v > i64::MAX as u64 {
        return None;
    }
    Some(v as i64)
}

#[inline]
fn parse_price(b: &[u8]) -> Option<u32> {
    if b.is_empty() || !b.iter().all(|c| c.is_ascii_digit()) {
        return None;
    }
    let mut i = 0;
    while i < b.len() && b[i] == b'0' {
        i += 1;
    }
    let r = &b[i..];
    if r.is_empty() || r.len() > 8 {
        return None;
    }
    let mut v: u32 = 0;
    for &c in r {
        v = v * 10 + (c - b'0') as u32;
    }
    if v >= 1 && v <= 10_000_000 {
        Some(v)
    } else {
        None
    }
}

// ---------------------------------------------------------------- lookup tables

struct StrMap<V: Copy> {
    t: HashTable<(S, V)>,
}
impl<V: Copy> StrMap<V> {
    fn new(cap: usize) -> Self {
        StrMap { t: HashTable::with_capacity(cap) }
    }
    #[inline]
    fn get_h(&self, h: u64, k: &[u8]) -> Option<V> {
        self.t.find(h, |e| e.0 == k).map(|e| e.1)
    }
    #[inline]
    fn get(&self, k: &[u8]) -> Option<V> {
        self.get_h(xxh3_64(k), k)
    }
    /// insert if absent; returns existing or new value
    fn intern(&mut self, k: S, v: V) -> (V, bool) {
        let h = xxh3_64(k);
        if let Some(e) = self.t.find(h, |e| e.0 == k) {
            return (e.1, false);
        }
        self.t.insert_unique(h, (k, v), |e| xxh3_64(e.0));
        (v, true)
    }
}

// ---------------------------------------------------------------- software write-combining scatter
const SWC: usize = 16;
struct Scatter<T: Copy> {
    buf: Vec<[std::mem::MaybeUninit<T>; SWC]>,
    cnt: [u8; NP],
    out: Vec<Vec<T>>,
}
impl<T: Copy> Scatter<T> {
    fn new(est: usize) -> Self {
        Scatter {
            buf: (0..NP).map(|_| [std::mem::MaybeUninit::uninit(); SWC]).collect(),
            cnt: [0; NP],
            out: (0..NP).map(|_| Vec::with_capacity(est + est / 4)).collect(),
        }
    }
    #[inline(always)]
    fn push(&mut self, p: usize, v: T) {
        unsafe {
            let c = *self.cnt.get_unchecked(p) as usize;
            let b = self.buf.get_unchecked_mut(p);
            b.get_unchecked_mut(c).write(v);
            if c + 1 == SWC {
                let o = self.out.get_unchecked_mut(p);
                o.extend_from_slice(std::slice::from_raw_parts(b.as_ptr() as *const T, SWC));
                *self.cnt.get_unchecked_mut(p) = 0;
            } else {
                *self.cnt.get_unchecked_mut(p) = (c + 1) as u8;
            }
        }
    }
    fn finish(mut self) -> Vec<Vec<T>> {
        for p in 0..NP {
            let c = self.cnt[p] as usize;
            if c > 0 {
                let b = &self.buf[p];
                let sl = unsafe { std::slice::from_raw_parts(b.as_ptr() as *const T, c) };
                self.out[p].extend_from_slice(sl);
            }
        }
        self.out
    }
}

// ---------------------------------------------------------------- records

#[derive(Clone, Copy)]
struct ERec {
    h: u64,
    ingest: i64,
    eid: S,
    uid: S,
    ts: i64,
    price: u32,
    cat: u16,
    typ: u8,
    valid: bool,
}

#[derive(Clone, Copy)]
struct PEv {
    nh: u64,
    ts: i64,
    name: S,
    price: u32,
    cat: u16,
    typ: u8,
}

#[derive(Clone, Copy)]
struct PCl {
    nh: u64,
    ts: i64,
    cid: S,
    camp: u16,
}

struct URow {
    name: S,
    day: i32,
    sessions: i64,
    events: i64,
    revenue: i64,
    top: i32, // -1 = null, else category index
}

struct Person {
    name: S,
    nh: u64,
}

struct Ctx {
    ws: i64,
    we: i64,
    prod: StrMap<u16>,
    ncat: usize, // including unknown at index ncat-1
    camp: StrMap<u16>,
}

struct LOcc {
    slot: u32,
    h: u64,
    h2: u64,
    s: S,
}
struct LEnt {
    h: u64,
    h2: u64,
    nh: u64,
    name: S,
}
#[derive(Clone, Copy)]
struct UEv {
    uh: u64,
    uh2: u64,
    uid: S,
    ts: i64,
    price: u32,
    cat: u16,
    typ: u8,
}
#[derive(Clone, Copy)]
struct UCl {
    uh: u64,
    uh2: u64,
    ts: i64,
    cid: S,
    camp: u16,
}
unsafe impl Send for ERec {}

#[inline]
fn hh(s: &[u8]) -> (u64, u64) {
    let v = xxhash_rust::xxh3::xxh3_128(s);
    (v as u64, (v >> 64) as u64)
}

fn read_manifest(p: &Path) -> (i64, i64) {
    let s = std::fs::read_to_string(p).expect("manifest");
    let get = |key: &str| -> i64 {
        let i = s.find(key).expect("key") + key.len();
        let rest = &s[i..];
        let q1 = rest.find('"').unwrap();
        let rest2 = &rest[q1 + 1..];
        let q2 = rest2.find('"').unwrap();
        parse_ts(rest2[..q2].as_bytes()).expect("manifest ts")
    };
    (get("\"window_start\""), get("\"window_end\""))
}

// union-find
fn find(p: &mut [u32], mut x: u32) -> u32 {
    while p[x as usize] != x {
        let gp = p[p[x as usize] as usize];
        p[x as usize] = gp;
        x = gp;
    }
    x
}

fn main() {
    let t0 = Instant::now();
    let args: Vec<String> = std::env::args().collect();
    let data_dir = std::env::var("DATA_DIR").ok().or_else(|| args.get(1).cloned()).unwrap_or("/data".into());
    let out_dir = std::env::var("OUTPUT_DIR").ok().or_else(|| args.get(2).cloned()).unwrap_or("/output".into());
    let nthreads: usize = std::env::var("NCPU").ok().and_then(|s| s.parse().ok()).unwrap_or_else(|| {
        std::thread::available_parallelism().map(|n| n.get()).unwrap_or(8)
    });
    rayon::ThreadPoolBuilder::new().num_threads(nthreads).build_global().ok();
    let data = PathBuf::from(&data_dir);
    let out = PathBuf::from(&out_dir);

    let (ws, we) = read_manifest(&data.join("manifest.json"));

    // ---- map all input files
    let ev_files = list_csv(&data.join("events"));
    let cl_files = list_csv(&data.join("clicks"));
    let ln_files = list_csv(&data.join("links"));
    let ev_bufs: Vec<S> = ev_files.par_iter().map(|p| map_file(p)).collect();
    let cl_bufs: Vec<S> = cl_files.par_iter().map(|p| map_file(p)).collect();
    let ln_bufs: Vec<S> = ln_files.par_iter().map(|p| map_file(p)).collect();
    let ev_total: usize = ev_bufs.iter().map(|b| b.len()).sum();
    let cl_total: usize = cl_bufs.iter().map(|b| b.len()).sum();
    tlog!(t0, "mapped: events {} files {} MB, clicks {} files {} MB, links {} files", ev_bufs.len(), ev_total >> 20, cl_bufs.len(), cl_total >> 20, ln_bufs.len());

    // ---- products & campaigns
    let pbuf = map_file(&data.join("products.csv"));
    let mut cat_names: Vec<S> = Vec::new();
    let mut cat_map: StrMap<u16> = StrMap::new(64);
    let mut prod: StrMap<u16> = StrMap::new(1 << 17);
    for c in chunks_of(&[pbuf], usize::MAX) {
        for_lines(c, |l| {
            if let Some([pid, cat]) = split::<2>(l) {
                let (ci, new) = cat_map.intern(cat, cat_names.len() as u16);
                if new {
                    cat_names.push(cat);
                }
                prod.intern(pid, ci);
            }
        });
    }
    cat_names.push(b"unknown");
    let ncat = cat_names.len();
    // rank for tie-breaks (byte-wise smallest name wins)
    let mut order: Vec<usize> = (0..ncat).collect();
    order.sort_by(|&a, &b| cat_names[a].cmp(cat_names[b]));
    let mut cat_rank = vec![0u32; ncat];
    for (r, &i) in order.iter().enumerate() {
        cat_rank[i] = r as u32;
    }

    let cbuf = map_file(&data.join("campaigns.csv"));
    let mut camp: StrMap<u16> = StrMap::new(4096);
    let mut camp_names: Vec<S> = Vec::new();
    let mut camp_chan: Vec<S> = Vec::new();
    for c in chunks_of(&[cbuf], usize::MAX) {
        for_lines(c, |l| {
            if let Some([cid, ch]) = split::<2>(l) {
                let (_, new) = camp.intern(cid, camp_names.len() as u16);
                if new {
                    camp_names.push(cid);
                    camp_chan.push(ch);
                }
            }
        });
    }
    let ncamp = camp_names.len();

    // ---- links (parallel, partitioned by id hash)
    let ln_chunks = chunks_of(&ln_bufs, 1 << 20);
    let pairs: Vec<Vec<(S, S)>> = ln_chunks
        .par_iter()
        .map(|&c| {
            let mut v = Vec::with_capacity(c.len() / 20);
            scan::<2>(c, |[a, b]| {
                if !a.is_empty() && !b.is_empty() {
                    v.push((a, b));
                }
            });
            v
        })
        .collect();
    let mut poff = Vec::with_capacity(pairs.len() + 1);
    let mut acc = 0usize;
    for v in &pairs {
        poff.push(acc);
        acc += v.len();
    }
    let npairs = acc;
    let occ: Vec<Vec<Vec<LOcc>>> = pairs
        .par_iter()
        .enumerate()
        .map(|(ci, v)| {
            let mut bk: Vec<Vec<LOcc>> = (0..NP).map(|_| Vec::with_capacity(v.len() * 2 / NP + 8)).collect();
            let base = poff[ci];
            for (k, &(x, y)) in v.iter().enumerate() {
                let (h, h2) = hh(x);
                bk[(h >> NP_SHIFT) as usize].push(LOcc { slot: ((base + k) * 2) as u32, h, h2, s: x });
                let (h, h2) = hh(y);
                bk[(h >> NP_SHIFT) as usize].push(LOcc { slot: ((base + k) * 2 + 1) as u32, h, h2, s: y });
            }
            bk
        })
        .collect();
    let lb: Vec<(Vec<(u64, u64, S)>, Vec<(u32, u32)>)> = (0..NP)
        .into_par_iter()
        .map(|bi| {
            let n: usize = occ.iter().map(|c| c[bi].len()).sum();
            let mut t: HashTable<(u64, u64, u32)> = HashTable::with_capacity(n);
            let mut ids: Vec<(u64, u64, S)> = Vec::with_capacity(n);
            let mut slots: Vec<(u32, u32)> = Vec::with_capacity(n);
            for c in &occ {
                for o in &c[bi] {
                    let local = match t.find(o.h.rotate_left(24), |x| x.0 == o.h && x.1 == o.h2) {
                        Some(x) => x.2,
                        None => {
                            let l = ids.len() as u32;
                            t.insert_unique(o.h.rotate_left(24), (o.h, o.h2, l), |x| x.0.rotate_left(24));
                            ids.push((o.h, o.h2, o.s));
                            l
                        }
                    };
                    slots.push((o.slot, local));
                }
            }
            (ids, slots)
        })
        .collect();
    drop(occ);
    let mut goff = vec![0u32; NP + 1];
    for bi in 0..NP {
        goff[bi + 1] = goff[bi] + lb[bi].0.len() as u32;
    }
    let nid = goff[NP] as usize;
    let mut ends = vec![0u32; npairs * 2];
    for bi in 0..NP {
        let g = goff[bi];
        for &(slot, l) in &lb[bi].1 {
            ends[slot as usize] = g + l;
        }
    }
    let mut parent: Vec<u32> = (0..nid as u32).collect();
    for k in 0..npairs {
        let ra = find(&mut parent, ends[2 * k]);
        let rb = find(&mut parent, ends[2 * k + 1]);
        if ra != rb {
            if ra < rb {
                parent[rb as usize] = ra;
            } else {
                parent[ra as usize] = rb;
            }
        }
    }
    for i in 0..nid {
        let r = find(&mut parent, i as u32);
        parent[i] = r;
    }
    let mut idstr: Vec<S> = Vec::with_capacity(nid);
    for bi in 0..NP {
        idstr.extend(lb[bi].0.iter().map(|x| x.2));
    }
    let best: Vec<AtomicU32> = (0..nid as u32).map(AtomicU32::new).collect();
    (0..nid).into_par_iter().for_each(|i| {
        let r = parent[i] as usize;
        if r == i {
            return;
        }
        let mut cur = best[r].load(Ordering::Relaxed);
        loop {
            if idstr[i] >= idstr[cur as usize] {
                break;
            }
            match best[r].compare_exchange_weak(cur, i as u32, Ordering::Relaxed, Ordering::Relaxed) {
                Ok(_) => break,
                Err(x) => cur = x,
            }
        }
    });
    let ltabs: Vec<HashTable<LEnt>> = (0..NP)
        .into_par_iter()
        .map(|bi| {
            let ids = &lb[bi].0;
            let mut t: HashTable<LEnt> = HashTable::with_capacity(ids.len());
            for (l, &(h, h2, _)) in ids.iter().enumerate() {
                let g = goff[bi] as usize + l;
                let r = parent[g] as usize;
                let name = idstr[best[r].load(Ordering::Relaxed) as usize];
                let nh = hh(name).0;
                t.insert_unique(h.rotate_left(24), LEnt { h, h2, nh, name }, |x| x.h.rotate_left(24));
            }
            t
        })
        .collect();
    tlog!(t0, "links: {} pairs, {} ids", npairs, nid);

    let ctx = Arc::new(Ctx { ws, we, prod, ncat, camp });

    // ---- events: parse and partition by event_id hash
    let ev_chunk_size = (ev_total / (nthreads * 6)).clamp(1 << 20, 64 << 20);
    let ev_chunks = chunks_of(&ev_bufs, ev_chunk_size);
    let parted: Vec<Vec<Vec<ERec>>> = ev_chunks
        .par_iter()
        .map(|&c| {
            let est = c.len() / 64 / NP + 16;
            let mut parts = Scatter::<ERec>::new(est);
            let ctx = &*ctx;
            scan::<7>(c, |f| {
                let [eid, uid, ts_s, ing_s, typ_s, pid, price_s] = f;
                if eid.is_empty() {
                    return;
                }
                let h = xxh3_64(eid);
                let ingest = parse_ts(ing_s).unwrap_or(i64::MAX);
                let ts = parse_ts(ts_s);
                let typ: u8 = match typ_s {
                    b"view" => 0,
                    b"cart" => 1,
                    b"purchase" => 2,
                    _ => 255,
                };
                let mut valid = !uid.is_empty() && typ != 255 && !pid.is_empty();
                let tsv = ts.unwrap_or(i64::MIN);
                if valid {
                    match ts {
                        Some(t) if t >= ctx.ws && t < ctx.we => {
                            if ingest.saturating_sub(t) > HOUR24 {
                                valid = false;
                            }
                        }
                        _ => valid = false,
                    }
                }
                let mut price = 0u32;
                let mut cat = 0u16;
                if valid && typ == 2 {
                    match parse_price(price_s) {
                        Some(p) => {
                            price = p;
                            cat = ctx.prod.get(pid).unwrap_or(ctx.ncat as u16 - 1);
                        }
                        None => valid = false,
                    }
                }
                parts.push((h >> NP_SHIFT) as usize, ERec { h, ingest, eid, uid, ts: tsv, price, cat, typ, valid });
            });
            parts.finish()
        })
        .collect();
    tlog!(t0, "events parsed: {} chunks", ev_chunks.len());

    // ---- dedup per eid partition -> survivors bucketed by uid hash
    let uevs: Vec<Vec<Vec<UEv>>> = (0..NP)
        .into_par_iter()
        .map(|pi| {
            let n: usize = parted.iter().map(|p| p[pi].len()).sum();
            let est = n / NP + 16;
            let mut outp = Scatter::<UEv>::new(est);
            let mut tab: HashTable<(u64, i64, *const ERec)> = HashTable::with_capacity(n);
            for p in &parted {
                for r in &p[pi] {
                    let rp = r as *const ERec;
                    let hh_ = r.h.rotate_left(24);
                    match tab.find_mut(hh_, |e| e.0 == r.h && unsafe { (*e.2).eid == r.eid }) {
                        Some(e) => {
                            if r.ingest < e.1 {
                                e.1 = r.ingest;
                                e.2 = rp;
                            }
                        }
                        None => {
                            tab.insert_unique(hh_, (r.h, r.ingest, rp), |e| e.0.rotate_left(24));
                        }
                    }
                }
            }
            for p in &parted {
                for r in &p[pi] {
                    if !r.valid {
                        continue;
                    }
                    let rp = r as *const ERec;
                    let win = tab.find(r.h.rotate_left(24), |e| e.0 == r.h && (e.2 == rp || unsafe { (*e.2).eid == r.eid }));
                    if let Some(e) = win {
                        if e.2 == rp {
                            let (uh, uh2) = hh(r.uid);
                            outp.push((uh >> NP_SHIFT) as usize, UEv { uh, uh2, uid: r.uid, ts: r.ts, price: r.price, cat: r.cat, typ: r.typ });
                        }
                    }
                }
            }
            outp.finish()
        })
        .collect();
    drop(parted);
    tlog!(t0, "events deduped");

    // ---- clicks: parse, validate, bucket by uid hash
    let cws = ws - WEEK;
    let cl_chunk_size = (cl_total / (nthreads * 4)).clamp(1 << 20, 64 << 20);
    let cl_chunks = chunks_of(&cl_bufs, cl_chunk_size);
    let ucls: Vec<Vec<Vec<UCl>>> = cl_chunks
        .par_iter()
        .map(|&c| {
            let est = c.len() / 48 / NP + 16;
            let mut parts = Scatter::<UCl>::new(est);
            let ctx = &*ctx;
            scan::<4>(c, |f| {
                let [cid, uid, camp_s, ts_s] = f;
                if cid.is_empty() || uid.is_empty() {
                    return;
                }
                let ts = match parse_ts(ts_s) {
                    Some(t) if t >= cws && t < ctx.we => t,
                    _ => return,
                };
                let ci = match ctx.camp.get(camp_s) {
                    Some(c) => c,
                    None => return,
                };
                let (uh, uh2) = hh(uid);
                parts.push((uh >> NP_SHIFT) as usize, UCl { uh, uh2, ts, cid, camp: ci });
            });
            parts.finish()
        })
        .collect();
    tlog!(t0, "clicks parsed: {} chunks", cl_chunks.len());

    // ---- resolve persons per uid bucket (small, cache-resident link tables)
    let resolved: Vec<(Vec<Vec<PEv>>, Vec<Vec<PCl>>)> = (0..NP)
        .into_par_iter()
        .map(|bi| {
            let t = &ltabs[bi];
            let ne: usize = uevs.iter().map(|p| p[bi].len()).sum();
            let nc: usize = ucls.iter().map(|p| p[bi].len()).sum();
            let ee = ne / NP + 16;
            let ce = nc / NP + 16;
            let mut eo = Scatter::<PEv>::new(ee);
            let mut co = Scatter::<PCl>::new(ce);
            for p in &uevs {
                for u in &p[bi] {
                    let (nh, name) = match t.find(u.uh.rotate_left(24), |x| x.h == u.uh && x.h2 == u.uh2) {
                        Some(x) => (x.nh, x.name),
                        None => (u.uh, u.uid),
                    };
                    eo.push((nh >> NP_SHIFT) as usize, PEv { nh, ts: u.ts, name, price: u.price, cat: u.cat, typ: u.typ });
                }
            }
            for p in &ucls {
                for u in &p[bi] {
                    let nh = match t.find(u.uh.rotate_left(24), |x| x.h == u.uh && x.h2 == u.uh2) {
                        Some(x) => x.nh,
                        None => u.uh,
                    };
                    co.push((nh >> NP_SHIFT) as usize, PCl { nh, ts: u.ts, cid: u.cid, camp: u.camp });
                }
            }
            (eo.finish(), co.finish())
        })
        .collect();
    drop(uevs);
    drop(ucls);
    let (deduped, clicks): (Vec<Vec<Vec<PEv>>>, Vec<Vec<Vec<PCl>>>) = resolved.into_iter().unzip();
    tlog!(t0, "persons resolved");

    // ---- per person-partition: sessionize, attribute, aggregate
    let ws_day = ws.div_euclid(DAY);
    let nd = ((we - 1).div_euclid(DAY) - ws_day + 1).max(1) as usize;
    let ncs = ncamp + 1; // + organic
    let folded: Vec<(Vec<Vec<URow>>, Vec<i64>)> = (0..NP)
        .into_par_iter()
        .fold(|| (Vec::new(), vec![0i64; ncs * nd * 4]), |(mut rv, mut agg), pi| {
            let ne: usize = deduped.iter().map(|p| p[pi].len()).sum();
            // group events by person hash: person-local ids
            let mut ptab: HashTable<(u64, u32)> = HashTable::with_capacity(ne / 8 + 16);
            let mut pcount: Vec<u32> = Vec::with_capacity(ne / 8 + 16);
            let mut pidx: Vec<u32> = Vec::with_capacity(ne);
            for p in &deduped {
                for e in &p[pi] {
                    let hh = e.nh.rotate_left(24);
                    let id = match ptab.find(hh, |x| x.0 == e.nh) {
                        Some(x) => x.1,
                        None => {
                            let id = pcount.len() as u32;
                            ptab.insert_unique(hh, (e.nh, id), |x| x.0.rotate_left(24));
                            pcount.push(0);
                            id
                        }
                    };
                    pcount[id as usize] += 1;
                    pidx.push(id);
                }
            }
            let np = pcount.len();
            let mut pstart: Vec<u32> = Vec::with_capacity(np + 1);
            let mut acc = 0u32;
            for &c in &pcount {
                pstart.push(acc);
                acc += c;
            }
            pstart.push(acc);
            let mut ev: Vec<PEv> = Vec::with_capacity(ne);
            unsafe { ev.set_len(ne) };
            {
                let mut fill = pstart.clone();
                let mut k = 0;
                for p in &deduped {
                    for e in &p[pi] {
                        let id = pidx[k] as usize;
                        k += 1;
                        ev[fill[id] as usize] = *e;
                        fill[id] += 1;
                    }
                }
            }
            drop(pidx);
            // clicks of persons with events only
            let mut ccount = vec![0u32; np + 1];
            let mut cidx: Vec<u32> = Vec::new();
            for p in &clicks {
                for c in &p[pi] {
                    let id = match ptab.find(c.nh.rotate_left(24), |x| x.0 == c.nh) {
                        Some(x) => x.1,
                        None => u32::MAX,
                    };
                    if id != u32::MAX {
                        ccount[id as usize] += 1;
                    }
                    cidx.push(id);
                }
            }
            let mut cstart: Vec<u32> = Vec::with_capacity(np + 1);
            let mut acc = 0u32;
            for &c in &ccount[..np] {
                cstart.push(acc);
                acc += c;
            }
            cstart.push(acc);
            let ncl = acc as usize;
            let mut cl: Vec<PCl> = Vec::with_capacity(ncl);
            unsafe { cl.set_len(ncl) };
            {
                let mut fill = cstart.clone();
                let mut k = 0;
                for p in &clicks {
                    for c in &p[pi] {
                        let id = cidx[k];
                        k += 1;
                        if id != u32::MAX {
                            cl[fill[id as usize] as usize] = *c;
                            fill[id as usize] += 1;
                        }
                    }
                }
            }
            drop(cidx);
            for q in 0..np {
                let es = &mut ev[pstart[q] as usize..pstart[q + 1] as usize];
                if es.len() > 1 {
                    es.sort_unstable_by_key(|e| e.ts);
                }
                let cs = &mut cl[cstart[q] as usize..cstart[q + 1] as usize];
                if cs.len() > 1 {
                    cs.sort_unstable_by(|a, b| (a.ts, a.cid).cmp(&(b.ts, b.cid)));
                }
            }

            // campaign aggregates: [ (camp*nd + d) * 4 + k ], k: sessions, purchasing, revenue, assisted
            let mut rows: Vec<URow> = Vec::with_capacity(ne / 8 + 16);
            let mut catsum = vec![0u64; ncat];
            let mut touched: Vec<u16> = Vec::with_capacity(ncat);
            let mut wcount = vec![0u32; ncamp];
            let mut active: Vec<u16> = Vec::new();
            let mut apos = vec![u32::MAX; ncamp];

            for q in 0..np {
                let i = pstart[q] as usize;
                let pe = pstart[q + 1] as usize;
                let name = ev[i].name;
                let cs = cstart[q] as usize;
                let ce = cstart[q + 1] as usize;
                let (mut lo, mut hi) = (cs, cs);

                let mut cur_day: i64 = i64::MIN;
                let mut r_sess = 0i64;
                let mut r_ev = 0i64;
                let mut r_rev = 0i64;

                let flush = |name: S, day: i64, s: i64, e: i64, r: i64, catsum: &mut Vec<u64>, touched: &mut Vec<u16>, rows: &mut Vec<URow>| {
                    let mut top: i32 = -1;
                    let mut best = 0u64;
                    for &c in touched.iter() {
                        let v = catsum[c as usize];
                        if top < 0 || v > best || (v == best && cat_rank[c as usize] < cat_rank[top as usize]) {
                            top = c as i32;
                            best = v;
                        }
                        catsum[c as usize] = 0;
                    }
                    touched.clear();
                    rows.push(URow { name, day: day as i32, sessions: s, events: e, revenue: r, top });
                };

                let mut k = i;
                while k < pe {
                    let sess_begin = k;
                    let start = ev[k].ts;
                    let mut last = start;
                    let mut cnt = 0i64;
                    let mut rev = 0i64;
                    let mut purch = false;
                    while k < pe && ev[k].ts - last <= GAP {
                        let e = &ev[k];
                        last = e.ts;
                        cnt += 1;
                        if e.typ == 2 {
                            purch = true;
                            rev += e.price as i64;
                        }
                        k += 1;
                    }
                    let sess_end = k;
                    let day = start.div_euclid(DAY);
                    if day != cur_day {
                        if cur_day != i64::MIN {
                            flush(name, cur_day, r_sess, r_ev, r_rev, &mut catsum, &mut touched, &mut rows);
                        }
                        cur_day = day;
                        r_sess = 0;
                        r_ev = 0;
                        r_rev = 0;
                    }
                    if purch {
                        for e in &ev[sess_begin..sess_end] {
                            if e.typ == 2 {
                                let c = e.cat as usize;
                                if catsum[c] == 0 {
                                    touched.push(e.cat);
                                }
                                catsum[c] += e.price as u64;
                            }
                        }
                    }
                    r_sess += 1;
                    r_ev += cnt;
                    r_rev += rev;
                    // NOTE: catsum accumulates per user-day; sessions on the same day share it.

                    // attribution window [start - WEEK, start]
                    while hi < ce && cl[hi].ts <= start {
                        let c = cl[hi].camp as usize;
                        if wcount[c] == 0 {
                            apos[c] = active.len() as u32;
                            active.push(c as u16);
                        }
                        wcount[c] += 1;
                        hi += 1;
                    }
                    let lim = start - WEEK;
                    while lo < hi && cl[lo].ts < lim {
                        let c = cl[lo].camp as usize;
                        wcount[c] -= 1;
                        if wcount[c] == 0 {
                            let p = apos[c] as usize;
                            let lastc = *active.last().unwrap();
                            active[p] = lastc;
                            apos[lastc as usize] = p as u32;
                            active.pop();
                            apos[c] = u32::MAX;
                        }
                        lo += 1;
                    }
                    let credited = if lo < hi {
                        let t = cl[hi - 1].ts;
                        let mut m = hi - 1;
                        while m > lo && cl[m - 1].ts == t {
                            m -= 1;
                        }
                        cl[m].camp as usize
                    } else {
                        ncamp
                    };
                    let d = (day - ws_day) as usize;
                    let base = (credited * nd + d) * 4;
                    agg[base] += 1;
                    if purch {
                        agg[base + 1] += 1;
                    }
                    agg[base + 2] += rev;
                    for &c in active.iter() {
                        agg[(c as usize * nd + d) * 4 + 3] += 1;
                    }
                }
                if cur_day != i64::MIN {
                    flush(name, cur_day, r_sess, r_ev, r_rev, &mut catsum, &mut touched, &mut rows);
                }
                // clear window
                for m in lo..hi {
                    let c = cl[m].camp as usize;
                    wcount[c] = 0;
                    apos[c] = u32::MAX;
                }
                active.clear();
            }
            rv.push(rows);
            (rv, agg)
        })
        .collect();
    let mut results: Vec<(Vec<URow>, Vec<i64>)> = Vec::new();
    let mut agg = vec![0i64; ncs * nd * 4];
    for (rv, a) in folded {
        for (x, y) in agg.iter_mut().zip(a.iter()) {
            *x += *y;
        }
        for r in rv {
            results.push((r, Vec::new()));
        }
    }
    tlog!(t0, "sessionized");

    // ---- merge campaign aggregates

    // ---- write outputs
    let ud = out.join("user_daily");
    let cd = out.join("campaign_daily");
    std::fs::create_dir_all(&ud).ok();
    std::fs::create_dir_all(&cd).ok();

    // user_daily: group results into ~nthreads files
    let total_rows: usize = results.iter().map(|r| r.0.len()).sum();
    let per_file = (total_rows / nthreads).max(1);
    let mut groups: Vec<Vec<&Vec<URow>>> = vec![Vec::new()];
    let mut acc = 0;
    for (r, _) in &results {
        if acc >= per_file {
            groups.push(Vec::new());
            acc = 0;
        }
        groups.last_mut().unwrap().push(r);
        acc += r.len();
    }
    let cat_names_ref = &cat_names;
    groups.par_iter().enumerate().for_each(|(gi, g)| {
        write_user_daily(&ud.join(format!("part-{:05}.parquet", gi)), g, cat_names_ref, ws_day);
    });

    // campaign_daily
    let mut crow: Vec<(S, S, i32, [i64; 4])> = Vec::new();
    for c in 0..ncs {
        for d in 0..nd {
            let b = (c * nd + d) * 4;
            let a = [agg[b], agg[b + 1], agg[b + 2], agg[b + 3]];
            if a[0] > 0 || a[3] > 0 {
                let (id, ch): (S, S) = if c == ncamp { (b"organic", b"organic") } else { (camp_names[c], camp_chan[c]) };
                crow.push((id, ch, (ws_day + d as i64) as i32, a));
            }
        }
    }
    write_campaign_daily(&cd.join("part-00000.parquet"), &crow);
    tlog!(t0, "written: {} user rows, {} campaign rows", total_rows, crow.len());
}

// ---------------------------------------------------------------- parquet output

use parquet::data_type::{ByteArray, ByteArrayType, Int32Type, Int64Type};
use parquet::file::properties::{EnabledStatistics, WriterProperties, WriterVersion};
use parquet::file::writer::SerializedFileWriter;
use parquet::schema::parser::parse_message_type;

fn props() -> Arc<WriterProperties> {
    Arc::new(
        WriterProperties::builder()
            .set_writer_version(WriterVersion::PARQUET_1_0)
            .set_dictionary_enabled(false)
            .set_statistics_enabled(EnabledStatistics::None)
            .set_max_row_group_size(1 << 30)
            .set_data_page_size_limit(8 << 20)
            .set_write_batch_size(1 << 20)
            .build(),
    )
}

fn write_user_daily(path: &Path, groups: &[&Vec<URow>], cats: &[S], _ws_day: i64) {
    let schema = Arc::new(
        parse_message_type(
            "message user_daily { required binary user_id (UTF8); required int32 day (DATE); required int64 sessions; required int64 events; required int64 revenue_cents; optional binary top_category (UTF8); }",
        )
        .unwrap(),
    );
    let f = File::create(path).unwrap();
    let mut w = SerializedFileWriter::new(f, schema, props()).unwrap();
    let rows: Vec<&URow> = groups.iter().flat_map(|g| g.iter()).collect();
    let mut rg = w.next_row_group().unwrap();
    let mut col = 0;
    while let Some(mut cw) = rg.next_column().unwrap() {
        match col {
            0 => {
                let v: Vec<ByteArray> = rows.iter().map(|r| ByteArray::from(r.name.to_vec())).collect();
                cw.typed::<ByteArrayType>().write_batch(&v, None, None).unwrap();
            }
            1 => {
                let v: Vec<i32> = rows.iter().map(|r| r.day).collect();
                cw.typed::<Int32Type>().write_batch(&v, None, None).unwrap();
            }
            2 | 3 | 4 => {
                let v: Vec<i64> = rows
                    .iter()
                    .map(|r| match col {
                        2 => r.sessions,
                        3 => r.events,
                        _ => r.revenue,
                    })
                    .collect();
                cw.typed::<Int64Type>().write_batch(&v, None, None).unwrap();
            }
            _ => {
                let defs: Vec<i16> = rows.iter().map(|r| if r.top >= 0 { 1 } else { 0 }).collect();
                let v: Vec<ByteArray> = rows.iter().filter(|r| r.top >= 0).map(|r| ByteArray::from(cats[r.top as usize].to_vec())).collect();
                cw.typed::<ByteArrayType>().write_batch(&v, Some(&defs), None).unwrap();
            }
        }
        cw.close().unwrap();
        col += 1;
    }
    rg.close().unwrap();
    w.close().unwrap();
}

fn write_campaign_daily(path: &Path, rows: &[(S, S, i32, [i64; 4])]) {
    let schema = Arc::new(
        parse_message_type(
            "message campaign_daily { required binary campaign_id (UTF8); required binary channel (UTF8); required int32 day (DATE); required int64 sessions; required int64 purchasing_sessions; required int64 revenue_cents; required int64 assisted_sessions; }",
        )
        .unwrap(),
    );
    let f = File::create(path).unwrap();
    let mut w = SerializedFileWriter::new(f, schema, props()).unwrap();
    let mut rg = w.next_row_group().unwrap();
    let mut col = 0;
    while let Some(mut cw) = rg.next_column().unwrap() {
        match col {
            0 | 1 => {
                let v: Vec<ByteArray> = rows.iter().map(|r| ByteArray::from(if col == 0 { r.0 } else { r.1 }.to_vec())).collect();
                cw.typed::<ByteArrayType>().write_batch(&v, None, None).unwrap();
            }
            2 => {
                let v: Vec<i32> = rows.iter().map(|r| r.2).collect();
                cw.typed::<Int32Type>().write_batch(&v, None, None).unwrap();
            }
            _ => {
                let k = col - 3;
                let v: Vec<i64> = rows.iter().map(|r| r.3[k]).collect();
                cw.typed::<Int64Type>().write_batch(&v, None, None).unwrap();
            }
        }
        cw.close().unwrap();
        col += 1;
    }
    rg.close().unwrap();
    w.close().unwrap();
}
