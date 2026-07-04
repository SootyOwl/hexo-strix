//! `prove` — the HeXO deep-prover research CLI.
//!
//! Runs one of several proof-search drivers (or a portfolio race) over the
//! validated VCF kernel and writes a JSON report. Pure file I/O; a thin Python
//! wrapper handles HDS network fetch and analysis-URL rendering.
//!
//! ```text
//! prove --position pos.json
//!       [--lines a.json b.json ...]
//!       [--driver idtt|dfpn|pdspn|hybrid|race]   (default: hybrid if --lines, else dfpn)
//!       [--width tight|wide]                     (default tight)
//!       [--depth-cap 40] [--node-budget 20000000]
//!       [--tt-mb 512] [--pn2-nodes 50000]
//!       [--leaf-budget 1000000] [--leaf-budget-max 10000000]
//!       [--time-limit 1800]                      (seconds, wall clock; 0 = none)
//!       [--race-set idtt,dfpn,pdspn]
//!       [--out report.json]
//! ```

use hexo_rs::prover::io::{Line, Position, Report};
use hexo_rs::prover::{DriverKind, ProverConfig, run};
use std::path::PathBuf;
use std::process::ExitCode;

struct Args {
    position: Option<PathBuf>,
    lines: Vec<PathBuf>,
    driver: Option<DriverKind>,
    cfg: ProverConfig,
    out: Option<PathBuf>,
}

fn usage() -> &'static str {
    "usage: prove --position pos.json [--lines a.json ...] [--driver idtt|dfpn|pdspn|hybrid|race] \
     [--width tight|wide] [--depth-cap N] [--node-budget N] [--tt-mb N] [--pn2-nodes N] \
     [--leaf-budget N] [--leaf-budget-max N] [--time-limit SECS] [--race-set d,d,...] [--out report.json]"
}

fn parse_args() -> Result<Args, String> {
    let mut a = Args {
        position: None,
        lines: Vec::new(),
        driver: None,
        cfg: ProverConfig::default(),
        out: None,
    };
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let mut i = 0;
    let next = |i: &mut usize, argv: &[String], flag: &str| -> Result<String, String> {
        *i += 1;
        argv.get(*i).cloned().ok_or_else(|| format!("{flag} needs a value"))
    };
    while i < argv.len() {
        let arg = argv[i].clone();
        match arg.as_str() {
            "--position" => a.position = Some(PathBuf::from(next(&mut i, &argv, "--position")?)),
            "--out" => a.out = Some(PathBuf::from(next(&mut i, &argv, "--out")?)),
            "--lines" => {
                // Consume following args until the next flag.
                while i + 1 < argv.len() && !argv[i + 1].starts_with("--") {
                    i += 1;
                    a.lines.push(PathBuf::from(&argv[i]));
                }
            }
            "--driver" => {
                let v = next(&mut i, &argv, "--driver")?;
                a.driver = Some(DriverKind::parse(&v).ok_or_else(|| format!("bad driver {v:?}"))?);
            }
            "--width" => {
                let v = next(&mut i, &argv, "--width")?;
                a.cfg.wide = match v.as_str() {
                    "tight" => false,
                    "wide" => true,
                    _ => return Err(format!("bad width {v:?} (want tight|wide)")),
                };
            }
            "--depth-cap" => a.cfg.depth_cap = parse_num(&next(&mut i, &argv, "--depth-cap")?)? as u8,
            "--node-budget" => a.cfg.node_budget = parse_num(&next(&mut i, &argv, "--node-budget")?)?,
            "--tt-mb" => a.cfg.tt_mb = parse_num(&next(&mut i, &argv, "--tt-mb")?)? as usize,
            "--pn2-nodes" => a.cfg.pn2_nodes = parse_num(&next(&mut i, &argv, "--pn2-nodes")?)?,
            "--leaf-budget" => a.cfg.leaf_budget = parse_num(&next(&mut i, &argv, "--leaf-budget")?)?,
            "--leaf-budget-max" => {
                a.cfg.leaf_budget_max = parse_num(&next(&mut i, &argv, "--leaf-budget-max")?)?
            }
            "--time-limit" => {
                a.cfg.time_limit_s = next(&mut i, &argv, "--time-limit")?
                    .parse::<f64>()
                    .map_err(|e| format!("bad --time-limit: {e}"))?
            }
            "--race-set" => {
                let v = next(&mut i, &argv, "--race-set")?;
                let mut set = Vec::new();
                for name in v.split(',') {
                    set.push(DriverKind::parse(name).ok_or_else(|| format!("bad race driver {name:?}"))?);
                }
                a.cfg.race_set = set;
            }
            "-h" | "--help" => return Err(usage().to_string()),
            other => return Err(format!("unknown argument {other:?}\n{}", usage())),
        }
        i += 1;
    }
    Ok(a)
}

fn parse_num(s: &str) -> Result<u64, String> {
    // Accept plain integers and underscores/`k`/`m` suffixes for convenience.
    let s = s.replace('_', "");
    if let Some(stripped) = s.strip_suffix('m').or_else(|| s.strip_suffix('M')) {
        return Ok(stripped.parse::<u64>().map_err(|e| e.to_string())? * 1_000_000);
    }
    if let Some(stripped) = s.strip_suffix('k').or_else(|| s.strip_suffix('K')) {
        return Ok(stripped.parse::<u64>().map_err(|e| e.to_string())? * 1_000);
    }
    s.parse::<u64>().map_err(|e| e.to_string())
}

fn print_summary(report: &Report) {
    eprintln!("verdict: {:?}", report.verdict);
    eprintln!("driver:  {}   width: {}", report.driver, report.width);
    if let Some(d) = report.depth {
        eprintln!("depth:   {d}");
    }
    let s = &report.stats;
    eprintln!(
        "stats:   nodes={} tt_hits={} tt_bytes={} leaf_solves={} line_steps={} elapsed={:.2}s",
        s.nodes, s.tt_hits, s.tt_bytes, s.leaf_solves, s.line_steps, s.elapsed_s
    );
    if !report.pv.is_empty() {
        eprintln!("pv:      {} cells", report.pv.len());
    }
    for r in &report.race {
        eprintln!("  race[{}]: {:?} ({:.2}s, {} nodes)", r.driver, r.verdict, r.elapsed_s, r.nodes);
    }
    for u in &report.unverified {
        eprintln!("  unverified ({}): {} cells", u.note, u.path.len());
    }
}

fn run_cli() -> Result<(), String> {
    let args = parse_args()?;
    let pos_path = args.position.ok_or_else(|| format!("--position is required\n{}", usage()))?;
    let pos = Position::load(&pos_path)?;
    let lines: Vec<Line> =
        args.lines.iter().map(|p| Line::load(p)).collect::<Result<Vec<_>, _>>()?;

    let mut cfg = args.cfg;
    cfg.driver = args.driver.unwrap_or({
        if lines.is_empty() { DriverKind::Dfpn } else { DriverKind::Hybrid }
    });

    let report = run(&pos, &lines, &cfg);
    let json = report.to_json();
    if let Some(out) = &args.out {
        std::fs::write(out, &json).map_err(|e| format!("writing {}: {e}", out.display()))?;
        eprintln!("wrote {}", out.display());
    } else {
        println!("{json}");
    }
    print_summary(&report);
    Ok(())
}

fn main() -> ExitCode {
    match run_cli() {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::FAILURE
        }
    }
}
