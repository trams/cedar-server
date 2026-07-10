// Copyright (c) 2026 Steven Rosenthal smr@dt3.org
// See LICENSE file in root directory for license terms.

//! Builds the pattern database that `--solver tetra3rs` loads at startup.
//!
//! Generation takes seconds and holds the whole Gaia catalog in memory, so it is
//! a build step rather than something cedar-box-server does on boot.
//!
//! ```text
//! cargo run --release --bin make-tetra3rs-db -- \
//!     --gaia_catalog ../../tetra3rs/data/gaia_merged.bin \
//!     --out ./tetra3rs_db.bin --min_fov 10 --max_fov 30
//! ```
//!
//! The FOV range is the range of *fields the database can solve*, not a
//! precision requirement — the solver's FOV seed is a loose hint (see
//! `docs/04-fov-seed-sensitivity.md`). Widening the range costs database size
//! and pattern count; narrowing it to the box camera's exact FOV is a mistake,
//! since the 10-30 deg multiscale database out-solves a single-scale one at
//! *every* seed, including a perfect one.

use std::time::Instant;

use cedar_server::tetra3rs_solver::generate_database;
use pico_args::Arguments;

const HELP: &str = "\
Build a tetra3rs solver database from a merged Gaia DR3 + Hipparcos catalog.

OPTIONS:
  --gaia_catalog <path>   Path to gaia_merged.bin (required)
  --out <path>            Where to write the database (required)
  --min_fov <deg>         Narrowest field the database can solve  [10]
  --max_fov <deg>         Widest field the database can solve     [30]
";

fn main() {
    let mut pargs = Arguments::from_env();
    if pargs.contains(["-h", "--help"]) {
        println!("{HELP}");
        return;
    }

    let gaia_catalog: String = match pargs.value_from_str("--gaia_catalog") {
        Ok(p) => p,
        Err(e) => {
            eprintln!("--gaia_catalog is required: {e}\n\n{HELP}");
            std::process::exit(1);
        }
    };
    let out: String = match pargs.value_from_str("--out") {
        Ok(p) => p,
        Err(e) => {
            eprintln!("--out is required: {e}\n\n{HELP}");
            std::process::exit(1);
        }
    };
    let min_fov: f32 = pargs.value_from_str("--min_fov").unwrap_or(10.0);
    let max_fov: f32 = pargs.value_from_str("--max_fov").unwrap_or(30.0);

    if !(0.0 < min_fov && min_fov <= max_fov && max_fov < 180.0) {
        eprintln!("need 0 < min_fov <= max_fov < 180; got {min_fov} and {max_fov}");
        std::process::exit(1);
    }

    println!("Generating {min_fov}-{max_fov} deg database from {gaia_catalog} ...");
    let started = Instant::now();
    let db = match generate_database(&gaia_catalog, min_fov, max_fov) {
        Ok(db) => db,
        Err(e) => {
            eprintln!("generation failed: {e:?}");
            std::process::exit(1);
        }
    };

    if let Err(e) = db.save_to_file(&out) {
        eprintln!("writing {out}: {e:?}");
        std::process::exit(1);
    }

    let size_mb = std::fs::metadata(&out).map(|m| m.len()).unwrap_or(0) as f64 / 1e6;
    println!(
        "Wrote {out} ({size_mb:.1} MB) in {:.1}s: {} patterns, {} stars, mag limit {:.2}",
        started.elapsed().as_secs_f64(),
        db.props.num_patterns,
        db.star_catalog.len(),
        db.props.star_max_magnitude,
    );
}
