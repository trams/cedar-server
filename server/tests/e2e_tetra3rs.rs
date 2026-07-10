//! The same engine-level corpus as `e2e_plate_solve.rs`, driven by the pure-Rust
//! `Tetra3RsSolver` instead of the Python tetra3 subprocess.
//!
//! Deliberately a separate test *binary*, for two reasons:
//!
//! - It needs no Python, no venv, and no `/tmp/cedar.sock`, so it must not be
//!   gated behind `e2e_plate_solve`'s preconditions.
//! - `DetectEngine`/`SolveEngine` worker threads outlive their `Stack` (neither
//!   engine implements `Drop`). Sharing a process with the tetra3 run would leave
//!   those workers spinning against the Python solver, contaminating the latency
//!   numbers this suite exists to measure. Cargo runs each `tests/*.rs` in its
//!   own process.
//!
//! ```text
//! export CEDAR_E2E_DATA_DIR=.../cedar-solve/tests/data/synthetic_large
//! cargo test --release --test e2e_tetra3rs -- --ignored --nocapture
//! ```
//!
//! A pattern database is generated from `tetra3rs/data/gaia_merged.bin` and
//! cached under `target/e2e-cache/` on first run; `CEDAR_E2E_TETRA3RS_DB`
//! overrides it.

use std::path::PathBuf;
use std::sync::Arc;

use cedar_elements::astro_util::{transform_to_celestial_coords, transform_to_image_coord};
use cedar_elements::solver_trait::SolverTrait;
use cedar_server::tetra3rs_solver::Tetra3RsSolver;
use image::GrayImage;
use tokio::sync::Mutex;

mod common;

use common::corpus::{self, Field};
use common::harness::{evaluate, expected_roll_deg, run_corpus, Stack};
use common::report::print_table;

type SharedSolver = Arc<Mutex<dyn SolverTrait + Send + Sync>>;

/// Every field is expected to solve.
const MIN_SOLVE_RATE: f64 = 1.0;

/// The FOV seed handed to tetra3rs, standing in for a Cedar box camera's known
/// horizontal FOV.
///
/// Deliberately wrong by +18.5%: the corpus's true FOV is 12.658 deg. Seeding
/// with ground truth would prove nothing, since the seed is only a hint --
/// `docs/04-fov-seed-sensitivity.md` measured 99/99 solves for any seed within
/// roughly +/-50% of truth. A rough but plausible number is the honest test.
const FOV_SEED_DEG: f64 = 15.0;

struct Env {
    data_dir: PathBuf,
    db_path: PathBuf,
}

/// `SolveState.bench = BenchConfig::from_env()` (solve_engine.rs:544) -- if this
/// is set, the solve worker starts writing BMPs to disk mid-test.
fn guard_bench_capture() {
    if let Some(dir) = std::env::var_os("CEDAR_BENCH_DIR") {
        panic!(
            "CEDAR_BENCH_DIR is set ({dir:?}); the solve worker would capture \
             benchmark images during this test. Unset it and rerun."
        );
    }
}

fn setup() -> Option<Env> {
    guard_bench_capture();
    let env = corpus::data_dir().and_then(|data_dir| {
        Ok(Env {
            data_dir,
            db_path: corpus::tetra3rs_database()?,
        })
    });
    match env {
        Ok(env) => Some(env),
        Err(why) => {
            eprintln!("\nSKIPPING e2e tetra3rs test:\n{why}\n");
            None
        }
    }
}

/// `fov_seed` of `None` exercises the blind FOV ladder -- what the calibrator's
/// first solve does in production, before any FOV is known.
fn solver(env: &Env, fov_seed: Option<f64>) -> SharedSolver {
    let solver = Tetra3RsSolver::from_database_file(
        env.db_path.to_str().expect("utf-8 database path"),
        fov_seed,
    )
    .expect("Tetra3RsSolver::from_database_file");
    Arc::new(Mutex::new(solver))
}

fn fields(env: &Env) -> Vec<Field> {
    corpus::load_manifest(&env.data_dir).expect("parse manifest.csv")
}

/// Gate 1: one field, checking the conventions that a quaternion-valued solver
/// could plausibly get wrong -- roll sense, FOV definition, and the sky<->pixel
/// transforms -- before spending a run on 99 fields.
#[tokio::test(flavor = "multi_thread")]
#[ignore = "needs CEDAR_E2E_DATA_DIR and a tetra3rs database (or the Gaia catalog)"]
async fn gate1_conventions() {
    let Some(env) = setup() else { return };
    let fields = fields(&env);
    let field = &fields[0];
    let image = corpus::load_image(&env.data_dir, field).expect("load png");

    let mut stack = Stack::new(solver(&env, Some(FOV_SEED_DEG)), image.clone()).await;
    let ps = stack.solve_image(image).await;

    println!("\n=== tetra3rs gate 1: {} ===", field.name);
    println!("  centroids       {}", ps.detect_result.star_candidates.len());

    let p = ps
        .plate_solution
        .as_ref()
        .unwrap_or_else(|| panic!("{} did not solve", field.name));
    let coord = p.image_sky_coord.as_ref().unwrap();

    println!(
        "  ground truth    ra {:.4}  dec {:.4}  rotation {:.1}",
        field.ra_deg, field.dec_deg, field.rotation_deg
    );
    println!("  solved          ra {:.4}  dec {:.4}", coord.ra, coord.dec);
    println!(
        "  roll            raw {:.4}   expected (180+rot)%360 = {:.4}",
        p.roll,
        expected_roll_deg(field.rotation_deg)
    );
    println!(
        "  fov             {:.4} (gt {:.4} gnomonic; manifest fov_x_deg {:.4})",
        p.fov,
        field.true_fov_x_deg(),
        field.fov_x_deg
    );
    println!(
        "  matches {}  rmse {:.3}\"  p90 {:.3}\"  max {:.3}\"  prob {:.2e}",
        p.num_matches, p.rmse, p.p90_error, p.max_error, p.prob
    );
    println!("  catalog_stars   {}", p.catalog_stars.len());

    let o = evaluate(field, &ps);
    println!(
        "  center {:.4}'   roll_err {:.4} deg   fov_err {:.4}%   {:.1} ms",
        o.center_arcmin,
        o.roll_err_deg,
        o.fov_err_frac * 100.0,
        o.solve_time_ms
    );

    assert!(o.solved, "{} did not solve", field.name);
    assert!(
        o.center_arcmin < 30.0,
        "center off by {:.2}' -- suspect a WCS convention mismatch",
        o.center_arcmin
    );
    assert!(
        o.roll_err_deg.abs() < 1.0,
        "roll {:.3} does not match (180 + {:.1}) % 360 = {:.3}",
        p.roll,
        field.rotation_deg,
        expected_roll_deg(field.rotation_deg)
    );

    // The proto must carry the fields solve_engine unconditionally unwraps.
    assert!(
        p.distortion.is_some(),
        "distortion is None; solve_engine.rs:1285 unwraps it"
    );
    assert_eq!(p.epoch_equinox, 2000, "catalog is not J2000/ICRS");
    assert_eq!(
        p.rotation_matrix.len(),
        9,
        "rotation_matrix must be a 3x3 row-major matrix"
    );
    assert!(
        !p.catalog_stars.is_empty(),
        "return_catalog was requested but no catalog stars came back; the \
         Setup-mode star overlay would be empty"
    );
}

/// The rotation matrix, FOV, and per-star pixels must be mutually consistent
/// *in cedar's own convention*, not just in tetra3rs's.
///
/// `astro_util::transform_to_image_coord` is cedar's port of tetra3's projection
/// and is what draws the Setup-mode catalog overlay. Feeding it our
/// `rotation_matrix` and `fov` must reproduce the pixel positions we computed
/// with tetra3rs's own `world_to_pixel`. A transposed, mirrored, or
/// axis-permuted matrix passes every pose gate above and fails only here.
#[tokio::test(flavor = "multi_thread")]
#[ignore = "needs CEDAR_E2E_DATA_DIR and a tetra3rs database (or the Gaia catalog)"]
async fn rotation_matrix_agrees_with_cedars_projection() {
    let Some(env) = setup() else { return };
    let fields = fields(&env);
    let field = &fields[0];
    let image = corpus::load_image(&env.data_dir, field).expect("load png");

    let mut stack = Stack::new(solver(&env, Some(FOV_SEED_DEG)), image.clone()).await;
    let ps = stack.solve_image(image).await;
    let p = ps.plate_solution.as_ref().expect("solved");

    let (w, h) = (field.nx as usize, field.ny as usize);
    let rm: [f64; 9] = p.rotation_matrix.clone().try_into().expect("3x3 matrix");
    let distortion = p.distortion.unwrap();

    // 1. The boresight must land on the image center.
    let center = p.image_sky_coord.as_ref().unwrap();
    let projected = transform_to_image_coord(&[center.ra, center.dec], w, h, p.fov, &rm, distortion);
    let center_err = ((projected[0] - w as f64 / 2.0).powi(2)
        + (projected[1] - h as f64 / 2.0).powi(2))
    .sqrt();
    println!(
        "\nboresight reprojects to ({:.3}, {:.3}); image center is ({}, {}) -- {:.4} px",
        projected[0],
        projected[1],
        w / 2,
        h / 2,
        center_err
    );
    assert!(center_err < 1.0, "boresight reprojection off by {center_err:.3} px");

    // 2. Every catalog star we reported must reproject onto the pixel we gave.
    let mut worst = 0.0f64;
    for star in &p.catalog_stars {
        let sky = star.sky_coord.as_ref().unwrap();
        let pixel = star.pixel.as_ref().unwrap();
        let got = transform_to_image_coord(&[sky.ra, sky.dec], w, h, p.fov, &rm, distortion);
        let err = ((got[0] - pixel.x).powi(2) + (got[1] - pixel.y).powi(2)).sqrt();
        worst = worst.max(err);
    }
    println!(
        "{} catalog stars reproject through astro_util within {worst:.4} px",
        p.catalog_stars.len()
    );
    assert!(
        worst < 2.0,
        "catalog star reprojection off by up to {worst:.3} px -- rotation_matrix \
         and fov disagree with the pixels we reported"
    );

    // 3. And the inverse: cedar's pixel -> sky must return the boresight.
    let back = transform_to_celestial_coords(
        &[w as f64 / 2.0, h as f64 / 2.0],
        w,
        h,
        p.fov,
        &rm,
        distortion,
    );
    let sep_arcmin = cedar_elements::astro_util::angular_separation(
        back[0].to_radians(),
        back[1].to_radians(),
        center.ra.to_radians(),
        center.dec.to_radians(),
    )
    .to_degrees()
        * 60.0;
    println!("image center back-projects to within {sep_arcmin:.5}' of image_sky_coord");
    assert!(sep_arcmin < 0.1, "inverse transform off by {sep_arcmin:.4}'");
}

/// The blind path: no FOV seed at all, as the calibrator's first solve runs.
/// Proves the FOV ladder covers the database's 10-30 deg range.
#[tokio::test(flavor = "multi_thread")]
#[ignore = "needs CEDAR_E2E_DATA_DIR and a tetra3rs database (or the Gaia catalog)"]
async fn solves_blind_without_a_fov_seed() {
    let Some(env) = setup() else { return };
    let fields = fields(&env);
    let field = &fields[0];
    let image = corpus::load_image(&env.data_dir, field).expect("load png");

    let mut stack = Stack::new(solver(&env, /*fov_seed=*/ None), image.clone()).await;
    let ps = stack.solve_image(image).await;
    let o = evaluate(field, &ps);

    println!(
        "\nblind solve: center {:.4}'  fov_err {:.4}%  {:.1} ms",
        o.center_arcmin,
        o.fov_err_frac * 100.0,
        o.solve_time_ms
    );
    assert!(o.passed(), "blind solve failed the pose gates: {o:?}");
}

/// The full corpus, plus the negative control.
#[tokio::test(flavor = "multi_thread")]
#[ignore = "needs CEDAR_E2E_DATA_DIR and a tetra3rs database (or the Gaia catalog)"]
async fn e2e_corpus_tetra3rs() {
    let Some(env) = setup() else { return };
    let fields = fields(&env);
    assert!(!fields.is_empty(), "empty manifest");
    println!(
        "\nRunning {} fields from {} against tetra3rs (FOV seed {FOV_SEED_DEG} deg)",
        fields.len(),
        env.data_dir.display()
    );

    let seed = corpus::load_image(&env.data_dir, &fields[0]).expect("load png");
    let mut stack = Stack::new(solver(&env, Some(FOV_SEED_DEG)), seed).await;
    let report = run_corpus(&mut stack, &env.data_dir, &fields, "tetra3rs").await;

    // A blank frame yields too few centroids, so solve_engine never reaches the
    // solver. If this ever produces a pose, the gates below mean nothing.
    let blank = GrayImage::new(fields[0].nx, fields[0].ny);
    let blank_ps = stack.solve_image(blank).await;
    assert!(
        blank_ps.plate_solution.is_none(),
        "a blank frame produced a plate solution ({} centroids)",
        blank_ps.detect_result.star_candidates.len()
    );

    print_table(&[&report]);
    match report.write_csv() {
        Ok(p) => println!("per-field CSV: {}", p.display()),
        Err(e) => eprintln!("could not write CSV: {e}"),
    }
    println!(
        "\nCompare against tetra3: target/e2e-report/tetra3.csv \
         (cargo test --release --test e2e_plate_solve -- --ignored --test-threads=1)"
    );

    report.print_failures();
    report.assert_gates(MIN_SOLVE_RATE);
}
