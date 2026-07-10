//! `Tetra3RsSolver` against **real camera frames** captured off a Cedar box,
//! rather than the synthetic corpus of `e2e_tetra3rs.rs`.
//!
//! This is the "real Pi captures" item left open by `docs/03-e2e-server-harness.md`
//! and `docs/05-tetra3rs-solver.md`. It exercises what synthetic images cannot:
//! sensor read noise, hot pixels, sky glow, star trailing, and a real optical PSF.
//!
//! **It is an agreement test, not a correctness test.** The frames' ground truth
//! is whatever the Pi's own Python tetra3 reported at capture time. Agreement
//! bounds how far the two solvers diverge; it cannot detect a bias they share.
//! The images are the full-resolution frames `DetectEngine` consumed, and the
//! recorded `roll`/`fov` come off the same `PlateSolution` proto this harness
//! reads, so no reprojection sits between the two answers.
//!
//! ```text
//! # defaults to <repo>/benchmarks/take1 and every 6th frame
//! cargo test --release --test e2e_tetra3rs_real -- --ignored --nocapture
//!
//! CEDAR_E2E_BENCH_DIR=/path/to/corpus CEDAR_E2E_BENCH_STRIDE=1 \
//!   cargo test --release --test e2e_tetra3rs_real -- --ignored --nocapture
//! ```
//!
//! No Python and no `/tmp/cedar.sock`: tetra3's answers are read from the
//! manifest, not recomputed. A separate test binary for the same reason
//! `e2e_tetra3rs.rs` is one -- engine worker threads outlive their `Stack`, so
//! sharing a process with the tetra3 suite would contaminate the latency numbers.

use std::path::PathBuf;
use std::sync::Arc;

use cedar_elements::solver_trait::SolverTrait;
use cedar_server::tetra3rs_solver::Tetra3RsSolver;
use tokio::sync::Mutex;

mod common;

use common::bench_corpus::{self as bench, BenchFrame};
use common::corpus;
use common::harness::Stack;
use common::report::{print_table, Report};

type SharedSolver = Arc<Mutex<dyn SolverTrait + Send + Sync>>;

/// tetra3rs's blind search does not solve every frame tetra3 solved.
///
/// Five frames in this corpus (the 999 ms dawn exposures at the end of the
/// session, 8-22 centroids each) fail, and the cause is characterized:
///
/// - It is not the catalog. Seeded with a neighbouring frame's attitude,
///   tetra3rs's *tracking* mode matches every centroid to a catalog star
///   (8/8, 19/21, 19/20, 22/22), at 35-50" rmse.
/// - It is not the database. Regenerating with tetra3's looser quantization
///   (`pattern_max_error` 0.005) or 4x the patterns (`patterns_per_lattice_field`
///   200, 66 MB) rescues none of them.
/// - It is the lost-in-space pattern hash. tetra3rs *samples*
///   `patterns_per_lattice_field` quadruples per sky field; a frame showing only
///   its 8 brightest stars offers just C(8,4)=70 quadruples, and none of them was
///   sampled. A night frame offers thousands and always hits one.
///
/// So the gate is the measured rate, not 1.0. It is paired with
/// `assert_solved_frames_agree`, which does not bend: declining to solve is a
/// capability limit; solving *wrongly* is a bug. Raise this if the blind search
/// improves; never lower it silently.
///
/// The fix is not a bigger database -- it is to pass the previous solution's
/// attitude as `SolveConfig::attitude_hint`, which is exactly what production's
/// frame-to-frame Operate loop has available. See `docs/06-tetra3rs-real-images.md`.
const MIN_SOLVE_RATE: f64 = 0.94;

/// The Cedar box camera's horizontal FOV, which the calibrator knows. The corpus
/// records 12.735-12.832 deg across the session; production would seed a single
/// nominal value, so that is what this does.
const FOV_SEED_DEG: f64 = 12.8;

/// Frames sampled from the corpus: every Nth row. 6 keeps a ~600-frame session
/// near 100 frames while spanning the whole run rather than one patch of sky.
const DEFAULT_STRIDE: usize = 6;

/// Two solved frames closer than this are the same pointing, so the mount was
/// tracking (not slewing) across the frame between them. Adjacent solved frames
/// sit 0.009 deg apart at the median; re-pointings jump by degrees.
const MOUNT_STEADY_DEG: f64 = 0.5;

/// How far a solve on an unsolvable frame may sit from its steady neighbours
/// before it is a hallucination rather than a bonus solve.
const NEIGHBOR_TOL_DEG: f64 = 1.0;

struct Env {
    bench_dir: PathBuf,
    db_path: PathBuf,
}

/// `SolveState.bench = BenchConfig::from_env()` (solve_engine.rs:544). Set, and
/// the solve worker starts writing a *new* corpus mid-test -- quite possibly on
/// top of the one being read.
fn guard_bench_capture() {
    if let Some(dir) = std::env::var_os("CEDAR_BENCH_DIR") {
        panic!(
            "CEDAR_BENCH_DIR is set ({dir:?}); the solve worker would capture \
             benchmark images during this test. Unset it and rerun. (The corpus \
             to read is named by CEDAR_E2E_BENCH_DIR.)"
        );
    }
}

fn setup() -> Option<Env> {
    guard_bench_capture();
    let env = bench::bench_dir().and_then(|bench_dir| {
        Ok(Env {
            bench_dir,
            db_path: corpus::tetra3rs_database()?,
        })
    });
    match env {
        Ok(env) => Some(env),
        Err(why) => {
            eprintln!("\nSKIPPING e2e tetra3rs real-image test:\n{why}\n");
            None
        }
    }
}

fn solver(env: &Env) -> SharedSolver {
    let solver = Tetra3RsSolver::from_database_file(
        env.db_path.to_str().expect("utf-8 database path"),
        Some(FOV_SEED_DEG),
    )
    .expect("Tetra3RsSolver::from_database_file");
    Arc::new(Mutex::new(solver))
}

fn frames(env: &Env) -> Vec<BenchFrame> {
    let frames = bench::load_manifest(&env.bench_dir).expect("parse bench manifest.csv");
    assert!(!frames.is_empty(), "empty bench manifest");
    frames
}

fn stride() -> usize {
    bench::env_usize("CEDAR_E2E_BENCH_STRIDE", DEFAULT_STRIDE)
}

/// One real frame, printed in full. Real sky differs from synthetic in centroid
/// count, PSF, and noise floor; see all of it before spending a run on a hundred
/// frames.
#[tokio::test(flavor = "multi_thread")]
#[ignore = "needs a captured corpus (CEDAR_E2E_BENCH_DIR) and a tetra3rs database"]
async fn gate1_real_frame() {
    let Some(env) = setup() else { return };
    let frames = frames(&env);
    let frame = frames
        .iter()
        .find(|f| f.truth.is_some())
        .expect("no tetra3-solved frame in the corpus");
    let truth = frame.truth.as_ref().unwrap();
    let image = bench::load_image(&env.bench_dir, frame).expect("load bmp");

    let mut stack = Stack::new(solver(&env), image.clone()).await;
    let ps = stack.solve_image(image).await;

    println!("\n=== tetra3rs on a real frame: {} ===", frame.name());
    println!(
        "  exposure        {} ms   ({}x{})",
        frame.exposure_ms, frame.width, frame.height
    );
    println!(
        "  centroids       {} (tetra3 matched {} on the Pi)",
        ps.detect_result.star_candidates.len(),
        truth.num_matches
    );

    let p = ps
        .plate_solution
        .as_ref()
        .unwrap_or_else(|| panic!("{} did not solve", frame.name()));
    let coord = p.image_sky_coord.as_ref().unwrap();

    println!(
        "  tetra3 (Pi)     ra {:.5}  dec {:.5}  roll {:.4}  fov {:.4}  rmse {:.2}\"  {:.1} ms",
        truth.ra_deg, truth.dec_deg, truth.roll_deg, truth.fov_deg, truth.rmse_arcsec, truth.solve_ms
    );
    println!(
        "  tetra3rs        ra {:.5}  dec {:.5}  roll {:.4}  fov {:.4}  rmse {:.2}\"  matches {}",
        coord.ra, coord.dec, p.roll, p.fov, p.rmse, p.num_matches
    );

    let o = bench::evaluate_agreement(frame, &ps);
    println!(
        "  disagreement    center {:.4}'   roll {:.4} deg   fov {:.4}%   solve {:.2} ms",
        o.center_arcmin,
        o.roll_err_deg,
        o.fov_err_frac * 100.0,
        o.solve_time_ms
    );

    assert!(o.solved, "{} did not solve", frame.name());
    assert!(
        o.center_arcmin < 30.0,
        "center disagrees by {:.2}' -- suspect a WCS convention mismatch, not noise",
        o.center_arcmin
    );
    assert!(
        o.roll_err_deg.abs() < 1.0,
        "roll {:.4} disagrees with tetra3's {:.4}",
        p.roll,
        truth.roll_deg
    );
    assert!(
        !p.catalog_stars.is_empty(),
        "no catalog stars returned; the Setup-mode overlay would be empty"
    );
}

/// A strided subset of the frames tetra3 solved. Grades tetra3rs's answer against
/// tetra3's, on the same pose gates the synthetic suite uses.
#[tokio::test(flavor = "multi_thread")]
#[ignore = "needs a captured corpus (CEDAR_E2E_BENCH_DIR) and a tetra3rs database"]
async fn e2e_real_corpus_tetra3rs() {
    let Some(env) = setup() else { return };
    let all = frames(&env);
    let solved: Vec<BenchFrame> = all.iter().filter(|f| f.truth.is_some()).cloned().collect();
    let sample = bench::subsample(&solved, stride());
    assert!(!sample.is_empty(), "subset is empty");

    println!(
        "\n{} of {} tetra3-solved frames (every {}th) from {}, FOV seed {FOV_SEED_DEG} deg",
        sample.len(),
        solved.len(),
        stride(),
        env.bench_dir.display()
    );

    let seed = bench::load_image(&env.bench_dir, &sample[0]).expect("load bmp");
    let mut stack = Stack::new(solver(&env), seed).await;

    let mut outcomes = Vec::with_capacity(sample.len());
    for frame in &sample {
        let image = bench::load_image(&env.bench_dir, frame).expect("load bmp");
        outcomes.push(bench::evaluate_agreement(
            frame,
            &stack.solve_image(image).await,
        ));
    }
    let report = Report::new("tetra3rs-real", outcomes);

    // What the Pi's tetra3 spent on these same frames, for the latency column.
    let pi_times: Vec<f64> = sample
        .iter()
        .map(|f| f.truth.as_ref().unwrap().solve_ms)
        .collect();
    let pi_mean = pi_times.iter().sum::<f64>() / pi_times.len() as f64;
    let pi_max = pi_times.iter().copied().fold(f64::MIN, f64::max);

    print_table(&[&report]);
    println!(
        "  tetra3 on the Pi solved these same frames in {:.1} ms mean, {:.1} ms max \
         (recorded at capture, on Pi hardware -- not comparable to this box's timings)",
        pi_mean, pi_max
    );
    match report.write_csv() {
        Ok(p) => println!("per-frame CSV: {}", p.display()),
        Err(e) => eprintln!("could not write CSV: {e}"),
    }

    let s = report.summary();
    println!(
        "\n  {} solved, {} not; median disagreement with tetra3: {:.3}' of arc, \
         {:.3}% of FOV",
        s.solved,
        s.total - s.solved,
        s.center_med_arcmin,
        s.fov_med_frac * 100.0
    );

    report.print_failures();
    report.assert_solved_frames_agree();
    report.assert_solve_rate(MIN_SOLVE_RATE);
    report.assert_latency();
}

/// The frames tetra3 could **not** solve -- twilight, cloud, a slewing mount.
///
/// A faster solver that also invents poses is worse than a slow one. These frames
/// have no ground truth, so correctness is checked against time: the mount moves
/// 0.009 deg between adjacent frames while tracking. When the solved frames either
/// side of a gap agree with each other, the mount was steady across it, and any
/// pose reported for the frame between them must agree too.
///
/// Solving zero of these is a perfectly good result. Solving one *wrongly* is not.
#[tokio::test(flavor = "multi_thread")]
#[ignore = "needs a captured corpus (CEDAR_E2E_BENCH_DIR) and a tetra3rs database"]
async fn does_not_hallucinate_on_frames_tetra3_could_not_solve() {
    let Some(env) = setup() else { return };
    let all = frames(&env);

    let unsolved: Vec<usize> = (0..all.len()).filter(|i| all[*i].truth.is_none()).collect();
    if unsolved.is_empty() {
        eprintln!("corpus has no tetra3-unsolved frames; nothing to check");
        return;
    }

    let seed_idx = all
        .iter()
        .position(|f| f.truth.is_some())
        .expect("no solved frame to seed the stack");
    let seed = bench::load_image(&env.bench_dir, &all[seed_idx]).expect("load bmp");
    let mut stack = Stack::new(solver(&env), seed).await;

    println!(
        "\n{} frames tetra3 could not solve; checking tetra3rs against its neighbours\n",
        unsolved.len()
    );

    let (mut solved_count, mut verified, mut unverifiable) = (0usize, 0usize, 0usize);
    for &i in &unsolved {
        let frame = &all[i];
        let image = bench::load_image(&env.bench_dir, frame).expect("load bmp");
        let ps = stack.solve_image(image).await;
        let centroids = ps.detect_result.star_candidates.len();

        let Some(p) = ps.plate_solution.as_ref() else {
            println!(
                "  {}  {:>4} ms  {:>3} centroids  no solve (as tetra3)",
                frame.name(),
                frame.exposure_ms,
                centroids
            );
            continue;
        };
        solved_count += 1;
        let coord = p.image_sky_coord.as_ref().unwrap();

        // Nearest tetra3-solved frame on each side.
        let before = all[..i].iter().rev().find_map(|f| f.truth.as_ref());
        let after = all[i + 1..].iter().find_map(|f| f.truth.as_ref());

        let steady = match (before, after) {
            (Some(b), Some(a)) => {
                bench::separation_deg(b.ra_deg, b.dec_deg, a.ra_deg, a.dec_deg) < MOUNT_STEADY_DEG
            }
            _ => false,
        };

        if !steady {
            unverifiable += 1;
            println!(
                "  {}  {:>4} ms  {:>3} centroids  SOLVED ra {:.4} dec {:.4} \
                 -- neighbours disagree (mount slewing), unverifiable",
                frame.name(),
                frame.exposure_ms,
                centroids,
                coord.ra,
                coord.dec
            );
            continue;
        }

        let b = before.unwrap();
        let drift = bench::separation_deg(b.ra_deg, b.dec_deg, coord.ra, coord.dec);
        verified += 1;
        println!(
            "  {}  {:>4} ms  {:>3} centroids  SOLVED ra {:.4} dec {:.4} \
             -- {:.4} deg from a steady neighbour",
            frame.name(),
            frame.exposure_ms,
            centroids,
            coord.ra,
            coord.dec,
            drift
        );
        assert!(
            drift < NEIGHBOR_TOL_DEG,
            "{} solved to ra {:.4} dec {:.4}, {:.3} deg from the pose its steady \
             neighbours bracket -- that is a hallucinated solution, not a bonus one",
            frame.name(),
            coord.ra,
            coord.dec,
            drift
        );
    }

    println!(
        "\n  tetra3rs solved {}/{} frames tetra3 could not: {} verified against a \
         steady neighbour, {} unverifiable (mount slewing)",
        solved_count,
        unsolved.len(),
        verified,
        unverifiable
    );
}
