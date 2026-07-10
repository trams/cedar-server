//! Tracking-mode plate solving for `Tetra3RsSolver`: seeding a solve with a prior
//! attitude (`SolveConfig::attitude_hint`) so tetra3rs solves by direct
//! correspondence instead of the blind lost-in-space pattern hash.
//!
//! This is the highest-value follow-up from `docs/06-tetra3rs-real-images.md`: the
//! five dawn frames tetra3rs's blind search misses are exactly the sparse fields a
//! hint rescues, and production's Operate loop has the previous frame's attitude on
//! hand for every solve.
//!
//! Two things are proven here, both against **real** tetra3rs solves:
//!
//! 1. `hint_reconstructs_the_solver_quaternion` -- the pose the adapter reports
//!    (`ra`/`dec`/`roll`) round-trips through `attitude_to_quaternion` back to the
//!    solver's own `qicrs2cam`, to within a fraction of a degree. This pins the
//!    roll/handedness convention end-to-end; a sign error here is undetectable by
//!    the unit tests (which are self-consistent by construction).
//! 2. `tracking_rescues_frames_the_blind_search_misses` -- a frame tetra3 solved
//!    but tetra3rs's blind search cannot, seeded with a neighbour's attitude, now
//!    solves and agrees with its own ground truth. With `strict_hint` false the
//!    blind search is still the fallback, so a solve here is a solve tracking (not
//!    the pattern hash) produced -- these frames have no blind solution to fall
//!    back to.
//!
//! ```text
//! # defaults to <repo>/benchmarks/take1; generates the tetra3rs db on first run
//! cargo test --release --test e2e_tetra3rs_tracking -- --ignored --nocapture
//! ```

use std::sync::Arc;

use cedar_elements::cedar::ImageCoord;
use cedar_elements::imu_trait::EquatorialCoordinates;
use cedar_elements::solver_trait::{SolveExtension, SolveParams, SolverTrait};
use cedar_server::tetra3rs_solver::{attitude_to_quaternion, roll_deg, Tetra3RsSolver};
use tetra3::{Centroid, SolveConfig, SolverDatabase};
use tokio::sync::Mutex;

mod common;

use common::bench_corpus::{self as bench, BenchFrame};
use common::corpus;
use common::harness::Stack;

/// The Cedar box camera's horizontal FOV; see `e2e_tetra3rs_real.rs`.
const FOV_SEED_DEG: f64 = 12.8;

/// How closely the reconstructed hint quaternion must match the solver's own
/// attitude. The pose is read back through the WCS (`pixel_to_world`) and the
/// `roll_deg`/`theta_rad` path, so exact equality is not expected -- but a
/// convention error (flipped roll, wrong handedness) is degrees, not arcminutes.
const CONVENTION_TOL_DEG: f64 = 0.5;

/// A rescued frame's solved pose must land this close to its tetra3 ground truth.
/// Generous: the point is that it solves at all and is not a hallucination.
const RESCUE_AGREE_DEG: f64 = 0.5;

struct Env {
    bench_dir: std::path::PathBuf,
    db_path: std::path::PathBuf,
}

fn setup() -> Option<Env> {
    if std::env::var_os("CEDAR_BENCH_DIR").is_some() {
        panic!("CEDAR_BENCH_DIR is set; the solve worker would capture images. Unset it.");
    }
    let env = bench::bench_dir().and_then(|bench_dir| {
        Ok(Env {
            bench_dir,
            db_path: corpus::tetra3rs_database()?,
        })
    });
    match env {
        Ok(env) => Some(env),
        Err(why) => {
            eprintln!("\nSKIPPING e2e tetra3rs tracking test:\n{why}\n");
            None
        }
    }
}

fn solver(env: &Env) -> Arc<Mutex<dyn SolverTrait + Send + Sync>> {
    Arc::new(Mutex::new(
        Tetra3RsSolver::from_database_file(
            env.db_path.to_str().expect("utf-8 db path"),
            Some(FOV_SEED_DEG),
        )
        .expect("Tetra3RsSolver::from_database_file"),
    ))
}

fn frames(env: &Env) -> Vec<BenchFrame> {
    let f = bench::load_manifest(&env.bench_dir).expect("parse bench manifest.csv");
    assert!(!f.is_empty(), "empty bench manifest");
    f
}

/// Centroids as the solve engine hands them to the solver: full-resolution image
/// coordinates, top-left origin.
fn centroids_of(ps: &cedar_server::solve_engine::PlateSolution) -> Vec<ImageCoord> {
    ps.detect_result
        .star_candidates
        .iter()
        .map(|sc| ImageCoord {
            x: sc.centroid_x,
            y: sc.centroid_y,
        })
        .collect()
}

/// Angle between two rotations, via the quaternion inner product. `q` and `-q`
/// are the same rotation, hence `abs()`.
fn quat_angle_deg(a: &tetra3::Quaternion, b: &tetra3::Quaternion) -> f64 {
    let d = a.dot(b).abs().min(1.0) as f64;
    2.0 * d.acos().to_degrees()
}

/// End-to-end convention anchor: reconstruct the attitude from the pose the
/// adapter reports and compare it to the solver's own quaternion.
#[tokio::test(flavor = "multi_thread")]
#[ignore = "needs a captured corpus (CEDAR_E2E_BENCH_DIR) and a tetra3rs database"]
async fn hint_reconstructs_the_solver_quaternion() {
    let Some(env) = setup() else { return };
    let all = frames(&env);
    let db = SolverDatabase::load_from_file(env.db_path.to_str().unwrap())
        .expect("load tetra3rs db");

    // Detect centroids on a real, tetra3-solved frame.
    let seed = bench::load_image(&env.bench_dir, &all[0]).expect("load bmp");
    let mut stack = Stack::new(solver(&env), seed).await;

    let mut checked = 0;
    for frame in all.iter().filter(|f| f.truth.is_some()) {
        let image = bench::load_image(&env.bench_dir, frame).expect("load bmp");
        let (w, h) = image.dimensions();
        let ps = stack.solve_image(image).await;
        let centroids = centroids_of(&ps);

        // Solve the same centroids directly, to get the raw Solution (qicrs2cam).
        let t3: Vec<Centroid> = centroids
            .iter()
            .map(|c| Centroid {
                x: (c.x - w as f64 / 2.0) as f32,
                y: (c.y - h as f64 / 2.0) as f32,
                mass: None,
                cov: None,
            })
            .collect();
        let mut config = SolveConfig::new(FOV_SEED_DEG.to_radians() as f32, w, h);
        config.match_max_error = Some(0.005);
        let Ok(sol) = db.solve_from_centroids(&t3, &config) else {
            continue; // blind search declined this frame; try another.
        };

        // The pose the adapter reports for this Solution.
        let (ra, dec) = sol.pixel_to_world(0.0, 0.0);
        let roll = roll_deg(sol.theta_rad, sol.parity_flip);

        let q_recon = attitude_to_quaternion(ra, dec, roll);
        let err = quat_angle_deg(&q_recon, &sol.qicrs2cam);
        println!(
            "  {}: ra {:.4} dec {:.4} roll {:.4} -> hint {:.4} deg from qicrs2cam",
            frame.name(),
            ra,
            dec,
            roll,
            err
        );
        assert!(
            err < CONVENTION_TOL_DEG,
            "{}: reconstructed hint is {err:.3} deg off the solver's own attitude -- \
             suspect a roll-sign or handedness error in attitude_to_quaternion",
            frame.name()
        );
        checked += 1;
        if checked >= 5 {
            break;
        }
    }
    assert!(checked > 0, "no frame produced a direct tetra3rs solution to check");
}

/// The promised win: a frame the blind search misses, rescued by a hint.
#[tokio::test(flavor = "multi_thread")]
#[ignore = "needs a captured corpus (CEDAR_E2E_BENCH_DIR) and a tetra3rs database"]
async fn tracking_rescues_frames_the_blind_search_misses() {
    let Some(env) = setup() else { return };
    let all = frames(&env);
    let shared = solver(&env);

    let seed = bench::load_image(&env.bench_dir, &all[0]).expect("load bmp");
    let mut stack = Stack::new(shared.clone(), seed).await;

    // Frames tetra3 solved (so we have ground truth) but tetra3rs's blind search
    // misses -- the exact regression tracking is meant to close.
    let mut rescued = 0;
    let mut attempted = 0;
    for (i, frame) in all.iter().enumerate() {
        let Some(truth) = frame.truth.as_ref() else { continue };
        let image = bench::load_image(&env.bench_dir, frame).expect("load bmp");
        let (w, h) = image.dimensions();
        let ps = stack.solve_image(image).await;
        if ps.plate_solution.is_some() {
            continue; // blind search already solved it; not a rescue candidate.
        }

        // A neighbouring solved frame supplies the hint attitude. The mount tracks
        // ~0.01 deg/frame, so an adjacent solved frame is a faithful prior.
        let Some(neighbour) = nearest_solved_neighbour(&all, i) else { continue };
        attempted += 1;

        let hint = EquatorialCoordinates {
            north_roll_angle: neighbour.roll_deg,
            ra: neighbour.ra_deg,
            dec: neighbour.dec_deg,
        };
        let params = SolveParams {
            match_max_error: Some(0.005),
            attitude_hint: Some(hint),
            ..Default::default()
        };
        let ext = SolveExtension::default();
        let centroids = centroids_of(&ps);

        let solved = shared
            .lock()
            .await
            .solve_from_centroids(&centroids, w as usize, h as usize, &ext, &params, None)
            .await;

        match solved {
            Ok(p) => {
                let c = p.image_sky_coord.as_ref().unwrap();
                let drift = bench::separation_deg(truth.ra_deg, truth.dec_deg, c.ra, c.dec);
                println!(
                    "  {} ({} centroids): blind MISS -> tracking SOLVED ra {:.4} dec {:.4}, \
                     {:.4} deg from truth",
                    frame.name(),
                    centroids.len(),
                    c.ra,
                    c.dec,
                    drift
                );
                assert!(
                    drift < RESCUE_AGREE_DEG,
                    "{}: tracking solved {drift:.3} deg from ground truth -- a hallucination, \
                     not a rescue",
                    frame.name()
                );
                rescued += 1;
            }
            Err(e) => {
                println!("  {} ({} centroids): blind MISS -> tracking also failed: {e:?}",
                    frame.name(), centroids.len());
            }
        }
    }

    println!("\n  tracking rescued {rescued}/{attempted} frames the blind search missed");
    if attempted == 0 {
        eprintln!(
            "no frame in this corpus was solved by tetra3 yet missed by tetra3rs's blind \
             search; nothing to rescue (the convention anchor still ran)."
        );
        return;
    }
    assert!(
        rescued > 0,
        "tracking rescued none of {attempted} blind-search misses -- the hint is not \
         engaging (suspect the attitude convention) or the corpus changed"
    );
}

/// The tetra3-solved frame nearest `idx` in capture order. The mount tracks
/// steadily across the corpus, so the closest solved frame on either side is a
/// faithful attitude prior; prefer the one before.
fn nearest_solved_neighbour(all: &[BenchFrame], idx: usize) -> Option<bench::Truth> {
    all[..idx]
        .iter()
        .rev()
        .find_map(|f| f.truth.clone())
        .or_else(|| all[idx + 1..].iter().find_map(|f| f.truth.clone()))
}
