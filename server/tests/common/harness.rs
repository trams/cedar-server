// Engine-level harness: drives cedar-server's real DetectEngine + SolveEngine
// over a static ImageCamera, with the solver as a swappable component.
//
// This deliberately stops short of the gRPC server: no tonic, no port 80, no
// Bluetooth, no Operate-mode state machine. What it does exercise is the same
// camera -> detect -> solve path the box runs in flight.

use std::future::Future;
use std::path::Path;
use std::pin::Pin;
use std::sync::Arc;
use std::time::Duration;

use cedar_camera::abstract_camera::AbstractCamera;
use cedar_camera::image_camera::ImageCamera;
use cedar_elements::astro_util::angular_separation;
use cedar_elements::cedar::{ImageCoord, LatLong, PlateSolution as PlateSolutionProto};
use cedar_elements::cedar_common::CelestialCoord;
use cedar_elements::solver_trait::SolverTrait;
use cedar_server::detect_engine::{DetectEngine, DetectResult};
use cedar_server::observation_log::ObservationLog;
use cedar_server::solve_engine::{PlateSolution, SolveEngine};
use image::GrayImage;
use tokio::sync::Mutex;

use super::corpus::{self, Field};
use super::report::Report;

// Production values, from server_main's MyCedar::new call (cedar_server.rs:4526)
// and the pico-args defaults (cedar_server.rs:4029).
const INITIAL_EXPOSURE: Duration = Duration::from_millis(100);
const MIN_EXPOSURE: Duration = Duration::from_micros(10); // --min_exposure 0.00001 s
const MAX_EXPOSURE: Duration = Duration::from_secs(1); // --max_exposure 1.0 s
const DETECTION_SIGMA: f64 = 8.0; // --sigma
const STAR_COUNT_GOAL: i32 = 20; // --star_count_goal
const STATS_CAPACITY: usize = 100;

/// Upper bound on solutions consumed while waiting for a swapped-in image to
/// reach the solver. Generous: in practice it settles within a few.
const MAX_SETTLE_RESULTS: usize = 32;

/// Production's detect binning for a sensor of `width` x `height`, mirroring
/// `MyCedar::compute_binning` (cedar_server.rs:2050), which the server calls on
/// every camera change (cedar_server.rs:969, :3617).
///
/// Binning is derived from sensor megapixels, NOT from `--binning`, which only
/// overrides it. A 1920x1080 frame is 2.07 mpix, so **production detects at
/// binning 2**, and the harness must too: at binning 1 a faint real star's peak
/// stays under the noise floor, while summing 2x2 lifts it over the sigma
/// threshold. On synthetic images this hardly matters -- the stars are clean and
/// bright either way -- which is how the harness ran at binning 1 for so long
/// without anyone noticing.
///
/// Returns `(detect_binning, display_sampling)`.
pub fn production_binning(width: u32, height: u32) -> (u32, bool) {
    let mpix = (width * height) as f64 / 1_000_000.0;
    if mpix <= 0.75 {
        (1, false)
    } else if mpix <= 3.0 {
        (2, false)
    } else if mpix <= 12.0 {
        (4, false)
    } else {
        (4, true)
    }
}

// Gates. Center/roll/FOV mirror cedar-solve/tests/test_solve_e2e.py:30-33.
pub const CENTER_TOL_ARCMIN: f64 = 5.0;
pub const ROLL_TOL_DEG: f64 = 1.0;
pub const FOV_TOL_FRAC: f64 = 0.02;

type SharedSolver = Arc<Mutex<dyn SolverTrait + Send + Sync>>;
type SharedCamera = Arc<Mutex<Box<dyn AbstractCamera + Send>>>;

fn wrap_camera(camera: ImageCamera) -> SharedCamera {
    Arc::new(Mutex::new(Box::new(camera) as Box<dyn AbstractCamera + Send>))
}

/// The engine stack, built once and reused across every field. Only the camera
/// is swapped per field, which is what `cedar_server.rs`'s demo-image path does.
pub struct Stack {
    detect: Arc<Mutex<DetectEngine>>,
    solve: SolveEngine,
    /// Cursor into SolveEngine's monotonically increasing `solution_id`.
    /// Deliberately not `frame_id`: that restarts at zero for each new
    /// ImageCamera and would alias across fields.
    last_id: Option<i32>,
}

impl Stack {
    pub async fn new(solver: SharedSolver, seed_image: GrayImage) -> Stack {
        let (width, height) = seed_image.dimensions();
        let camera = wrap_camera(
            ImageCamera::new(seed_image)
                .await
                .expect("ImageCamera::new"),
        );

        let detect = Arc::new(Mutex::new(DetectEngine::new(
            INITIAL_EXPOSURE,
            MIN_EXPOSURE,
            MAX_EXPOSURE,
            DETECTION_SIGMA,
            STAR_COUNT_GOAL,
            camera,
            STATS_CAPACITY,
            /*hot_pixel_map=*/ None,
        )));

        // ImageCamera ignores exposure changes -- the pixels are baked into the
        // PNG -- so autoexposure cannot converge on anything. Disabling it keeps
        // the run deterministic and stops the worker from hunting.
        detect.lock().await.set_autoexposure_enabled(false).await;

        // DetectEngine defaults detect_binning to 1 (detect_engine.rs:135); the
        // server overrides it from sensor size before the first frame. Do the
        // same, or CedarDetect runs on the full-resolution frame and finds a
        // different set of stars than the box does. `detect_binning` lives in
        // DetectEngine state, not the camera, so this survives replace_camera.
        // Centroids come back in full-res coordinates regardless of binning
        // (cedar-detect/src/algorithm.rs:766), so nothing downstream changes.
        let (detect_binning, display_sampling) = production_binning(width, height);
        detect
            .lock()
            .await
            .set_detect_binning(detect_binning, display_sampling)
            .await;

        let pre_solve: Arc<
            dyn Fn() -> Pin<
                    Box<dyn Future<Output = (Option<CelestialCoord>, Option<CelestialCoord>)> + Send>,
                > + Send
                + Sync,
        > = Arc::new(|| Box::pin(async { (None, None) }));

        let post_solve: Arc<
            dyn Fn(
                    Option<ImageCoord>,
                    Option<DetectResult>,
                    Option<PlateSolutionProto>,
                ) -> Pin<Box<dyn Future<Output = Option<LatLong>> + Send>>
                + Send
                + Sync,
        > = Arc::new(|_, _, _| Box::pin(async { None }));

        let solve = SolveEngine::new(
            solver,
            /*cedar_sky=*/ None,
            /*hot_pixel_map=*/ None,
            /*imu_tracker=*/ None,
            detect.clone(),
            STATS_CAPACITY,
            pre_solve,
            post_solve,
            /*observer_location=*/ None,
            /*observation_log=*/ Arc::new(ObservationLog::disabled()),
        )
        .expect("SolveEngine::new");

        Stack {
            detect,
            solve,
            last_id: None,
        }
    }

    /// Swaps in `image` and returns the first solution that provably belongs to
    /// it.
    ///
    /// Identity is established by comparing pixels, not by counting results.
    /// Draining a fixed number of solutions does NOT work: while the solve
    /// worker is busy (a solve costs tens of ms), the detect worker keeps
    /// capturing from the camera it still holds, so several post-swap solutions
    /// can carry a DetectResult sourced from the *previous* image. Those solve
    /// successfully and look plausible -- they just describe the wrong field.
    /// `DetectResult` carries the frame it was computed from, so we match on it.
    pub async fn solve_image(&mut self, image: GrayImage) -> PlateSolution {
        let camera = wrap_camera(
            ImageCamera::new(image.clone())
                .await
                .expect("ImageCamera::new"),
        );
        self.detect.lock().await.replace_camera(camera).await;
        self.solve.clear_plate_solution().await;

        for _ in 0..MAX_SETTLE_RESULTS {
            let ps = self.next_result().await;
            if ps.detect_result.captured_image.image.as_raw() == image.as_raw() {
                return ps;
            }
        }
        panic!(
            "after {MAX_SETTLE_RESULTS} solutions the engine was still reporting \
             on a stale frame; the camera swap never took effect"
        );
    }

    async fn next_result(&mut self) -> PlateSolution {
        let ps = self
            .solve
            .get_next_result(self.last_id, /*non_blocking=*/ false)
            .await
            .expect("blocking get_next_result returns Some");
        self.last_id = Some(ps.solution_id);
        ps
    }
}

/// Drives every field of the corpus through one already-built stack.
///
/// Takes `&mut Stack` rather than a solver because the stack must be reusable
/// afterwards (for the blank-frame control) and because `Tetra3Solver` may not
/// be constructed twice -- it binds the hardcoded `/tmp/cedar.sock`.
pub async fn run_corpus(
    stack: &mut Stack,
    data_dir: &Path,
    fields: &[Field],
    solver_name: &str,
) -> Report {
    let mut outcomes = Vec::with_capacity(fields.len());
    for field in fields {
        let image = corpus::load_image(data_dir, field).expect("load png");
        outcomes.push(evaluate(field, &stack.solve_image(image).await));
    }
    Report::new(solver_name, outcomes)
}

/// Per-field measurement. Errors are recorded whether or not they pass, so the
/// report can show near-misses.
#[derive(Debug, Clone)]
pub struct Outcome {
    pub name: String,
    pub solved: bool,
    pub center_arcmin: f64,
    pub roll_err_deg: f64,
    pub fov_err_frac: f64,
    pub solve_time_ms: f64,
    pub num_matches: i32,
    pub num_centroids: usize,
}

impl Outcome {
    pub fn passed(&self) -> bool {
        self.solved
            && self.center_arcmin < CENTER_TOL_ARCMIN
            && self.roll_err_deg.abs() < ROLL_TOL_DEG
            && self.fov_err_frac < FOV_TOL_FRAC
    }
}

/// Smallest signed difference between two angles, in [-180, 180].
pub fn circular_diff_deg(a: f64, b: f64) -> f64 {
    (a - b + 180.0).rem_euclid(360.0) - 180.0
}

/// tetra3 reports Roll as celestial north relative to image up. Pinned
/// empirically by the earlier Python suite: Roll == (180 + rotation) mod 360.
/// Raw at this boundary -- serve_engine.rs:548 rotates roll for display, but
/// that is downstream of get_next_result.
pub fn expected_roll_deg(rotation_deg: f64) -> f64 {
    (180.0 + rotation_deg).rem_euclid(360.0)
}

pub fn evaluate(field: &Field, ps: &PlateSolution) -> Outcome {
    let num_centroids = ps.detect_result.star_candidates.len();
    let Some(p) = ps.plate_solution.as_ref() else {
        return Outcome {
            name: field.name.clone(),
            solved: false,
            center_arcmin: f64::NAN,
            roll_err_deg: f64::NAN,
            fov_err_frac: f64::NAN,
            solve_time_ms: f64::NAN,
            num_matches: 0,
            num_centroids,
        };
    };

    let coord = p
        .image_sky_coord
        .as_ref()
        .expect("a solved PlateSolution always carries image_sky_coord");

    // angular_separation is radians in, radians out; proto coords are degrees.
    let center_arcmin = angular_separation(
        field.ra_deg.to_radians(),
        field.dec_deg.to_radians(),
        coord.ra.to_radians(),
        coord.dec.to_radians(),
    )
    .to_degrees()
        * 60.0;

    let roll_err_deg = circular_diff_deg(p.roll, expected_roll_deg(field.rotation_deg));
    // Against the gnomonic FOV, not the manifest's small-angle fov_x_deg -- see
    // Field::true_fov_x_deg.
    let true_fov = field.true_fov_x_deg();
    let fov_err_frac = (p.fov - true_fov).abs() / true_fov;

    let solve_time_ms = p
        .solve_time
        .as_ref()
        .map(|d| d.seconds as f64 * 1000.0 + d.nanos as f64 / 1.0e6)
        .unwrap_or(f64::NAN);

    Outcome {
        name: field.name.clone(),
        solved: true,
        center_arcmin,
        roll_err_deg,
        fov_err_frac,
        solve_time_ms,
        num_matches: p.num_matches,
        num_centroids,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn field(rotation_deg: f64) -> Field {
        Field {
            name: "t".into(),
            ra_deg: 10.0,
            dec_deg: 20.0,
            rotation_deg,
            fov_x_deg: 12.71,
            fov_y_deg: 7.149_375,
            pixscale_arcsec: 23.831_25,
            nx: 1920,
            ny: 1080,
            n_rendered: 50,
        }
    }

    fn outcome(center_arcmin: f64, roll_err_deg: f64, fov_err_frac: f64) -> Outcome {
        Outcome {
            name: "t".into(),
            solved: true,
            center_arcmin,
            roll_err_deg,
            fov_err_frac,
            solve_time_ms: 4.0,
            num_matches: 30,
            num_centroids: 50,
        }
    }

    #[test]
    fn on_truth_passes() {
        assert!(outcome(0.04, 0.0, 0.004).passed());
    }

    #[test]
    fn center_just_outside_tolerance_fails() {
        assert!(!outcome(6.0, 0.0, 0.004).passed());
    }

    #[test]
    fn fov_outside_tolerance_fails() {
        assert!(!outcome(0.04, 0.0, 0.03).passed());
    }

    #[test]
    fn unsolved_never_passes() {
        let mut o = outcome(0.0, 0.0, 0.0);
        o.solved = false;
        assert!(!o.passed());
    }

    #[test]
    fn roll_wraps_across_zero() {
        // rotation 179.5 -> expected roll 359.5; solver says 0.2. The true error
        // is 0.7 deg, not 359.3.
        let f = field(179.5);
        assert!((expected_roll_deg(f.rotation_deg) - 359.5).abs() < 1e-9);
        let err = circular_diff_deg(0.2, expected_roll_deg(f.rotation_deg));
        assert!((err - 0.7).abs() < 1e-9, "err = {err}");
        assert!(outcome(0.04, err, 0.004).passed());
    }

    #[test]
    fn roll_convention_matches_pinned_formula() {
        // Spot values from docs/02-plan-e2e-solve-tests.md.
        assert_eq!(expected_roll_deg(0.0), 180.0);
        assert_eq!(expected_roll_deg(-180.0), 0.0);
        assert_eq!(expected_roll_deg(30.0), 210.0);
        assert_eq!(expected_roll_deg(-45.0), 135.0);
    }

    #[test]
    fn roll_error_beyond_tolerance_fails() {
        assert!(!outcome(0.04, 1.5, 0.004).passed());
    }

    /// The corpus resolution both suites run at. 2.07 mpix lands in the 2x bucket,
    /// so the harness must detect at binning 2 -- the earlier claim that
    /// production runs full-res (docs/03 finding #2) skipped compute_binning.
    #[test]
    fn corpus_resolution_bins_by_two() {
        assert_eq!(production_binning(1920, 1080), (2, false));
    }

    #[test]
    fn binning_buckets_follow_compute_binning() {
        assert_eq!(production_binning(640, 480), (1, false)); // 0.31 mpix
        assert_eq!(production_binning(1024, 768), (2, false)); // 0.79 mpix
        assert_eq!(production_binning(2028, 1520), (4, false)); // 3.08 mpix
        assert_eq!(production_binning(4056, 3040), (4, true)); // 12.3 mpix
    }
}
