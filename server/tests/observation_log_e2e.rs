//! End-to-end validation of the observation-log hook: drive a real star image
//! through cedar-server's real DetectEngine + SolveEngine and confirm the
//! observation log accumulates `"type":"solve"` records on disk.
//!
//! The solver here is a trivial stub returning a canned solved solution, so
//! this test needs neither the Python tetra3 subprocess nor a tetra3rs pattern
//! database. It exercises the wiring that matters: capture -> detect -> solve ->
//! `finalize_and_post_result` -> `maybe_log_observation` -> JSON Lines file.
//!
//! Not gated behind the `e2e` opt-in feature: it is fast and self-contained.

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use canonical_error::{invalid_argument_error, CanonicalError};
use cedar_camera::abstract_camera::AbstractCamera;
use cedar_camera::image_camera::ImageCamera;
use cedar_elements::cedar::{ImageCoord, LatLong, PlateSolution as PlateSolutionProto};
use cedar_elements::cedar_common::CelestialCoord;
use cedar_elements::imu_trait::EquatorialCoordinates;
use cedar_elements::solver_trait::{SolveExtension, SolveParams, SolverTrait};
use cedar_server::detect_engine::{DetectEngine, DetectResult};
use cedar_server::observation_log::ObservationLog;
use cedar_server::position_reporter::TelescopePosition;
use cedar_server::solve_engine::SolveEngine;
use tokio::sync::Mutex;

const DEMO_IMAGE: &str = "run/demo_images/bright_star_align.jpg";

// A solver that ignores its inputs and returns a fixed solved solution, as long
// as detection produced enough centroids. Lets us validate the log's solved
// path without a real solver backend.
const STUB_RA: f64 = 123.456;
const STUB_DEC: f64 = -12.34;

struct StubSolver;

#[async_trait]
impl SolverTrait for StubSolver {
    async fn solve_from_centroids(
        &self,
        star_centroids: &[ImageCoord],
        _width: usize,
        _height: usize,
        _extension: &SolveExtension,
        _params: &SolveParams,
        _imu_estimate: Option<EquatorialCoordinates>,
    ) -> Result<PlateSolutionProto, CanonicalError> {
        if star_centroids.len() < 4 {
            return Err(invalid_argument_error("stub: too few centroids"));
        }
        Ok(PlateSolutionProto {
            image_sky_coord: Some(CelestialCoord {
                ra: STUB_RA,
                dec: STUB_DEC,
                epoch: None,
            }),
            roll: 45.0,
            fov: 10.0,
            rmse: 1.5,
            num_matches: star_centroids.len() as i32,
            solve_time: Some(prost_types::Duration {
                seconds: 0,
                nanos: 25_000_000,
            }),
            ..Default::default()
        })
    }
    fn cancel(&self) {}
    fn default_timeout(&self) -> Duration {
        Duration::from_secs(1)
    }
}

// Mirror of MyCedar::compute_binning for the corpus resolutions (see harness.rs).
fn detect_binning(width: u32, height: u32) -> u32 {
    let mpix = (width * height) as f64 / 1_000_000.0;
    if mpix <= 0.75 {
        1
    } else if mpix <= 3.0 {
        2
    } else if mpix <= 12.0 {
        4
    } else {
        4
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn observation_log_records_solves() {
    // Point the observation log at a unique temp file, unthrottled so every
    // solve is recorded. ObservationLog reads these at construction time.
    let dir = std::env::temp_dir().join(format!("cedar_obslog_test_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let log_path = dir.join("obs.jsonl");
    let _ = std::fs::remove_file(&log_path);
    std::env::set_var("CEDAR_OBSLOG", "1");
    std::env::set_var("CEDAR_OBSLOG_PATH", &log_path);
    std::env::set_var("CEDAR_OBSLOG_INTERVAL_SECS", "0");
    let observation_log = Arc::new(ObservationLog::from_env());
    // Keep a handle for the GoTo-path check below; the solve engine gets a clone.
    let goto_log = observation_log.clone();

    // Load the bundled star-alignment image (real sky, plenty of stars).
    let img_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .join(DEMO_IMAGE);
    let image = image::open(&img_path)
        .unwrap_or_else(|e| panic!("open {:?}: {:?}", img_path, e))
        .to_luma8();
    let (width, height) = image.dimensions();

    let camera: Arc<Mutex<Box<dyn AbstractCamera + Send>>> = Arc::new(Mutex::new(Box::new(
        ImageCamera::new(image).await.expect("ImageCamera::new"),
    )));

    let detect = Arc::new(Mutex::new(DetectEngine::new(
        Duration::from_millis(100),
        Duration::from_micros(10),
        Duration::from_secs(1),
        /*sigma=*/ 8.0,
        /*star_count_goal=*/ 20,
        camera,
        /*stats_capacity=*/ 100,
        /*hot_pixel_map=*/ None,
    )));
    detect.lock().await.set_autoexposure_enabled(false).await;
    detect
        .lock()
        .await
        .set_detect_binning(detect_binning(width, height), false)
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

    let solver: Arc<Mutex<dyn SolverTrait + Send + Sync>> = Arc::new(Mutex::new(StubSolver));
    let mut solve = SolveEngine::new(
        solver,
        /*cedar_sky=*/ None,
        /*hot_pixel_map=*/ None,
        /*imu_tracker=*/ None,
        detect.clone(),
        /*stats_capacity=*/ 100,
        pre_solve,
        post_solve,
        /*observer_location=*/ None,
        observation_log,
    )
    .expect("SolveEngine::new");

    // Drain a handful of solutions; each posts a PlateSolution and fires the
    // observation-log hook. get_next_result lazily starts the worker.
    let mut last_id: Option<i32> = None;
    for _ in 0..8 {
        let ps = solve
            .get_next_result(last_id, /*non_blocking=*/ false)
            .await
            .expect("blocking get_next_result returns Some");
        last_id = Some(ps.solution_id);
    }

    // Give the last async write a beat, then read the log back.
    let contents = std::fs::read_to_string(&log_path)
        .unwrap_or_else(|e| panic!("read {:?}: {:?}", log_path, e));
    let lines: Vec<&str> = contents.lines().filter(|l| !l.is_empty()).collect();
    eprintln!("observation log ({} lines):", lines.len());
    for l in &lines {
        eprintln!("  {}", l);
    }
    assert!(!lines.is_empty(), "observation log is empty");

    let mut saw_solved = false;
    for line in &lines {
        let v: serde_json::Value =
            serde_json::from_str(line).unwrap_or_else(|e| panic!("bad JSON {:?}: {:?}", line, e));
        assert_eq!(v["type"], "solve", "unexpected record type in {}", line);
        assert!(v["time"].is_string(), "missing time in {}", line);
        assert!(v["solution_id"].is_number(), "missing solution_id in {}", line);
        assert!(v.get("from_imu").is_none(), "IMU record leaked into log: {}", line);
        if v["solved"] == true {
            saw_solved = true;
            // The stub always returns our canned pointing.
            let ra = v["ra_deg"].as_f64().expect("solved record has ra_deg");
            let dec = v["dec_deg"].as_f64().expect("solved record has dec_deg");
            assert!((ra - STUB_RA).abs() < 1e-6, "ra {}", ra);
            assert!((dec - STUB_DEC).abs() < 1e-6, "dec {}", dec);
            assert!(v["num_matches"].as_i64().unwrap() >= 4);
        }
    }
    assert!(
        saw_solved,
        "no solved record was logged; detection may have found <4 stars in the demo image"
    );

    // GoTo path: the strong-intent events land in the same file, unthrottled.
    // Drive the TelescopePosition seam rather than log_goto() directly — that
    // seam is what every protocol (gRPC, LX200, Alpaca) funnels through, so
    // this exercises the same code MyCedar::initiate_action does. The LX200
    // command level has its own test in lx200_server.rs.
    let mut telescope_position = TelescopePosition::new(goto_log.clone());
    telescope_position.set_slew_target(101.287, -16.716, "grpc");
    telescope_position.clear_slew("grpc");

    let goto_lines: Vec<serde_json::Value> = std::fs::read_to_string(&log_path)
        .unwrap()
        .lines()
        .filter(|l| !l.is_empty())
        .map(|l| serde_json::from_str::<serde_json::Value>(l).unwrap())
        .filter(|v| v["type"] == "goto")
        .collect();
    assert_eq!(goto_lines.len(), 2, "expected two goto records");

    let initiate = &goto_lines[0];
    assert_eq!(initiate["action"], "initiate_slew");
    assert_eq!(initiate["source"], "grpc");
    assert!((initiate["ra_deg"].as_f64().unwrap() - 101.287).abs() < 1e-6);
    assert!((initiate["dec_deg"].as_f64().unwrap() - (-16.716)).abs() < 1e-6);
    assert!(initiate["time"].is_string());

    let stop = &goto_lines[1];
    assert_eq!(stop["action"], "stop_slew");
    // stop_slew carries no coordinates.
    assert!(stop.get("ra_deg").is_none(), "stop_slew should omit ra_deg");
    assert!(stop.get("dec_deg").is_none(), "stop_slew should omit dec_deg");

    let _ = std::fs::remove_dir_all(&dir);
}
