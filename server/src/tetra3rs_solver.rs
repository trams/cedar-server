// Copyright (c) 2026 Steven Rosenthal smr@dt3.org
// See LICENSE file in root directory for license terms.

//! `SolverTrait` over [`tetra3rs`](https://tetra3rs.dev/), a pure-Rust
//! lost-in-space plate solver.
//!
//! This is the in-process alternative to [`tetra3_server::tetra3_solver`],
//! which spawns a Python subprocess and talks gRPC over `/tmp/cedar.sock`.
//! Selected at startup with `--solver tetra3rs` (see `cedar_server.rs`).
//!
//! Three impedance mismatches, all resolved here:
//!
//! 1. **Sync vs async.** `SolverDatabase::solve_from_centroids` is synchronous.
//!    We run it on `spawn_blocking` so a slow blind solve cannot stall a tokio
//!    worker.
//! 2. **Quaternion vs proto.** tetra3rs returns a quaternion, a CD matrix and
//!    `theta_rad`; cedar's proto wants `(ra, dec, roll, fov)` in tetra3's
//!    conventions plus a rotation matrix. See [`roll_deg`] and
//!    [`tetra3_rotation_matrix`].
//! 3. **FOV seed.** tetra3rs needs a `CameraModel` with a focal length, whereas
//!    cedar solves blind (`SolveParams::fov_estimate == None`) until the
//!    calibrator has run. The seed is a *loose hint* — see [`fov_seeds`] and
//!    `docs/04-fov-seed-sensitivity.md`.
//!
//! `cancel()` has no true analogue: tetra3rs offers no interrupt hook, only a
//! timeout. It is honored between FOV seeds and before the solve starts.

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use async_trait::async_trait;
use canonical_error::{
    deadline_exceeded_error, failed_precondition_error, internal_error,
    invalid_argument_error, not_found_error, CanonicalError,
};
use log::{info, warn};

use cedar_elements::cedar::{ImageCoord, PlateSolution, StarInfo};
use cedar_elements::cedar_common::CelestialCoord;
use cedar_elements::imu_trait::EquatorialCoordinates;
use cedar_elements::solver_trait::{SolveExtension, SolveParams, SolverTrait};

use numeris::Matrix3;
use tetra3::{
    Centroid, GenerateDatabaseConfig, Quaternion, Solution, SolveConfig, SolveStatus,
    SolverDatabase,
};

/// Matches `Tetra3Solver::default_timeout()`, so swapping solvers does not
/// silently change the solve deadline.
const DEFAULT_TIMEOUT: Duration = Duration::from_secs(5);

/// Cap on `catalog_stars` returned for the Setup-mode overlay. The overlay
/// (serve_engine.rs:912) discards anything beyond half the image height from
/// center anyway, so an unbounded list would just inflate the proto.
const MAX_CATALOG_STARS: usize = 200;

/// A FOV seed `s` solves fields whose true FOV lies in
/// `[SEED_COVERS_BELOW * s, SEED_COVERS_ABOVE * s]`.
///
/// tetra3rs's pattern keys are edge *ratios*, which are scale-invariant to
/// first order, so the seed is a hint rather than a measurement. The full-corpus
/// sweep in `docs/04-fov-seed-sensitivity.md` solved 99/99 for every seed
/// between 0.5x and 1.5x the true FOV -- i.e. a true FOV anywhere in
/// `[0.67s, 2.0s]`. These constants keep a margin inside that measured plateau.
const SEED_COVERS_BELOW: f64 = 0.75;
const SEED_COVERS_ABOVE: f64 = 1.75;

/// Guards against a pathological database range producing an unbounded ladder.
const MAX_BLIND_SEEDS: usize = 8;

/// Angular uncertainty attached to a tracking-mode attitude hint. tetra3rs uses
/// it to size the catalog cone and the pixel-space nearest-neighbor match radius.
///
/// The hints cedar supplies are the previous frame's pointing (frame-to-frame
/// drift ~0.01 deg while tracking, arcminutes of prior-solve pointing error), so
/// 1 deg is generous while keeping the match radius tight enough to avoid spurious
/// correspondences. A hint wrong by more than this fails tracking and falls back
/// to the blind lost-in-space search, so it is a performance knob, not a
/// correctness one. Matches tetra3rs's own default.
const HINT_UNCERTAINTY_DEG: f32 = 1.0;

/// Generates a solver database from a merged Gaia DR3 + Hipparcos catalog
/// (`gaia_merged.bin`), spanning `[min_fov_deg, max_fov_deg]`.
///
/// Slow (seconds) and memory-hungry; call it from a build tool, not from server
/// startup. See `src/bin/make_tetra3rs_db.rs`.
pub fn generate_database(
    gaia_catalog_path: &str,
    min_fov_deg: f32,
    max_fov_deg: f32,
) -> Result<SolverDatabase, CanonicalError> {
    let config = GenerateDatabaseConfig {
        max_fov_deg,
        min_fov_deg: Some(min_fov_deg),
        ..Default::default()
    };
    SolverDatabase::generate_from_gaia(gaia_catalog_path, &config).map_err(|e| {
        failed_precondition_error(
            format!("generating solver database from {gaia_catalog_path:?}: {e:?}")
                .as_str(),
        )
    })
}

/// Normalize an angle to `[0, 360)`.
///
/// `rem_euclid(360.0)` alone is not enough: for a tiny negative input it
/// returns exactly `360.0` (Python's `% 360` has the same edge).
fn normalize_deg(deg: f64) -> f64 {
    let r = deg.rem_euclid(360.0);
    if r >= 360.0 {
        0.0
    } else {
        r
    }
}

/// tetra3's `Roll`: the rotation of celestial north relative to the image's
/// "up" (tetra3.py:1501), which is what cedar's proto carries.
///
/// `theta_rad` is the angle from the tangent-plane xi (East) axis to camera +X,
/// counter-clockwise, and equals `roll - 180` degrees. Verified against 16
/// corpus fields spanning the full rotation range (worst error 0.0025 deg).
///
/// When `parity_flip` is set, `theta_rad` is measured in the *mirror-corrected*
/// frame, so the sign of the roll inverts. Cedar's optics never mirror the
/// image, so this branch is not exercised in production -- and no corpus field
/// catches it either, since every synthetic field is generated `flip=False`.
/// It is here because getting it silently wrong costs up to 148 degrees.
pub fn roll_deg(theta_rad: f64, parity_flip: bool) -> f64 {
    let r = theta_rad.to_degrees() + 180.0;
    normalize_deg(if parity_flip { -r } else { r })
}

/// The 3x3 ICRS -> camera rotation matrix in *tetra3's* axis convention,
/// row-major, as `cedar_elements::astro_util::transform_to_image_coord` expects.
///
/// The two solvers use different camera frames:
///
/// | axis | tetra3rs (`qicrs2cam`) | tetra3 (`rotation_matrix`) |
/// |------|------------------------|----------------------------|
/// | 0    | +X, image right        | boresight                  |
/// | 1    | +Y, image down         | image left  (`-X`)         |
/// | 2    | +Z, boresight          | image up    (`-Y`)         |
///
/// so the tetra3 matrix is a signed row permutation of the tetra3rs one.
/// (`astro_util::compute_vector` builds `[1, (W/2 - x)s, (H/2 - y)s]`, which is
/// where "left" and "up" come from.)
///
/// Under `parity_flip` the returned matrix describes the mirror-corrected image,
/// exactly as `qicrs2cam` does — a mirrored frame is not a proper rotation and
/// has no faithful tetra3-style matrix. Cedar does not mirror.
pub fn tetra3_rotation_matrix(solution: &Solution) -> [f64; 9] {
    let m = solution.qicrs2cam.to_rotation_matrix();
    let e = |r: usize, c: usize| m[(r, c)] as f64;
    [
        e(2, 0), e(2, 1), e(2, 2),       // boresight  =  cam +Z
        -e(0, 0), -e(0, 1), -e(0, 2),    // image left = -cam +X
        -e(1, 0), -e(1, 1), -e(1, 2),    // image up   = -cam +Y
    ]
}

/// Cross product of two 3-vectors.
fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

/// Builds a tetra3rs attitude quaternion (`qicrs2cam`, the convention of
/// [`Solution::qicrs2cam`]) from a boresight sky position and roll, for use as a
/// tracking-mode [`SolveConfig::attitude_hint`].
///
/// This inverts the pose the adapter *reports*: `ra_deg`/`dec_deg` are the
/// image-center pointing (`Solution::pixel_to_world(0, 0)`), and `roll_deg` is
/// cedar's north position angle as produced by [`roll_deg`]. Reconstructing the
/// attitude from those three numbers is exact enough for a hint -- tracking uses
/// it only to center a catalog cone and to project stars for a nearest-neighbor
/// match, both slackened by `hint_uncertainty_rad`, and then refits the attitude
/// by SVD.
///
/// `qicrs2cam.to_rotation_matrix()` maps ICRS vectors into the camera frame
/// (+X right, +Y down, +Z boresight), so its rows are the camera axes expressed
/// in ICRS: row 0 = +X, row 1 = +Y, row 2 = boresight (see `track.rs` and
/// [`tetra3_rotation_matrix`]). We build those three ICRS rows and pass them to
/// `Quaternion::from_rotation_matrix`, the exact inverse of `to_rotation_matrix`.
///
/// Roll enters through `theta_rad = (roll - 180) deg` (the inverse of
/// [`roll_deg`]): camera +X lies in the tangent plane at angle `theta` from the
/// East axis toward North.
pub fn attitude_to_quaternion(ra_deg: f64, dec_deg: f64, roll_deg: f64) -> Quaternion {
    let ra = ra_deg.to_radians();
    let dec = dec_deg.to_radians();
    let theta = (roll_deg - 180.0).to_radians();

    // Boresight (+Z_cam) in ICRS.
    let b = [dec.cos() * ra.cos(), dec.cos() * ra.sin(), dec.sin()];
    // Tangent-plane basis at the boresight: East (increasing RA), North
    // (increasing Dec). Orthonormal and perpendicular to the boresight.
    let east = [-ra.sin(), ra.cos(), 0.0];
    let north = [-dec.sin() * ra.cos(), -dec.sin() * ra.sin(), dec.cos()];

    // Camera +X (image right): theta from East toward North.
    let (st, ct) = (theta.sin(), theta.cos());
    let x = [
        ct * east[0] + st * north[0],
        ct * east[1] + st * north[1],
        ct * east[2] + st * north[2],
    ];
    // Camera +Y (image down) completes the right-handed frame (X x Y = Z = b).
    let y = cross(b, x);

    let m = Matrix3::<f32>::new([
        [x[0] as f32, x[1] as f32, x[2] as f32],
        [y[0] as f32, y[1] as f32, y[2] as f32],
        [b[0] as f32, b[1] as f32, b[2] as f32],
    ]);
    Quaternion::from_rotation_matrix(&m)
}

/// The FOV seeds to try, in order, for one solve.
///
/// A known FOV (from the calibrator, or configured for the box camera) yields a
/// single seed. Blind, we walk a geometric ladder across the database's FOV
/// range: each seed covers a factor of `SEED_COVERS_ABOVE / SEED_COVERS_BELOW`
/// in true FOV, so the ladder tiles `[db_min, db_max]` with no gaps.
///
/// `SolveParams::fov_estimate` also carries a tolerance, which is deliberately
/// discarded: tetra3rs's `fov_max_error_rad` is a *rejection filter* around the
/// seed rather than a search width, and setting it narrows the usable seed range
/// from about +/-50% to +/-10% while making failures ~100x slower. See
/// `docs/04-fov-seed-sensitivity.md`.
pub fn fov_seeds(
    fov_estimate_deg: Option<f64>,
    db_min_fov_deg: f64,
    db_max_fov_deg: f64,
) -> Vec<f64> {
    if let Some(fov) = fov_estimate_deg {
        return vec![fov];
    }
    let mut seeds = Vec::new();
    let mut lo = db_min_fov_deg;
    while seeds.len() < MAX_BLIND_SEEDS {
        let seed = lo / SEED_COVERS_BELOW;
        seeds.push(seed);
        let covered_to = seed * SEED_COVERS_ABOVE;
        if covered_to >= db_max_fov_deg {
            break;
        }
        lo = covered_to;
    }
    seeds
}

/// Radians -> arcseconds. tetra3rs reports residuals in radians; the proto
/// documents `rmse`/`p90_error`/`max_error` as arcseconds.
fn rad_to_arcsec(rad: f32) -> f64 {
    (rad as f64).to_degrees() * 3600.0
}

/// Everything one solve needs, owned, so it can cross a `spawn_blocking`
/// boundary without borrowing from the caller.
struct SolveInputs {
    centroids: Vec<Centroid>,
    width: usize,
    height: usize,
    seeds: Vec<f64>,
    match_radius: f32,
    match_threshold: f64,
    match_max_error: Option<f32>,
    /// Tracking-mode attitude hint (`qicrs2cam`). When set, tetra3rs attempts a
    /// direct-correspondence solve against it before falling back to the blind
    /// pattern-hash search.
    attitude_hint: Option<Quaternion>,
    timeout: Duration,
    target_pixels: Vec<ImageCoord>,
    target_sky_coords: Vec<CelestialCoord>,
    return_matches: bool,
    return_catalog: bool,
    return_rotation_matrix: bool,
    distortion: f64,
}

pub struct Tetra3RsSolver {
    db: Arc<SolverDatabase>,
    /// Catalog id -> index into `db.star_catalog.stars()`. `Solution` reports
    /// matches by catalog id; the overlay needs the star's position and
    /// magnitude.
    star_by_id: Arc<HashMap<i64, usize>>,
    /// The box camera's known horizontal FOV, used when the caller solves blind.
    /// `None` falls back to the blind ladder over the database's FOV range.
    default_fov_deg: Option<f64>,
    cancel: Arc<AtomicBool>,
}

impl Tetra3RsSolver {
    pub fn new(db: SolverDatabase, default_fov_deg: Option<f64>) -> Self {
        let props = &db.props;
        info!(
            "Using tetra3rs solver: {} patterns, FOV {:.2}-{:.2} deg, mag limit {:.2}",
            props.num_patterns,
            props.min_fov_rad.to_degrees(),
            props.max_fov_rad.to_degrees(),
            props.star_max_magnitude,
        );
        match default_fov_deg {
            Some(fov) => info!("tetra3rs FOV seed: {fov:.3} deg"),
            None => info!(
                "tetra3rs has no FOV seed; solving blind over seeds {:?}",
                fov_seeds(
                    None,
                    props.min_fov_rad.to_degrees() as f64,
                    props.max_fov_rad.to_degrees() as f64
                )
            ),
        }

        let star_by_id = db
            .star_catalog_ids
            .iter()
            .enumerate()
            .map(|(idx, &id)| (id, idx))
            .collect();

        Tetra3RsSolver {
            db: Arc::new(db),
            star_by_id: Arc::new(star_by_id),
            default_fov_deg,
            cancel: Arc::new(AtomicBool::new(false)),
        }
    }

    /// Loads a database produced by `make-tetra3rs-db`.
    pub fn from_database_file(
        path: &str,
        default_fov_deg: Option<f64>,
    ) -> Result<Self, CanonicalError> {
        let db = SolverDatabase::load_from_file(path).map_err(|e| {
            failed_precondition_error(
                format!(
                    "loading tetra3rs database {path:?}: {e:?}. Build one with:\n  \
                     cargo run --release --bin make-tetra3rs-db -- \
                     --gaia_catalog <gaia_merged.bin> --out {path}"
                )
                .as_str(),
            )
        })?;
        Ok(Self::new(db, default_fov_deg))
    }

    fn db_fov_range_deg(&self) -> (f64, f64) {
        (
            self.db.props.min_fov_rad.to_degrees() as f64,
            self.db.props.max_fov_rad.to_degrees() as f64,
        )
    }
}

/// Runs the solve. Synchronous and CPU-bound; called on a blocking thread.
fn solve_blocking(
    db: &SolverDatabase,
    star_by_id: &HashMap<i64, usize>,
    inputs: SolveInputs,
    cancel: &AtomicBool,
) -> Result<PlateSolution, CanonicalError> {
    let started = Instant::now();
    let deadline = started + inputs.timeout;

    let mut last_status = SolveStatus::NoMatch;
    for (n, seed_fov_deg) in inputs.seeds.iter().enumerate() {
        if cancel.load(Ordering::Relaxed) {
            return Err(deadline_exceeded_error("plate solve canceled"));
        }
        let remaining = deadline.saturating_duration_since(Instant::now());
        if remaining.is_zero() {
            return Err(deadline_exceeded_error(
                format!("plate solve timed out after {} FOV seeds", n).as_str(),
            ));
        }

        let mut config =
            SolveConfig::new(seed_fov_deg.to_radians() as f32,
                             inputs.width as u32, inputs.height as u32);
        config.match_radius = inputs.match_radius;
        config.match_threshold = inputs.match_threshold;
        config.solve_timeout_ms = Some(remaining.as_millis() as u64);
        // fov_max_error_rad is deliberately None: it is a rejection filter, not
        // a search width. See fov_seeds() and docs/04-fov-seed-sensitivity.md.
        config.fov_max_error_rad = None;
        // match_max_error is the *query* tolerance on pattern edge ratios, not
        // the database's quantization. tetra3rs floors it at the database's
        // pattern_max_error (0.001), so leaving it None asks for a match as tight
        // as the database can represent -- fine for synthetic images, too tight
        // for real ones. Saturated stars in a real frame carry ~1-2 px of
        // centroid error, which is ~0.001 of edge ratio all by itself, and the
        // pattern hash then misses. Honor SolveParams (0.005, solve_engine.rs:528,
        // the same tolerance the Python tetra3 database quantizes at); measured on
        // benchmarks/take1 it recovers frames that 0.001 loses, regresses nothing,
        // and costs ~0.02 ms.
        config.match_max_error = inputs.match_max_error;

        // Tracking mode: with a hint set, tetra3rs solves by direct
        // correspondence against the hinted attitude (sub-millisecond, robust to
        // the sparse/low-SNR fields the pattern hash misses) and falls back to
        // lost-in-space on failure. strict_hint stays false so a stale or wrong
        // hint can never lose a frame the blind search would have solved.
        if let Some(hint) = inputs.attitude_hint {
            config.attitude_hint = Some(hint);
            config.hint_uncertainty_rad = HINT_UNCERTAINTY_DEG.to_radians();
        }

        match db.solve_from_centroids(&inputs.centroids, &config) {
            Ok(solution) => {
                return Ok(build_proto(db, star_by_id, &inputs, &solution, started));
            }
            Err(failure) => {
                // TooFew is a property of the input, not the seed: retrying
                // other seeds cannot help.
                if failure.status == SolveStatus::TooFew {
                    return Err(invalid_argument_error(
                        format!(
                            "too few centroids ({}) to form a pattern",
                            inputs.centroids.len()
                        )
                        .as_str(),
                    ));
                }
                last_status = failure.status;
            }
        }
    }

    match last_status {
        SolveStatus::Timeout => Err(deadline_exceeded_error(
            format!("plate solve timed out after {:?}", started.elapsed()).as_str(),
        )),
        _ => Err(not_found_error(
            format!(
                "no match for {} centroids over {} FOV seed(s)",
                inputs.centroids.len(),
                inputs.seeds.len()
            )
            .as_str(),
        )),
    }
}

/// Converts a tetra3rs `Solution` into cedar's `PlateSolution` proto.
///
/// `elapsed_from` is when the whole `solve_from_centroids` call began, so
/// `solve_time` covers every FOV seed tried plus this conversion -- the number
/// that matters to the box, and the analogue of what `Tetra3Solver` reports.
fn build_proto(
    db: &SolverDatabase,
    star_by_id: &HashMap<i64, usize>,
    inputs: &SolveInputs,
    solution: &Solution,
    elapsed_from: Instant,
) -> PlateSolution {
    let (w, h) = (inputs.width as f64, inputs.height as f64);
    // tetra3rs pixel coords are centered; cedar's ImageCoord is top-left origin.
    let to_centered = |x: f64, y: f64| (x - w / 2.0, y - h / 2.0);
    let to_corner = |x: f64, y: f64| (x + w / 2.0, y + h / 2.0);

    let (ra, dec) = solution.pixel_to_world(0.0, 0.0);

    let mut plate_solution = PlateSolution {
        image_sky_coord: Some(CelestialCoord {
            ra,
            dec,
            epoch: None,
        }),
        roll: roll_deg(solution.theta_rad, solution.parity_flip),
        fov: solution.fov_rad.to_degrees() as f64,
        // solve_engine.rs:1285 unwraps this. tetra3rs fits no distortion term,
        // so echo what was requested (production passes 0.0).
        distortion: Some(inputs.distortion),
        rmse: rad_to_arcsec(solution.rmse_rad),
        p90_error: rad_to_arcsec(solution.p90e_rad),
        max_error: rad_to_arcsec(solution.max_err_rad),
        num_matches: solution.num_matches as i32,
        prob: solution.prob,
        epoch_equinox: db.props.epoch_equinox as i32,
        epoch_proper_motion: db.props.epoch_proper_motion_year,
        solve_time: prost_types::Duration::try_from(elapsed_from.elapsed()).ok(),
        // tetra3rs does not surface the 4 stars of the matched pattern.
        pattern_centroids: Vec::new(),
        ..Default::default()
    };

    // SolveExtension.target_pixel -> where those pixels point on the sky.
    for tp in &inputs.target_pixels {
        let (cx, cy) = to_centered(tp.x, tp.y);
        let (ra, dec) = solution.pixel_to_world(cx, cy);
        plate_solution
            .target_sky_coord
            .push(CelestialCoord { ra, dec, epoch: None });
    }

    // SolveExtension.target_sky_coord -> where those coords land in the image.
    // (-1, -1) when the target is not in frame, per the proto.
    for tsc in &inputs.target_sky_coords {
        let pixel = solution
            .world_to_pixel(tsc.ra, tsc.dec)
            .map(|(cx, cy)| to_corner(cx, cy))
            .filter(|&(x, y)| x >= 0.0 && x < w && y >= 0.0 && y < h)
            .unwrap_or((-1.0, -1.0));
        plate_solution.target_pixel.push(ImageCoord {
            x: pixel.0,
            y: pixel.1,
        });
    }

    if inputs.return_rotation_matrix {
        plate_solution.rotation_matrix = tetra3_rotation_matrix(solution).to_vec();
    }

    if inputs.return_matches {
        for (&cat_id, &cent_idx) in solution
            .matched_catalog_ids
            .iter()
            .zip(solution.matched_centroid_indices.iter())
        {
            let (Some(&star_idx), Some(centroid)) =
                (star_by_id.get(&cat_id), inputs.centroids.get(cent_idx))
            else {
                continue;
            };
            let star = &db.star_catalog.stars()[star_idx];
            let (x, y) = to_corner(centroid.x as f64, centroid.y as f64);
            plate_solution.matched_stars.push(StarInfo {
                pixel: Some(ImageCoord { x, y }),
                sky_coord: Some(CelestialCoord {
                    ra: (star.ra_rad as f64).to_degrees(),
                    dec: (star.dec_rad as f64).to_degrees(),
                    epoch: None,
                }),
                mag: star.mag,
            });
        }
    }

    if inputs.return_catalog {
        plate_solution.catalog_stars =
            catalog_stars_in_frame(db, inputs, solution, ra, dec);
    }

    plate_solution
}

/// Catalog stars falling inside the frame, brightest first. Feeds the Setup-mode
/// star overlay (serve_engine.rs:912).
fn catalog_stars_in_frame(
    db: &SolverDatabase,
    inputs: &SolveInputs,
    solution: &Solution,
    center_ra: f64,
    center_dec: f64,
) -> Vec<StarInfo> {
    let (w, h) = (inputs.width as f64, inputs.height as f64);
    // Cone radius: the image half-diagonal, in the solved (refined) pixel scale.
    let f = solution.camera_model.focal_length_px;
    let half_diag_px = ((w / 2.0).powi(2) + (h / 2.0).powi(2)).sqrt();
    let radius_rad = (half_diag_px / f).atan() as f32;

    let indices = db.star_catalog.query_indices(
        (center_ra as f32).to_radians(),
        (center_dec as f32).to_radians(),
        radius_rad,
    );
    let all = db.star_catalog.stars();
    let mut stars: Vec<&tetra3::Star> = indices.iter().map(|&i| &all[i]).collect();
    stars.sort_by(|a, b| a.mag.total_cmp(&b.mag));

    let mut out = Vec::new();
    for star in stars {
        if out.len() >= MAX_CATALOG_STARS {
            break;
        }
        let (ra, dec) = (
            (star.ra_rad as f64).to_degrees(),
            (star.dec_rad as f64).to_degrees(),
        );
        let Some((cx, cy)) = solution.world_to_pixel(ra, dec) else {
            continue;
        };
        let (x, y) = (cx + w / 2.0, cy + h / 2.0);
        if x < 0.0 || x >= w || y < 0.0 || y >= h {
            continue;
        }
        out.push(StarInfo {
            pixel: Some(ImageCoord { x, y }),
            sky_coord: Some(CelestialCoord { ra, dec, epoch: None }),
            mag: star.mag,
        });
    }
    out
}

#[async_trait]
impl SolverTrait for Tetra3RsSolver {
    async fn solve_from_centroids(
        &self,
        star_centroids: &[ImageCoord],
        width: usize,
        height: usize,
        extension: &SolveExtension,
        params: &SolveParams,
        imu_estimate: Option<EquatorialCoordinates>,
    ) -> Result<PlateSolution, CanonicalError> {
        // A fresh solve is not retroactively canceled by an earlier cancel().
        self.cancel.store(false, Ordering::Relaxed);

        // Tracking-mode hint: the engine's explicit previous-frame attitude wins;
        // fall back to an IMU-fused estimate if that is all that is available.
        // Both are EquatorialCoordinates (boresight ra/dec + north-up roll), which
        // is exactly what attitude_to_quaternion inverts.
        let attitude_hint = params.attitude_hint.or(imu_estimate).map(|ec| {
            attitude_to_quaternion(ec.ra, ec.dec, ec.north_roll_angle)
        });

        // tetra3rs centroids are measured from the image center, +X right, +Y
        // down; cedar's ImageCoord origin is the top-left corner.
        //
        // `mass` is the brightness tetra3rs sorts by, and `SolverTrait` does not
        // carry it -- solve_engine hands us bare ImageCoords (solve_engine.rs:938).
        // Passing None is nonetheless correct here: cedar_detect already returns
        // star_candidates brightest-first (cedar-detect/src/algorithm.rs:1017),
        // solve_engine preserves that order, and tetra3rs's
        // `sort_indices_by_brightness` maps every None to the same key and sorts
        // stably -- so the incoming order survives. Reorder the centroids upstream
        // and the pattern search silently starts with the wrong stars.
        let (w, h) = (width as f64, height as f64);
        let centroids: Vec<Centroid> = star_centroids
            .iter()
            .map(|sc| Centroid {
                x: (sc.x - w / 2.0) as f32,
                y: (sc.y - h / 2.0) as f32,
                mass: None,
                cov: None,
            })
            .collect();

        let distortion = params.distortion.unwrap_or(0.0);
        if distortion.abs() > 1e-9 {
            warn!(
                "tetra3rs does not fit a distortion term; ignoring \
                 requested distortion {distortion}"
            );
        }

        let (db_min, db_max) = self.db_fov_range_deg();
        let seeds = fov_seeds(
            params.fov_estimate.map(|(fov, _tol)| fov).or(self.default_fov_deg),
            db_min,
            db_max,
        );

        let inputs = SolveInputs {
            centroids,
            width,
            height,
            seeds,
            match_radius: params.match_radius.unwrap_or(0.01) as f32,
            match_threshold: params.match_threshold.unwrap_or(1e-5),
            match_max_error: params.match_max_error.map(|e| e as f32),
            attitude_hint,
            timeout: params.solve_timeout.unwrap_or(DEFAULT_TIMEOUT),
            target_pixels: extension.target_pixel.clone().unwrap_or_default(),
            target_sky_coords: extension.target_sky_coord.clone().unwrap_or_default(),
            return_matches: extension.return_matches,
            return_catalog: extension.return_catalog,
            return_rotation_matrix: extension.return_rotation_matrix,
            distortion,
        };

        let db = self.db.clone();
        let star_by_id = self.star_by_id.clone();
        let cancel = self.cancel.clone();
        tokio::task::spawn_blocking(move || {
            solve_blocking(&db, &star_by_id, inputs, &cancel)
        })
        .await
        .map_err(|e| internal_error(format!("solve task panicked: {e:?}").as_str()))?
    }

    /// tetra3rs exposes no interrupt hook, only a timeout. This stops the next
    /// FOV seed from starting; an in-flight seed runs to completion. Typical
    /// solves are under a millisecond, so there is little to interrupt.
    fn cancel(&self) {
        self.cancel.store(true, Ordering::Relaxed);
    }

    fn default_timeout(&self) -> Duration {
        DEFAULT_TIMEOUT
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tetra3::{Quaternion, Vector3};

    // ---- roll ------------------------------------------------------------

    /// `theta_rad` is the generator's rotation angle (verified on 16 corpus
    /// fields), and cedar's pinned convention is `roll == (180 + rotation) % 360`.
    #[test]
    fn roll_matches_the_pinned_convention() {
        for rotation_deg in [-180.0, -45.0, 0.0, 30.0, 153.0, 179.5] {
            let expected = (180.0f64 + rotation_deg).rem_euclid(360.0);
            let got = roll_deg(rotation_deg.to_radians(), false);
            assert!(
                (got - expected).abs() < 1e-9,
                "rotation {rotation_deg}: got {got}, want {expected}"
            );
        }
    }

    /// A mirrored image negates the roll. No corpus field exercises this (they
    /// are all generated flip=False), and cedar's optics never mirror -- but
    /// the unnegated form is off by up to 148 deg when parity does flip.
    #[test]
    fn roll_negates_under_parity_flip() {
        for rotation_deg in [-45.0f64, 0.0, 37.0, 143.0] {
            let theta = rotation_deg.to_radians();
            let want = (-(180.0 + rotation_deg)).rem_euclid(360.0);
            let got = roll_deg(theta, true);
            assert!(
                (got - want).abs() < 1e-9 || (got - want).abs() > 359.999_999_999,
                "rotation {rotation_deg}: got {got}, want {want}"
            );
        }
    }

    /// `(-1e-15).rem_euclid(360.0)` is exactly `360.0`, which is outside the
    /// documented `[0, 360)` range and breaks naive comparisons downstream.
    #[test]
    fn roll_never_returns_360() {
        assert_eq!(normalize_deg(-1e-15), 0.0);
        assert_eq!(normalize_deg(360.0), 0.0);
        assert_eq!(roll_deg((-180.0f64).to_radians(), false), 0.0);
    }

    // ---- rotation matrix -------------------------------------------------

    fn solution_with_attitude(q: Quaternion) -> Solution {
        Solution {
            qicrs2cam: q,
            fov_rad: 0.2,
            num_matches: 10,
            rmse_rad: 0.0,
            p90e_rad: 0.0,
            max_err_rad: 0.0,
            prob: 0.0,
            solve_time_ms: 0.0,
            parity_flip: false,
            matched_catalog_ids: vec![],
            matched_centroid_indices: vec![],
            cd_matrix: [[0.0; 2]; 2],
            crval_rad: [0.0; 2],
            camera_model: tetra3::CameraModel::from_fov(0.2, 100, 100),
            theta_rad: 0.0,
        }
    }

    fn mat_vec(m: &[f64; 9], v: [f64; 3]) -> [f64; 3] {
        std::array::from_fn(|r| (0..3).map(|c| m[r * 3 + c] * v[c]).sum())
    }

    fn det(m: &[f64; 9]) -> f64 {
        m[0] * (m[4] * m[8] - m[5] * m[7]) - m[1] * (m[3] * m[8] - m[5] * m[6])
            + m[2] * (m[3] * m[7] - m[4] * m[6])
    }

    /// Whatever the attitude, the matrix must be a proper rotation -- otherwise
    /// astro_util's transforms silently mirror the sky.
    ///
    /// Tolerance is f32-scale because `qicrs2cam` is an f32 quaternion; the
    /// matrix is only ever used to project catalog stars onto the display image,
    /// where 1e-6 rad is far below a pixel.
    #[test]
    fn rotation_matrix_is_a_proper_rotation() {
        for q in [
            Quaternion::identity(),
            Quaternion::rotx(0.4),
            Quaternion::roty(-1.1),
            Quaternion::rotz(2.7),
            Quaternion::from_axis_angle(Vector3::from_array([0.3, -0.5, 0.81]).normalize(), 1.3),
        ] {
            let m = tetra3_rotation_matrix(&solution_with_attitude(q));
            assert!((det(&m) - 1.0).abs() < 1e-6, "det {} != 1", det(&m));
            // Rows orthonormal.
            for r in 0..3 {
                let row: [f64; 3] = std::array::from_fn(|c| m[r * 3 + c]);
                let norm: f64 = row.iter().map(|x| x * x).sum::<f64>().sqrt();
                assert!((norm - 1.0).abs() < 1e-6, "row {r} norm {norm}");
            }
        }
    }

    /// The defining property: tetra3's frame is (boresight, left, up), so the
    /// camera boresight must map to [1, 0, 0], image-right to [0, -1, 0], and
    /// image-down to [0, 0, -1].
    #[test]
    fn rotation_matrix_maps_camera_axes_to_tetra3_axes() {
        let q = Quaternion::from_axis_angle(Vector3::from_array([0.3, -0.5, 0.81]).normalize(), 1.3);
        let sol = solution_with_attitude(q);
        let m = tetra3_rotation_matrix(&sol);
        let inv = q.inverse();

        // ICRS directions of the camera's own axes.
        let axis_icrs = |v: [f32; 3]| {
            let r = inv * Vector3::from_array(v);
            [r[0] as f64, r[1] as f64, r[2] as f64]
        };

        let close = |a: [f64; 3], b: [f64; 3]| {
            a.iter().zip(b.iter()).all(|(x, y)| (x - y).abs() < 1e-6)
        };

        let boresight = mat_vec(&m, axis_icrs([0.0, 0.0, 1.0]));
        assert!(close(boresight, [1.0, 0.0, 0.0]), "boresight -> {boresight:?}");

        let right = mat_vec(&m, axis_icrs([1.0, 0.0, 0.0]));
        assert!(close(right, [0.0, -1.0, 0.0]), "image right -> {right:?}");

        let down = mat_vec(&m, axis_icrs([0.0, 1.0, 0.0]));
        assert!(close(down, [0.0, 0.0, -1.0]), "image down -> {down:?}");
    }

    // ---- attitude hint ---------------------------------------------------

    fn row(m: &numeris::Matrix3<f32>, r: usize) -> [f64; 3] {
        [m[(r, 0)] as f64, m[(r, 1)] as f64, m[(r, 2)] as f64]
    }

    fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
        a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
    }

    /// The reconstructed attitude must point its boresight at (ra, dec) and be a
    /// proper rotation. Boresight placement is the half of the convention that is
    /// checkable without a solve; the roll sign is anchored end-to-end against a
    /// real tetra3rs solve in `e2e_tetra3rs_tracking.rs`.
    #[test]
    fn attitude_hint_boresight_matches_radec() {
        for (ra, dec) in [
            (0.0, 0.0),
            (123.4, -25.0),
            (359.9, 89.0),
            (221.0, -25.0),
            (270.0, -89.5),
        ] {
            let q = attitude_to_quaternion(ra, dec, 137.0);
            let m = q.to_rotation_matrix();
            let (rr, dr) = (ra.to_radians(), dec.to_radians());
            let want = [dr.cos() * rr.cos(), dr.cos() * rr.sin(), dr.sin()];
            let boresight = row(&m, 2); // row 2 = boresight (see track.rs).
            let cos_sep = dot(boresight, want).clamp(-1.0, 1.0);
            assert!(
                cos_sep > 0.999_999,
                "ra {ra} dec {dec}: boresight off by {:.4}'",
                cos_sep.acos().to_degrees() * 60.0
            );
        }
    }

    /// `roll` recovered from the reconstructed attitude with the convention
    /// `roll_deg` documents (theta = angle East->camera-+X) round-trips. This
    /// pins that `attitude_to_quaternion` is the exact inverse of the pose the
    /// adapter reports.
    #[test]
    fn attitude_hint_roll_round_trips() {
        for &(ra, dec) in &[(50.0, 10.0), (221.0, -25.0), (200.0, 60.0)] {
            for roll in [0.0, 37.0, 179.9, 200.0, 350.0] {
                let q = attitude_to_quaternion(ra, dec, roll);
                let m = q.to_rotation_matrix();
                let (rr, dr) = (ra.to_radians(), dec.to_radians());
                let east = [-rr.sin(), rr.cos(), 0.0];
                let north = [-dr.sin() * rr.cos(), -dr.sin() * rr.sin(), dr.cos()];
                let x = row(&m, 0); // row 0 = camera +X (image right).
                let theta = dot(x, north).atan2(dot(x, east));
                let got = normalize_deg(theta.to_degrees() + 180.0);
                let want = normalize_deg(roll);
                let err = (got - want).abs().min(360.0 - (got - want).abs());
                assert!(err < 1e-3, "ra {ra} dec {dec} roll {roll}: got {got}");
            }
        }
    }

    /// Whatever the inputs, the hint must be a proper rotation (det +1,
    /// orthonormal rows) or tetra3rs's projection would mirror the sky.
    #[test]
    fn attitude_hint_is_a_proper_rotation() {
        for &(ra, dec, roll) in &[
            (0.0, 0.0, 0.0),
            (123.0, 45.0, 210.0),
            (300.0, -60.0, 95.0),
            (359.0, 89.0, 15.0),
        ] {
            let m = attitude_to_quaternion(ra, dec, roll).to_rotation_matrix();
            let rows = [row(&m, 0), row(&m, 1), row(&m, 2)];
            for r in 0..3 {
                assert!((dot(rows[r], rows[r]) - 1.0).abs() < 1e-5, "row {r} not unit");
                for c in (r + 1)..3 {
                    assert!(dot(rows[r], rows[c]).abs() < 1e-5, "rows {r},{c} not orthogonal");
                }
            }
            // Right-handed: row0 x row1 == row2.
            let x_cross_y = cross(rows[0], rows[1]);
            assert!(dot(x_cross_y, rows[2]) > 0.999_99, "not right-handed");
        }
    }

    // ---- FOV seeds -------------------------------------------------------

    #[test]
    fn a_known_fov_yields_exactly_one_seed() {
        assert_eq!(fov_seeds(Some(12.66), 10.0, 30.0), vec![12.66]);
    }

    /// Blind seeds must tile the database's FOV range: every FOV in
    /// [min, max] has to fall inside some seed's coverage window.
    #[test]
    fn blind_seeds_tile_the_database_range() {
        for (min, max) in [(10.0, 30.0), (5.0, 60.0), (12.0, 12.0), (1.0, 40.0)] {
            let seeds = fov_seeds(None, min, max);
            assert!(!seeds.is_empty(), "no seeds for {min}..{max}");
            assert!(seeds.len() <= MAX_BLIND_SEEDS);

            let covered = |fov: f64| {
                seeds.iter().any(|s| {
                    fov >= s * SEED_COVERS_BELOW - 1e-9
                        && fov <= s * SEED_COVERS_ABOVE + 1e-9
                })
            };
            for i in 0..=100 {
                let fov = min + (max - min) * i as f64 / 100.0;
                assert!(covered(fov), "{fov} uncovered in {min}..{max} by {seeds:?}");
            }
        }
    }

    /// The corpus/box case: a 10-30 deg database needs no more than a couple of
    /// tries when solving blind.
    #[test]
    fn blind_ladder_is_short_for_the_shipped_range() {
        assert!(fov_seeds(None, 10.0, 30.0).len() <= 2);
    }

    // ---- units -----------------------------------------------------------

    #[test]
    fn residuals_convert_to_arcseconds() {
        // 1 arcsec in radians.
        let one_arcsec = (1.0f64 / 3600.0).to_radians() as f32;
        assert!((rad_to_arcsec(one_arcsec) - 1.0).abs() < 1e-4);
    }
}
