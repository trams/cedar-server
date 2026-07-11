// Loading the *real* benchmark corpus: frames captured off a Cedar box's camera
// by `CEDAR_BENCH_DIR` capture (see docs/enabling-benchmark-capture.md).
//
// This corpus differs from the synthetic one in a way that changes what a test
// over it can claim. The synthetic manifest carries the pose the image was
// *rendered* from -- real ground truth. This manifest carries the pose the Pi's
// own tetra3 *reported*. So a run against it measures **agreement with tetra3**,
// not correctness. A bias both solvers share is invisible here; a bias either one
// has alone shows up as disagreement, without saying which is at fault.
//
// What makes the comparison sound is that `write_bench_frame` (solve_engine.rs:181)
// records `p.roll` and `p.fov` straight off the same `PlateSolution` proto that
// `SolveEngine::get_next_result` hands this harness, and saves
// `detect_result.captured_image` -- the full-resolution frame DetectEngine
// consumed. Same image in, same proto fields out, no reprojection in between.

use std::path::{Path, PathBuf};

use cedar_elements::astro_util::angular_separation;
use cedar_server::solve_engine::PlateSolution;
use image::GrayImage;

use super::harness::{circular_diff_deg, Outcome};

/// What the Pi's tetra3 reported for a frame. Absent when it did not solve.
#[derive(Debug, Clone, PartialEq)]
pub struct Truth {
    pub ra_deg: f64,
    pub dec_deg: f64,
    pub roll_deg: f64,
    pub fov_deg: f64,
    pub rmse_arcsec: f64,
    pub num_matches: i32,
    pub solve_ms: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct BenchFrame {
    pub filename: String,
    pub frame_id: i32,
    pub solution_id: i32,
    pub exposure_ms: u32,
    pub width: u32,
    pub height: u32,
    pub truth: Option<Truth>,
}

impl BenchFrame {
    /// Filename without its image extension, for report rows and CSV keys.
    /// Handles both `.png` (current) and `.bmp` (legacy corpora).
    pub fn name(&self) -> String {
        self.filename
            .strip_suffix(".png")
            .or_else(|| self.filename.strip_suffix(".bmp"))
            .unwrap_or(&self.filename)
            .to_string()
    }
}

/// The exact header `write_bench_frame` (solve_engine.rs:169) emits. Parsing is
/// positional, so header drift must be a hard error, not a silent column shift.
const BENCH_HEADER: &str = "filename,frame_id,solution_id,exposure_ms,width,height,solved,\
                            from_imu,ra_deg,dec_deg,roll_deg,fov_deg,rmse_arcsec,num_matches,\
                            solve_ms,readout_time_iso";

pub fn parse_bench_manifest(text: &str) -> Result<Vec<BenchFrame>, String> {
    let mut lines = text.lines().filter(|l| !l.trim().is_empty());

    let header = lines.next().ok_or("bench manifest.csv is empty")?.trim();
    if header != BENCH_HEADER {
        return Err(format!(
            "bench manifest.csv header drift.\n  expected: {BENCH_HEADER}\n  found:    {header}"
        ));
    }

    let mut frames = Vec::new();
    for (n, line) in lines.enumerate() {
        let row: Vec<&str> = line.trim().split(',').collect();
        if row.len() != 16 {
            return Err(format!(
                "bench manifest.csv row {} has {} columns, expected 16",
                n + 2,
                row.len()
            ));
        }
        let num = |i: usize| -> Result<f64, String> {
            row[i]
                .parse::<f64>()
                .map_err(|e| format!("bench manifest.csv row {} col {}: {e}", n + 2, i))
        };

        // IMU interpolations are not fresh camera solves. write_bench_frame skips
        // them, so this is a defensive drop rather than an expected path.
        if row[7] == "true" {
            continue;
        }

        let truth = match row[6] {
            "true" => Some(Truth {
                ra_deg: num(8)?,
                dec_deg: num(9)?,
                roll_deg: num(10)?,
                fov_deg: num(11)?,
                rmse_arcsec: num(12)?,
                num_matches: num(13)? as i32,
                solve_ms: num(14)?,
            }),
            "false" => None,
            other => {
                return Err(format!(
                    "bench manifest.csv row {}: `solved` is {other:?}, expected true/false",
                    n + 2
                ))
            }
        };

        frames.push(BenchFrame {
            filename: row[0].to_string(),
            frame_id: num(1)? as i32,
            solution_id: num(2)? as i32,
            exposure_ms: num(3)? as u32,
            width: num(4)? as u32,
            height: num(5)? as u32,
            truth,
        });
    }
    Ok(frames)
}

/// The captured-frame corpus directory, or why it cannot be found.
///
/// Defaults to `<repo_root>/benchmarks/take1`, the corpus pulled off the Pi;
/// `CEDAR_E2E_BENCH_DIR` overrides. Not to be confused with `CEDAR_BENCH_DIR`,
/// which makes a *running server* write a corpus and must stay unset here.
pub fn bench_dir() -> Result<PathBuf, String> {
    let dir = match std::env::var_os("CEDAR_E2E_BENCH_DIR") {
        Some(d) => PathBuf::from(d),
        None => Path::new(env!("CARGO_MANIFEST_DIR"))
            .ancestors()
            .nth(2)
            .expect("CARGO_MANIFEST_DIR has >=2 ancestors")
            .join("benchmarks/take1"),
    };
    if !dir.join("manifest.csv").is_file() {
        return Err(format!(
            "no manifest.csv under {} -- point CEDAR_E2E_BENCH_DIR at a corpus \
             captured per docs/enabling-benchmark-capture.md",
            dir.display()
        ));
    }
    Ok(dir)
}

pub fn load_manifest(dir: &Path) -> Result<Vec<BenchFrame>, String> {
    let path = dir.join("manifest.csv");
    let text = std::fs::read_to_string(&path)
        .map_err(|e| format!("reading {}: {e}", path.display()))?;
    parse_bench_manifest(&text)
}

/// Loads a captured frame as 8-bit grayscale -- byte-for-byte what DetectEngine
/// saw on the Pi.
pub fn load_image(dir: &Path, frame: &BenchFrame) -> Result<GrayImage, String> {
    let path = dir.join(&frame.filename);
    let img = image::open(&path)
        .map_err(|e| format!("opening {}: {e}", path.display()))?
        .to_luma8();
    let (w, h) = (img.width(), img.height());
    if (w, h) != (frame.width, frame.height) {
        return Err(format!(
            "{}: image is {w}x{h}, manifest says {}x{}",
            path.display(),
            frame.width,
            frame.height
        ));
    }
    Ok(img)
}

/// Every `stride`th frame. Deterministic, and keeps the sampled frames spread
/// across the whole session rather than clustered in one patch of sky.
pub fn subsample<T: Clone>(items: &[T], stride: usize) -> Vec<T> {
    let stride = stride.max(1);
    items.iter().step_by(stride).cloned().collect()
}

/// Reads a positive-integer env var, falling back to `default`.
pub fn env_usize(var: &str, default: usize) -> usize {
    std::env::var(var)
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|n| *n > 0)
        .unwrap_or(default)
}

/// Angular separation between two sky positions, in degrees.
pub fn separation_deg(ra1: f64, dec1: f64, ra2: f64, dec2: f64) -> f64 {
    angular_separation(
        ra1.to_radians(),
        dec1.to_radians(),
        ra2.to_radians(),
        dec2.to_radians(),
    )
    .to_degrees()
}

/// Scores one frame against tetra3's recorded answer.
///
/// Deliberately shaped as a `harness::Outcome` so the existing `Report` machinery
/// applies unchanged -- but read every column as *disagreement with tetra3*, not
/// as error. `frame.truth` must be `Some`; frames tetra3 could not solve have no
/// ground truth and belong in the false-positive test instead.
pub fn evaluate_agreement(frame: &BenchFrame, ps: &PlateSolution) -> Outcome {
    let truth = frame
        .truth
        .as_ref()
        .expect("evaluate_agreement needs a tetra3-solved frame");
    let num_centroids = ps.detect_result.star_candidates.len();

    let Some(p) = ps.plate_solution.as_ref() else {
        return Outcome {
            name: frame.name(),
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

    Outcome {
        name: frame.name(),
        solved: true,
        center_arcmin: separation_deg(truth.ra_deg, truth.dec_deg, coord.ra, coord.dec) * 60.0,
        roll_err_deg: circular_diff_deg(p.roll, truth.roll_deg),
        fov_err_frac: (p.fov - truth.fov_deg).abs() / truth.fov_deg,
        solve_time_ms: p
            .solve_time
            .as_ref()
            .map(|d| d.seconds as f64 * 1000.0 + d.nanos as f64 / 1.0e6)
            .unwrap_or(f64::NAN),
        num_matches: p.num_matches,
        num_centroids,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE: &str = "filename,frame_id,solution_id,exposure_ms,width,height,solved,from_imu,\
        ra_deg,dec_deg,roll_deg,fov_deg,rmse_arcsec,num_matches,solve_ms,readout_time_iso\n\
        img_000000_200ms_20260710_053424.bmp,1130,0,200,1920,1080,false,false,,,,,,,,2026-07-10T05:34:24+0100\n\
        img_000001_190ms_20260710_053426.bmp,1141,11,190,1920,1080,true,false,233.165675,-25.029172,93.5498,12.7954,18.74,14,62.06,2026-07-10T05:34:26+0100\n";

    #[test]
    fn parses_solved_and_unsolved_rows() {
        let frames = parse_bench_manifest(SAMPLE).expect("parse");
        assert_eq!(frames.len(), 2);

        assert_eq!(frames[0].name(), "img_000000_200ms_20260710_053424");
        assert_eq!(frames[0].exposure_ms, 200);
        assert!(
            frames[0].truth.is_none(),
            "an unsolved row must carry no ground truth"
        );

        let t = frames[1].truth.as_ref().expect("solved row has truth");
        assert_eq!(frames[1].frame_id, 1141);
        assert_eq!(t.ra_deg, 233.165675);
        assert_eq!(t.dec_deg, -25.029172);
        assert_eq!(t.roll_deg, 93.5498);
        assert_eq!(t.fov_deg, 12.7954);
        assert_eq!(t.num_matches, 14);
        assert_eq!(t.solve_ms, 62.06);
    }

    /// The one row shape that could silently corrupt a run: an IMU interpolation
    /// carries a pose but was never a fresh camera frame.
    #[test]
    fn drops_imu_interpolated_rows() {
        let imu = SAMPLE.replace(",true,false,233.165675", ",true,true,233.165675");
        let frames = parse_bench_manifest(&imu).expect("parse");
        assert_eq!(frames.len(), 1);
        assert!(frames[0].truth.is_none());
    }

    #[test]
    fn rejects_header_drift() {
        let bad = "filename,frame_id\nimg.bmp,1\n";
        assert!(parse_bench_manifest(bad).unwrap_err().contains("header drift"));
    }

    #[test]
    fn rejects_short_row() {
        let bad = format!("{BENCH_HEADER}\nimg.bmp,1,2\n");
        assert!(parse_bench_manifest(&bad).unwrap_err().contains("expected 16"));
    }

    #[test]
    fn rejects_unknown_solved_flag() {
        let bad = SAMPLE.replace(",1080,true,false,233", ",1080,maybe,false,233");
        assert!(parse_bench_manifest(&bad).unwrap_err().contains("expected true/false"));
    }

    #[test]
    fn subsample_takes_every_nth() {
        let v: Vec<i32> = (0..10).collect();
        assert_eq!(subsample(&v, 3), vec![0, 3, 6, 9]);
        assert_eq!(subsample(&v, 1), v);
        assert_eq!(subsample(&v, 0), v, "stride 0 must not divide by zero");
    }

    #[test]
    fn separation_of_a_point_with_itself_is_zero() {
        assert!(separation_deg(233.1, -25.0, 233.1, -25.0) < 1e-9);
    }

    #[test]
    fn separation_along_a_meridian_is_the_dec_difference() {
        assert!((separation_deg(10.0, 20.0, 10.0, 21.5) - 1.5).abs() < 1e-9);
    }
}
