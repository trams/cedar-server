// Copyright (c) 2026 Steven Rosenthal smr@dt3.org
// See LICENSE file in root directory for license terms.

//! Observation log: an automatic, append-only record of what was observed
//! during a session, written to a local JSON Lines file. Two event types share
//! one file, distinguished by a `type` field:
//!
//! * `"solve"` — one record per *real* camera plate solve (the chain of
//!   pointing coordinates), throttled to bound file size. IMU-interpolated
//!   results are skipped; they are not fresh observations.
//! * `"goto"` — one record per GoTo request (`initiate_slew` / `stop_slew`),
//!   the strong user-intent signal. Never throttled.
//!
//! This is deliberately modeled on the benchmark-corpus capture in
//! `solve_engine.rs` (`BenchConfig`, `write_bench_frame`): config from the
//! environment, synchronous append, and — crucially — I/O errors are logged and
//! swallowed, never propagated, so logging can never break the solve loop.

use std::io::Write as _;
use std::path::PathBuf;
use std::sync::Mutex;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use chrono::{DateTime, Local};
use log::{info, warn};
use serde::Serialize;

use crate::solve_engine::PlateSolution;
use cedar_elements::cedar_common::CelestialCoord;

// Default log file, relative to the process working directory. On the Pi the
// service runs with WorkingDirectory=/home/cedar/run, so this resolves to
// /home/cedar/run/cedar_observation_log.jsonl (next to the benchmark corpus).
const DEFAULT_PATH: &str = "./cedar_observation_log.jsonl";

// Default minimum wall-clock interval between logged solve records.
const DEFAULT_THROTTLE_SECS: f64 = 1.0;

#[derive(Clone, Debug)]
struct ObsLogConfig {
    // Output JSON Lines file. Created (with parent dirs) on first write.
    path: PathBuf,
    // Minimum interval between logged solve records. GoTo records ignore this.
    throttle: Duration,
    // When false, all logging calls are no-ops (feature disabled).
    enabled: bool,
}

// Pure config parse, decoupled from std::env so it is unit-testable without
// racing on process-global environment variables. See `from_env`.
//   CEDAR_OBSLOG               "0"/"false"/"no" disables (default: enabled)
//   CEDAR_OBSLOG_PATH          output file (default ./cedar_observation_log.jsonl)
//   CEDAR_OBSLOG_INTERVAL_SECS solve-record throttle seconds (default 1.0)
fn parse_config<F: Fn(&str) -> Option<String>>(get: F) -> ObsLogConfig {
    let enabled = !matches!(
        get("CEDAR_OBSLOG").as_deref(),
        Some("0") | Some("false") | Some("no")
    );
    let path = get("CEDAR_OBSLOG_PATH")
        .filter(|s| !s.is_empty())
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(DEFAULT_PATH));
    let throttle_secs = get("CEDAR_OBSLOG_INTERVAL_SECS")
        .and_then(|s| s.parse::<f64>().ok())
        .filter(|v| *v >= 0.0)
        .unwrap_or(DEFAULT_THROTTLE_SECS);
    ObsLogConfig {
        path,
        throttle: Duration::from_secs_f64(throttle_secs),
        enabled,
    }
}

// A logged plate solve. Sky fields are None (and omitted from the JSON) for
// frames that did not solve; `solved` distinguishes the two cases. Only real
// camera solves are recorded — IMU interpolations are filtered out upstream.
#[derive(Serialize)]
struct SolveRecord {
    #[serde(rename = "type")]
    kind: &'static str,
    // ISO-8601 local time of the frame's readout (capture), not the write.
    time: String,
    solution_id: i32,
    frame_id: i32,
    solved: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    ra_deg: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    dec_deg: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    roll_deg: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    fov_deg: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    rmse_arcsec: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    num_matches: Option<i32>,
    // Solver-internal time (from the solve proto), milliseconds.
    #[serde(skip_serializing_if = "Option::is_none")]
    solve_ms: Option<f64>,
    // End-to-end capture->solve-finish latency, milliseconds.
    #[serde(skip_serializing_if = "Option::is_none")]
    pipeline_ms: Option<f64>,
}

// A logged GoTo request. `action` is "initiate_slew" or "stop_slew"; the target
// coordinates are present only for initiate_slew.
#[derive(Serialize)]
struct GotoRecord {
    #[serde(rename = "type")]
    kind: &'static str,
    time: String,
    action: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    ra_deg: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    dec_deg: Option<f64>,
}

/// Append-only observation log, shared as an `Arc` between the solve engine
/// (solve records) and the gRPC handler (GoTo records). Each write opens the
/// file in append mode, so there is no long-lived handle to coordinate; the
/// only shared mutable state is the solve-record throttle timestamp.
pub struct ObservationLog {
    config: ObsLogConfig,
    last_solve_write: Mutex<Option<Instant>>,
}

impl ObservationLog {
    /// Builds the log from the environment (see `parse_config`). Always returns
    /// an instance; when disabled, every logging call is a no-op.
    pub fn from_env() -> ObservationLog {
        let config = parse_config(|k| std::env::var(k).ok());
        if config.enabled {
            info!(
                "Observation log enabled: path={:?} throttle={:.1}s",
                config.path,
                config.throttle.as_secs_f64()
            );
        }
        ObservationLog {
            config,
            last_solve_write: Mutex::new(None),
        }
    }

    /// A disabled log whose logging calls are all no-ops. Useful for tests and
    /// harnesses that construct a SolveEngine but should not write to disk.
    pub fn disabled() -> ObservationLog {
        ObservationLog {
            config: ObsLogConfig {
                path: PathBuf::from(DEFAULT_PATH),
                throttle: Duration::from_secs_f64(DEFAULT_THROTTLE_SECS),
                enabled: false,
            },
            last_solve_write: Mutex::new(None),
        }
    }

    /// Records the just-stored plate solution. Skips IMU interpolations and
    /// applies the throttle; solved and unsolved real frames are both recorded
    /// (distinguished by `solved`). Never fails: I/O errors are logged.
    pub fn log_solve(&self, ps: &PlateSolution) {
        if !self.config.enabled {
            return;
        }
        let proto = ps.plate_solution.as_ref();
        // Skip IMU interpolations: they are derived pointing, not fresh frames.
        if proto.map(|p| p.solution_from_imu).unwrap_or(false) {
            return;
        }

        let now = Instant::now();
        {
            let mut last = self.last_solve_write.lock().unwrap();
            if !should_write(*last, now, self.config.throttle) {
                return;
            }
            *last = Some(now);
        }

        let captured = &ps.detect_result.captured_image;
        let solved = proto.is_some();
        let (ra_deg, dec_deg, roll_deg, fov_deg, rmse_arcsec, num_matches, solve_ms) =
            match proto {
                Some(p) => {
                    let (ra, dec) = match p.image_sky_coord.as_ref() {
                        Some(c) => (Some(c.ra), Some(c.dec)),
                        None => (None, None),
                    };
                    let solve_ms = p
                        .solve_time
                        .clone()
                        .and_then(|d| Duration::try_from(d).ok())
                        .map(|d| d.as_secs_f64() * 1000.0);
                    (
                        ra,
                        dec,
                        Some(p.roll),
                        Some(p.fov),
                        Some(p.rmse),
                        Some(p.num_matches),
                        solve_ms,
                    )
                }
                None => (None, None, None, None, None, None, None),
            };
        let pipeline_ms = ps
            .solve_finish_time
            .and_then(|finish| finish.duration_since(captured.readout_time).ok())
            .map(|d| d.as_secs_f64() * 1000.0);

        let record = SolveRecord {
            kind: "solve",
            time: iso_local(captured.readout_time),
            solution_id: ps.solution_id,
            frame_id: ps.detect_result.frame_id,
            solved,
            ra_deg,
            dec_deg,
            roll_deg,
            fov_deg,
            rmse_arcsec,
            num_matches,
            solve_ms,
            pipeline_ms,
        };
        self.write_record(&record);
    }

    /// Records a GoTo request. `action` is "initiate_slew" or "stop_slew";
    /// `coord` carries the (J2000) target for initiate_slew, None otherwise.
    pub fn log_goto(&self, action: &str, coord: Option<&CelestialCoord>) {
        if !self.config.enabled {
            return;
        }
        let record = GotoRecord {
            kind: "goto",
            time: iso_local(SystemTime::now()),
            action: action.to_string(),
            ra_deg: coord.map(|c| c.ra),
            dec_deg: coord.map(|c| c.dec),
        };
        self.write_record(&record);
    }

    fn write_record<T: Serialize>(&self, record: &T) {
        let line = match serde_json::to_string(record) {
            Ok(s) => s,
            Err(e) => {
                warn!("Observation log: serialize failed: {:?}", e);
                return;
            }
        };
        if let Err(e) = self.append_line(&line) {
            warn!("Observation log: write to {:?} failed: {:?}", self.config.path, e);
        }
    }

    fn append_line(&self, line: &str) -> std::io::Result<()> {
        if let Some(parent) = self.config.path.parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent)?;
            }
        }
        let mut file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.config.path)?;
        writeln!(file, "{}", line)
    }
}

// Throttle decision for solve records, factored out for testing. True when
// enough time has elapsed since the last write (or there was none).
fn should_write(last: Option<Instant>, now: Instant, throttle: Duration) -> bool {
    match last {
        Some(last) => now.duration_since(last) >= throttle,
        None => true,
    }
}

// Formats a SystemTime as ISO-8601 in the local timezone, matching the
// benchmark corpus's `%Y-%m-%dT%H:%M:%S%z` convention (whole seconds).
fn iso_local(t: SystemTime) -> String {
    let secs = t.duration_since(UNIX_EPOCH).unwrap_or(Duration::ZERO).as_secs();
    let datetime_local: DateTime<Local> =
        DateTime::from(DateTime::from_timestamp(secs as i64, 0).unwrap());
    datetime_local.format("%Y-%m-%dT%H:%M:%S%z").to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn lookup(pairs: &[(&str, &str)]) -> impl Fn(&str) -> Option<String> {
        let map: HashMap<String, String> = pairs
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect();
        move |k: &str| map.get(k).cloned()
    }

    #[test]
    fn config_defaults_when_unset() {
        let c = parse_config(lookup(&[]));
        assert!(c.enabled);
        assert_eq!(c.path, PathBuf::from(DEFAULT_PATH));
        assert_eq!(c.throttle, Duration::from_secs_f64(1.0));
    }

    #[test]
    fn config_disable_flag() {
        assert!(!parse_config(lookup(&[("CEDAR_OBSLOG", "0")])).enabled);
        assert!(!parse_config(lookup(&[("CEDAR_OBSLOG", "false")])).enabled);
        assert!(!parse_config(lookup(&[("CEDAR_OBSLOG", "no")])).enabled);
        // Any other value keeps it enabled.
        assert!(parse_config(lookup(&[("CEDAR_OBSLOG", "1")])).enabled);
    }

    #[test]
    fn config_path_and_interval_overrides() {
        let c = parse_config(lookup(&[
            ("CEDAR_OBSLOG_PATH", "/tmp/obs.jsonl"),
            ("CEDAR_OBSLOG_INTERVAL_SECS", "0.5"),
        ]));
        assert_eq!(c.path, PathBuf::from("/tmp/obs.jsonl"));
        assert_eq!(c.throttle, Duration::from_secs_f64(0.5));
    }

    #[test]
    fn throttle_logic() {
        let now = Instant::now();
        let throttle = Duration::from_secs(1);
        // No previous write: always allowed.
        assert!(should_write(None, now, throttle));
        // Just wrote: not yet.
        assert!(!should_write(Some(now), now, throttle));
        // Enough time elapsed.
        let earlier = now.checked_sub(Duration::from_secs(2)).unwrap();
        assert!(should_write(Some(earlier), now, throttle));
        // A zero throttle always allows.
        assert!(should_write(Some(now), now, Duration::ZERO));
    }

    #[test]
    fn solve_record_serializes_solved() {
        let rec = SolveRecord {
            kind: "solve",
            time: "2026-07-13T21:04:11-0700".to_string(),
            solution_id: 842,
            frame_id: 842,
            solved: true,
            ra_deg: Some(83.822),
            dec_deg: Some(-5.391),
            roll_deg: Some(112.4),
            fov_deg: Some(10.2),
            rmse_arcsec: Some(1.8),
            num_matches: Some(22),
            solve_ms: Some(31.2),
            pipeline_ms: Some(58.0),
        };
        let json = serde_json::to_string(&rec).unwrap();
        assert!(json.starts_with(r#"{"type":"solve","time":"2026-07-13T21:04:11-0700""#));
        assert!(json.contains(r#""solved":true"#));
        assert!(json.contains(r#""ra_deg":83.822"#));
        assert!(json.contains(r#""num_matches":22"#));
    }

    #[test]
    fn solve_record_omits_none_sky_fields_when_unsolved() {
        let rec = SolveRecord {
            kind: "solve",
            time: "2026-07-13T21:04:11-0700".to_string(),
            solution_id: 1,
            frame_id: 1,
            solved: false,
            ra_deg: None,
            dec_deg: None,
            roll_deg: None,
            fov_deg: None,
            rmse_arcsec: None,
            num_matches: None,
            solve_ms: None,
            pipeline_ms: None,
        };
        let json = serde_json::to_string(&rec).unwrap();
        assert!(json.contains(r#""solved":false"#));
        assert!(!json.contains("ra_deg"));
        assert!(!json.contains("num_matches"));
        assert!(!json.contains("solve_ms"));
    }

    #[test]
    fn goto_record_serializes() {
        let rec = GotoRecord {
            kind: "goto",
            time: "2026-07-13T21:05:02-0700".to_string(),
            action: "initiate_slew".to_string(),
            ra_deg: Some(101.287),
            dec_deg: Some(-16.716),
        };
        let json = serde_json::to_string(&rec).unwrap();
        assert!(json.contains(r#""type":"goto""#));
        assert!(json.contains(r#""action":"initiate_slew""#));
        assert!(json.contains(r#""ra_deg":101.287"#));

        // stop_slew carries no coordinates.
        let stop = GotoRecord {
            kind: "goto",
            time: "2026-07-13T21:06:00-0700".to_string(),
            action: "stop_slew".to_string(),
            ra_deg: None,
            dec_deg: None,
        };
        let json = serde_json::to_string(&stop).unwrap();
        assert!(json.contains(r#""action":"stop_slew""#));
        assert!(!json.contains("ra_deg"));
    }
}
