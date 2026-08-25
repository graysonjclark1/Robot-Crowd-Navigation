import json
import math
from pathlib import Path
from typing import Dict, List, Tuple, Any


INPUT_JSON = "mall_crowd_annotations.json"
OUTPUT_JSON = "mall_crowd_annotations_interpolated.json"
DT_SEC = 0.05  # variable spacing in seconds


Point = Tuple[int, int]


def lerp(a: float, b: float, alpha: float) -> float:
    return a + alpha * (b - a)


def lerp_point(p0: List[int], p1: List[int], alpha: float) -> List[int]:
    return [
        int(round(lerp(float(p0[0]), float(p1[0]), alpha))),
        int(round(lerp(float(p0[1]), float(p1[1]), alpha))),
    ]


def person_map(frame: Dict[str, Any]) -> Dict[int, Dict[str, Any]]:
    return {int(p["person_id"]): p for p in frame.get("persons", [])}


def build_time_grid(t_start: float, t_end: float, dt_sec: float) -> List[float]:
    n = int(math.floor((t_end - t_start) / dt_sec + 1e-9))
    grid = [round(t_start + i * dt_sec, 10) for i in range(n + 1)]
    if grid[-1] < t_end - 1e-9:
        grid.append(round(t_end, 10))
    return grid


def interpolate_annotation_file(
    input_json: str,
    output_json: str,
    dt_sec: float = 0.05,
) -> None:
    if dt_sec <= 0:
        raise ValueError("dt_sec must be positive")

    with open(input_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    original_frames = data.get("frames", [])
    annotated_frames = [fr for fr in original_frames if not fr.get("skipped", False)]

    if len(annotated_frames) < 2:
        raise ValueError("Need at least two non-skipped annotated frames to interpolate.")

    annotated_frames.sort(key=lambda fr: float(fr["timestamp_sec"]))

    t_start = float(annotated_frames[0]["timestamp_sec"])
    t_end = float(annotated_frames[-1]["timestamp_sec"])
    new_times = build_time_grid(t_start, t_end, dt_sec)

    # Keep exact annotated timestamps available for direct copy when aligned.
    exact_time_to_frame: Dict[float, Dict[str, Any]] = {
        round(float(fr["timestamp_sec"]), 10): fr for fr in annotated_frames
    }

    # Precompute consecutive annotated intervals.
    intervals = []
    for i in range(len(annotated_frames) - 1):
        f0 = annotated_frames[i]
        f1 = annotated_frames[i + 1]
        t0 = float(f0["timestamp_sec"])
        t1 = float(f1["timestamp_sec"])
        if t1 <= t0:
            continue
        intervals.append((t0, t1, f0, f1))

    new_frames: List[Dict[str, Any]] = []
    interval_idx = 0

    for out_idx, t in enumerate(new_times):
        t_rounded = round(t, 10)

        # If this output time matches an original annotated frame, copy it exactly.
        if t_rounded in exact_time_to_frame:
            src = exact_time_to_frame[t_rounded]
            persons_copy = []
            for p in src.get("persons", []):
                persons_copy.append({
                    "person_id": int(p["person_id"]),
                    "left_shoulder": list(p["left_shoulder"]),
                    "right_shoulder": list(p["right_shoulder"]),
                    "center": list(p["center"]),
                })

            new_frames.append({
                "frame_index": out_idx,
                "timestamp_sec": t_rounded,
                "timestamp_ms": round(t_rounded * 1000.0, 10),
                "skipped": False,
                "persons": persons_copy,
            })
            continue

        while interval_idx < len(intervals) and t > intervals[interval_idx][1] + 1e-9:
            interval_idx += 1

        if interval_idx >= len(intervals):
            break

        t0, t1, f0, f1 = intervals[interval_idx]
        if not (t0 - 1e-9 <= t <= t1 + 1e-9):
            continue

        alpha = 0.0 if abs(t1 - t0) < 1e-12 else (t - t0) / (t1 - t0)

        p0_map = person_map(f0)
        p1_map = person_map(f1)

        # Only interpolate persons present in both endpoint annotated frames.
        shared_ids = sorted(set(p0_map.keys()) & set(p1_map.keys()))

        persons_interp = []
        for pid in shared_ids:
            p0 = p0_map[pid]
            p1 = p1_map[pid]

            persons_interp.append({
                "person_id": pid,
                "left_shoulder": lerp_point(p0["left_shoulder"], p1["left_shoulder"], alpha),
                "right_shoulder": lerp_point(p0["right_shoulder"], p1["right_shoulder"], alpha),
                "center": lerp_point(p0["center"], p1["center"], alpha),
            })

        new_frames.append({
            "frame_index": out_idx,
            "timestamp_sec": t_rounded,
            "timestamp_ms": round(t_rounded * 1000.0, 10),
            "skipped": False,
            "persons": persons_interp,
        })

    output_data = {
        "video_path": data.get("video_path", ""),
        "fps": data.get("fps", None),
        "frame_step": data.get("frame_step", 1),
        "interpolated_dt_sec": dt_sec,
        "source_file": str(Path(input_json).name),
        "frames": new_frames,
    }

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)

    print(f"Wrote {len(new_frames)} frames to {output_json}")


if __name__ == "__main__":
    interpolate_annotation_file(INPUT_JSON, OUTPUT_JSON, DT_SEC)