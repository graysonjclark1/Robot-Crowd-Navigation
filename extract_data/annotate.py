import cv2
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

VIDEO_PATH = "videos\\edited_people_walking.mp4"

# Rename for each video
OUTPUT_JSON = "epw_shoulder_annotations.json"
SESSION_JSON = "epw_shoulder_annotation_session.json"
FRAME_STEP = 1

WINDOW_NAME = "Shoulder Annotation"

SHOULDER_DOT_RADIUS = 4
CENTER_DOT_RADIUS = 7
HIGHLIGHT_RADIUS = 11

COLOR_SHOULDER = (0, 255, 0)
COLOR_CENTER = (0, 0, 255)
COLOR_HIGHLIGHT = (0, 255, 255)
COLOR_DONE_PREV = (255, 0, 255)
COLOR_LINE = (255, 0, 0)
COLOR_TEXT = (255, 255, 255)
COLOR_HISTORY = (180, 180, 0)


@dataclass
class PersonAnnotation:
    person_id: int
    left_shoulder: Tuple[int, int]
    right_shoulder: Tuple[int, int]
    center: Tuple[int, int]


@dataclass
class FrameAnnotation:
    frame_index: int
    timestamp_sec: float
    timestamp_ms: float
    persons: List[PersonAnnotation]
    skipped: bool = False


def midpoint(p1: Tuple[int, int], p2: Tuple[int, int]) -> Tuple[int, int]:
    return ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2)


def clip_point(pt: Tuple[int, int], width: int, height: int) -> Tuple[int, int]:
    x = max(0, min(width - 1, pt[0]))
    y = max(0, min(height - 1, pt[1]))
    return x, y


class ShoulderAnnotator:
    def __init__(self, video_path: str, output_json: str, session_json: str, frame_step: int = 1):
        self.video_path = video_path
        self.output_json = output_json
        self.session_json = session_json
        self.frame_step = max(1, frame_step)

        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")

        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 0.0

        self.frame_cache: Dict[int, any] = {}
        self.annotations: Dict[int, FrameAnnotation] = {}
        self.next_person_id = 1

        self.current_frame_index: int = 0
        self.current_frame = None
        self.display_frame = None

        self.current_clicks: List[Tuple[int, int]] = []
        self.current_persons: Dict[int, PersonAnnotation] = {}

        self.prev_person_queue: List[PersonAnnotation] = []
        self.prev_queue_index: int = 0
        self.processed_prev_ids: set[int] = set()

        self.mode: str = "update_prev"
        self.action_stack: List[dict] = []
        self.frame_dirty: bool = False

        # Most recent frame BEFORE current_frame_index that actually had person annotations
        self.reference_person_frame_index: Optional[int] = None

        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(WINDOW_NAME, self.on_mouse)

        self.load_session_if_exists()

    def read_frame(self, frame_index: int):
        if frame_index in self.frame_cache:
            return self.frame_cache[frame_index].copy()

        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = self.cap.read()
        if not ok:
            return None

        self.frame_cache[frame_index] = frame.copy()
        return frame

    def frame_timestamp_sec(self, frame_index: int) -> float:
        if self.fps > 0:
            return frame_index / self.fps
        return 0.0

    def find_previous_saved_frame(self, frame_index: int) -> Optional[int]:
        prev_candidates = [idx for idx in self.annotations.keys() if idx < frame_index]
        if not prev_candidates:
            return None
        return max(prev_candidates)

    def find_previous_person_frame(self, frame_index: int) -> Optional[int]:
        candidates = []
        for idx, ann in self.annotations.items():
            if idx < frame_index and len(ann.persons) > 0:
                candidates.append(idx)
        if not candidates:
            return None
        return max(candidates)

    def load_frame(self, frame_index: int) -> bool:
        frame = self.read_frame(frame_index)
        if frame is None:
            return False

        self.current_frame_index = frame_index
        self.current_frame = frame
        self.display_frame = frame.copy()

        self.current_clicks = []
        self.current_persons = {}
        self.prev_person_queue = []
        self.prev_queue_index = 0
        self.processed_prev_ids = set()
        self.action_stack = []
        self.frame_dirty = False

        self.reference_person_frame_index = self.find_previous_person_frame(frame_index)

        if self.reference_person_frame_index is not None:
            prev_persons = sorted(
                self.annotations[self.reference_person_frame_index].persons,
                key=lambda p: p.person_id
            )
            self.prev_person_queue = prev_persons
            self.mode = "update_prev" if len(prev_persons) > 0 else "add_new"
        else:
            self.mode = "add_new"

        self.redraw()
        return True

    def get_current_prev_person(self) -> Optional[PersonAnnotation]:
        if self.mode != "update_prev":
            return None
        if self.prev_queue_index >= len(self.prev_person_queue):
            return None
        return self.prev_person_queue[self.prev_queue_index]

    def advance_prev_queue(self):
        self.prev_queue_index += 1
        self.current_clicks = []
        if self.prev_queue_index >= len(self.prev_person_queue):
            self.mode = "add_new"
        self.redraw()

    def redraw(self):
        self.display_frame = self.current_frame.copy()

        # Draw the last known person positions from the reference frame.
        # This is what persists across skipped frames.
        for prev_person in self.prev_person_queue:
            cx, cy = prev_person.center
            cv2.circle(self.display_frame, (cx, cy), 5, COLOR_HISTORY, -1)
            cv2.putText(
                self.display_frame,
                f"ID {prev_person.person_id}",
                (cx + 6, cy + 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                COLOR_HISTORY,
                1,
                cv2.LINE_AA
            )

        # Draw finalized current-frame people
        for pid in sorted(self.current_persons.keys()):
            p = self.current_persons[pid]
            lx, ly = p.left_shoulder
            rx, ry = p.right_shoulder
            cx, cy = p.center

            cv2.circle(self.display_frame, (lx, ly), SHOULDER_DOT_RADIUS, COLOR_SHOULDER, -1)
            cv2.circle(self.display_frame, (rx, ry), SHOULDER_DOT_RADIUS, COLOR_SHOULDER, -1)
            cv2.line(self.display_frame, (lx, ly), (rx, ry), COLOR_LINE, 1)
            cv2.circle(self.display_frame, (cx, cy), CENTER_DOT_RADIUS, COLOR_CENTER, -1)
            cv2.putText(
                self.display_frame,
                f"ID {pid}",
                (cx + 8, cy + 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                COLOR_CENTER,
                1,
                cv2.LINE_AA
            )

        # Mark already-processed previous people
        for prev_person in self.prev_person_queue:
            if prev_person.person_id in self.processed_prev_ids:
                cx, cy = prev_person.center
                cv2.circle(self.display_frame, (cx, cy), HIGHLIGHT_RADIUS - 3, COLOR_DONE_PREV, 2)

        # Highlight the current previous person to update
        current_prev = self.get_current_prev_person()
        if current_prev is not None:
            cx, cy = current_prev.center
            cv2.circle(self.display_frame, (cx, cy), HIGHLIGHT_RADIUS, COLOR_HIGHLIGHT, 2)
            cv2.putText(
                self.display_frame,
                f"Update ID {current_prev.person_id}",
                (cx + 10, cy - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                COLOR_HIGHLIGHT,
                2,
                cv2.LINE_AA
            )

        # Draw temporary current clicks
        for pt in self.current_clicks:
            cv2.circle(self.display_frame, pt, SHOULDER_DOT_RADIUS, COLOR_SHOULDER, -1)

        ts_sec = self.frame_timestamp_sec(self.current_frame_index)
        hud_lines = [
            f"Frame: {self.current_frame_index + 1}/{self.total_frames}",
            f"Timestamp: {ts_sec:.3f} sec",
            f"Mode: {self.mode}",
            f"Started frame: {'yes' if self.frame_dirty else 'no'}",
        ]

        if self.reference_person_frame_index is not None:
            hud_lines.append(f"Reference person frame: {self.reference_person_frame_index + 1}")
        else:
            hud_lines.append("Reference person frame: none")

        if not self.frame_dirty:
            hud_lines.append("Press n now to skip this frame entirely")

        if self.mode == "update_prev":
            current_prev = self.get_current_prev_person()
            if current_prev is not None:
                hud_lines.append(f"Current previous ID: {current_prev.person_id}")
                hud_lines.append("Click 2 shoulders for this person, or press x if they left")
        else:
            hud_lines.append("Add mode: click 2 shoulders for each new person")
            hud_lines.append("Press n when done with this frame")

        hud_lines.append("u: undo | b: back | s: save | q: quit")

        y = 25
        for line in hud_lines:
            cv2.putText(
                self.display_frame,
                line,
                (15, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                COLOR_TEXT,
                2,
                cv2.LINE_AA
            )
            y += 24

    def save_current_frame_annotation(self, skipped: bool = False):
        ts_sec = self.frame_timestamp_sec(self.current_frame_index)
        self.annotations[self.current_frame_index] = FrameAnnotation(
            frame_index=self.current_frame_index,
            timestamp_sec=ts_sec,
            timestamp_ms=ts_sec * 1000.0,
            persons=[self.current_persons[k] for k in sorted(self.current_persons.keys())],
            skipped=skipped,
        )

    def can_skip_current_frame(self) -> bool:
        return (
            not self.frame_dirty
            and len(self.current_clicks) == 0
            and len(self.current_persons) == 0
            and len(self.action_stack) == 0
            and len(self.processed_prev_ids) == 0
        )

    def go_to_next_frame(self) -> bool:
        next_index = self.current_frame_index + self.frame_step
        if next_index >= self.total_frames:
            print("Reached end of video.")
            return False
        return self.load_frame(next_index)

    def skip_current_frame(self) -> bool:
        self.save_current_frame_annotation(skipped=True)
        self.save_session()
        return self.go_to_next_frame()

    def finish_current_frame(self) -> bool:
        self.save_current_frame_annotation(skipped=False)
        self.save_session()
        return self.go_to_next_frame()

    def update_existing_person_from_clicks(self):
        current_prev = self.get_current_prev_person()
        if current_prev is None or len(self.current_clicks) != 2:
            return

        p1, p2 = self.current_clicks
        center = midpoint(p1, p2)

        updated = PersonAnnotation(
            person_id=current_prev.person_id,
            left_shoulder=p1,
            right_shoulder=p2,
            center=center,
        )

        self.current_persons[updated.person_id] = updated
        self.processed_prev_ids.add(updated.person_id)
        self.action_stack.append({
            "type": "update_existing",
            "person_id": updated.person_id,
            "prev_queue_index_before": self.prev_queue_index,
        })

        self.current_clicks = []
        self.frame_dirty = True
        self.advance_prev_queue()

    def add_new_person_from_clicks(self):
        if len(self.current_clicks) != 2:
            return

        p1, p2 = self.current_clicks
        center = midpoint(p1, p2)

        new_person = PersonAnnotation(
            person_id=self.next_person_id,
            left_shoulder=p1,
            right_shoulder=p2,
            center=center,
        )
        self.next_person_id += 1

        self.current_persons[new_person.person_id] = new_person
        self.action_stack.append({
            "type": "add_new",
            "person_id": new_person.person_id,
        })

        self.current_clicks = []
        self.frame_dirty = True
        self.redraw()

    def mark_current_prev_person_gone(self):
        current_prev = self.get_current_prev_person()
        if current_prev is None:
            return

        self.processed_prev_ids.add(current_prev.person_id)
        self.action_stack.append({
            "type": "mark_gone",
            "person_id": current_prev.person_id,
            "prev_queue_index_before": self.prev_queue_index,
        })

        self.current_clicks = []
        self.frame_dirty = True
        self.advance_prev_queue()

    def undo_last_action(self):
        if self.current_clicks:
            self.current_clicks.pop()
            self.redraw()
            return

        if not self.action_stack:
            return

        action = self.action_stack.pop()
        typ = action["type"]

        if typ == "add_new":
            pid = action["person_id"]
            if pid in self.current_persons:
                self.current_persons.pop(pid)
            if pid == self.next_person_id - 1:
                self.next_person_id -= 1

        elif typ == "update_existing":
            pid = action["person_id"]
            if pid in self.current_persons:
                self.current_persons.pop(pid)
            self.processed_prev_ids.discard(pid)
            self.prev_queue_index = action["prev_queue_index_before"]
            self.mode = "update_prev"

        elif typ == "mark_gone":
            pid = action["person_id"]
            self.processed_prev_ids.discard(pid)
            self.prev_queue_index = action["prev_queue_index_before"]
            self.mode = "update_prev"

        self.current_clicks = []
        self.frame_dirty = (
            len(self.action_stack) > 0
            or len(self.current_persons) > 0
            or len(self.processed_prev_ids) > 0
        )
        self.redraw()

    def go_back(self) -> bool:
        prev_index = self.current_frame_index - self.frame_step
        if prev_index < 0:
            print("Already at first frame.")
            return True

        ok = self.load_frame(prev_index)
        if not ok:
            return False

        if prev_index in self.annotations:
            stored = self.annotations[prev_index]
            self.current_persons = {p.person_id: p for p in stored.persons}
            self.prev_person_queue = []
            self.prev_queue_index = 0
            self.processed_prev_ids = set()
            self.current_clicks = []
            self.action_stack = []
            self.mode = "add_new"
            self.frame_dirty = len(self.current_persons) > 0
            self.redraw()

        return True

    def on_mouse(self, event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return

        pt = clip_point((x, y), self.width, self.height)
        self.current_clicks.append(pt)

        if len(self.current_clicks) == 2:
            if self.mode == "update_prev":
                self.update_existing_person_from_clicks()
            else:
                self.add_new_person_from_clicks()
        else:
            self.redraw()

    def export_annotations(self):
        data = {
            "video_path": os.path.abspath(self.video_path),
            "fps": self.fps,
            "frame_step": self.frame_step,
            "frames": [],
        }

        for frame_index in sorted(self.annotations.keys()):
            frame_ann = self.annotations[frame_index]
            data["frames"].append({
                "frame_index": frame_ann.frame_index,
                "timestamp_sec": frame_ann.timestamp_sec,
                "timestamp_ms": frame_ann.timestamp_ms,
                "skipped": frame_ann.skipped,
                "persons": [
                    {
                        "person_id": p.person_id,
                        "left_shoulder": list(p.left_shoulder),
                        "right_shoulder": list(p.right_shoulder),
                        "center": list(p.center),
                    }
                    for p in frame_ann.persons
                ],
            })

        with open(self.output_json, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

        print(f"Saved annotations to: {self.output_json}")

    def save_session(self):
        data = {
            "video_path": os.path.abspath(self.video_path),
            "output_json": os.path.abspath(self.output_json),
            "frame_step": self.frame_step,
            "fps": self.fps,
            "next_person_id": self.next_person_id,
            "current_frame_index": self.current_frame_index,
            "annotations": [],
        }

        for frame_index in sorted(self.annotations.keys()):
            frame_ann = self.annotations[frame_index]
            data["annotations"].append({
                "frame_index": frame_ann.frame_index,
                "timestamp_sec": frame_ann.timestamp_sec,
                "timestamp_ms": frame_ann.timestamp_ms,
                "skipped": frame_ann.skipped,
                "persons": [
                    {
                        "person_id": p.person_id,
                        "left_shoulder": list(p.left_shoulder),
                        "right_shoulder": list(p.right_shoulder),
                        "center": list(p.center),
                    }
                    for p in frame_ann.persons
                ],
            })

        with open(self.session_json, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

        print(f"Session saved to: {self.session_json}")

    def load_session_if_exists(self):
        if not os.path.exists(self.session_json):
            return

        with open(self.session_json, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.next_person_id = data.get("next_person_id", 1)
        self.current_frame_index = data.get("current_frame_index", 0)

        loaded_annotations = {}
        for entry in data.get("annotations", []):
            persons = []
            for p in entry.get("persons", []):
                persons.append(
                    PersonAnnotation(
                        person_id=p["person_id"],
                        left_shoulder=tuple(p["left_shoulder"]),
                        right_shoulder=tuple(p["right_shoulder"]),
                        center=tuple(p["center"]),
                    )
                )

            loaded_annotations[entry["frame_index"]] = FrameAnnotation(
                frame_index=entry["frame_index"],
                timestamp_sec=entry.get("timestamp_sec", 0.0),
                timestamp_ms=entry.get("timestamp_ms", 0.0),
                persons=persons,
                skipped=entry.get("skipped", False),
            )

        self.annotations = loaded_annotations
        print(f"Loaded existing session from: {self.session_json}")

    def run(self):
        if self.annotations:
            last_done = max(self.annotations.keys())
            start_frame = min(last_done + self.frame_step, self.total_frames - 1)
        else:
            start_frame = self.current_frame_index

        if not self.load_frame(start_frame):
            raise RuntimeError("Could not load starting frame.")

        while True:
            cv2.imshow(WINDOW_NAME, self.display_frame)
            key = cv2.waitKey(20) & 0xFF

            if key == ord("q"):
                if self.frame_dirty:
                    self.save_current_frame_annotation(skipped=False)
                self.save_session()
                self.export_annotations()
                break

            elif key == ord("s"):
                if self.frame_dirty:
                    self.save_current_frame_annotation(skipped=False)
                self.save_session()
                self.export_annotations()

            elif key == ord("u"):
                self.undo_last_action()

            elif key == ord("x"):
                if self.mode == "update_prev":
                    self.mark_current_prev_person_gone()

            elif key == ord("n"):
                if self.can_skip_current_frame():
                    ok = self.skip_current_frame()
                    if not ok:
                        self.save_session()
                        self.export_annotations()
                        break
                else:
                    if self.mode == "add_new":
                        ok = self.finish_current_frame()
                        if not ok:
                            self.save_session()
                            self.export_annotations()
                            break

            elif key == ord("b"):
                ok = self.go_back()
                if not ok:
                    self.save_session()
                    self.export_annotations()
                    break

        self.cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    annotator = ShoulderAnnotator(
        video_path=VIDEO_PATH,
        output_json=OUTPUT_JSON,
        session_json=SESSION_JSON,
        frame_step=FRAME_STEP,
    )
    annotator.run()