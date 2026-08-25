import json

ANNOTATION_JSON = "ec_shoulder_annotations.json"


def find_humans_who_left(json_path: str) -> None:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    frames = sorted(data["frames"], key=lambda fr: fr["frame_index"])

    previous_annotated_ids = set()
    previous_annotated_frame = None

    for frame in frames:
        if frame.get("skipped", False):
            continue

        current_ids = {person["person_id"] for person in frame.get("persons", [])}

        if previous_annotated_frame is not None:
            left_ids = previous_annotated_ids - current_ids

            for person_id in sorted(left_ids):
                print(
                    f"Human {person_id} left the scene by frame {frame['frame_index']} "
                    f"(last seen in annotated frame {previous_annotated_frame})"
                )

        previous_annotated_ids = current_ids
        previous_annotated_frame = frame["frame_index"]


if __name__ == "__main__":
    find_humans_who_left(ANNOTATION_JSON)