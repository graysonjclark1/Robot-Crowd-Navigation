import json

JSON_FILE = "mall_crowd_annotations.json"


with open(JSON_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

max_count = 0
max_frames = []

for frame in data.get("frames", []):
    people_count = len(frame.get("persons", []))

    if people_count > max_count:
        max_count = people_count
        max_frames = [frame.get("frame_index")]
    elif people_count == max_count:
        max_frames.append(frame.get("frame_index"))

print(f"Highest number of people: {max_count}")
print(f"Frame(s): {max_frames}")