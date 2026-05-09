import pickle
from collections import Counter
from pathlib import Path

pkl = Path("data/main_mt/MTGS/road_block-331220_4690660_331190_4690710/video_scene_dict.pkl")
data = pickle.load(open(pkl, "rb"))

vehicle_ids = Counter()
travel_counts = Counter()

for video_token, v in data.items():
    travel_id = int(video_token.split("-")[-1])
    travel_counts[travel_id] += 1
    for f in v["frame_infos"]:
        for name, token in zip(f["gt_names"], f["track_tokens"]):
            if name == "vehicle":
                vehicle_ids[token] += 1

print("travel_ids:", sorted(travel_counts.items())[:20], "total:", len(travel_counts))
print("vehicles total:", len(vehicle_ids))
print("top vehicles:")
for token, count in vehicle_ids.most_common(20):
    print(token, count)
