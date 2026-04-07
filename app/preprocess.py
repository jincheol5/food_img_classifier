import os
import shutil
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict

# 경로 설정
base_dir = Path(os.path.join("..", "data", "chronolab"))
pre_raw_images_dir = base_dir / "pre_raw_images"
raw_images_dir = base_dir / "raw_images"

raw_images_dir.mkdir(parents=True, exist_ok=True)

# food_id별 카운터
counters = defaultdict(int)

# 모든 png 파일 수집
all_files = []
for food_group_dir in tqdm(sorted(pre_raw_images_dir.iterdir()), desc="Scanning groups"):
    if not food_group_dir.is_dir():
        continue

    for food_id_dir in sorted(food_group_dir.iterdir()):
        if not food_id_dir.is_dir():
            continue

        food_id = food_id_dir.name

        for png_file in food_id_dir.rglob("*.png"):
            if png_file.is_file():
                all_files.append((food_id, png_file))

# 이동 (tqdm 적용)
for food_id, png_file in tqdm(all_files, desc="Moving images"):
    target_food_dir = raw_images_dir / food_id
    target_food_dir.mkdir(parents=True, exist_ok=True)

    counters[food_id] += 1
    new_name = f"{food_id}_{counters[food_id]}.png"
    target_path = target_food_dir / new_name

    shutil.move(str(png_file), str(target_path))

print("모든 png 파일 이동 완료")