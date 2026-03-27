import os
import json

# 1. 영양성분
dir_path=os.path.join("dataset","images","nutrition")
file_names=[
    os.path.splitext(f)[0]
    for f in os.listdir(dir_path)
    if os.path.isfile(os.path.join(dir_path,f))
]

nutrition_label_json_list=[]
for file_name in file_names:
    label_json={
        "image":file_name,
        "label":0
    }
    nutrition_label_json_list.append(label_json)

# 2. 원재료
dir_path=os.path.join("dataset","images","ingredient")
file_names=[
    os.path.splitext(f)[0]
    for f in os.listdir(dir_path)
    if os.path.isfile(os.path.join(dir_path,f))
]

ingredient_label_json_list=[]
for file_name in file_names:
    label_json={
        "image":file_name,
        "label":1
    }
    ingredient_label_json_list.append(label_json)

# 3. 그 외
dir_path=os.path.join("dataset","images","others")
file_names=[
    os.path.splitext(f)[0]
    for f in os.listdir(dir_path)
    if os.path.isfile(os.path.join(dir_path,f))
]

others_label_json_list=[]
for file_name in file_names:
    label_json={
        "image":file_name,
        "label":2
    }
    others_label_json_list.append(label_json)

label_json_list=nutrition_label_json_list+ingredient_label_json_list+others_label_json_list
with open("label.json","w",encoding="utf-8") as f:
    json.dump(label_json_list,f,ensure_ascii=False,indent=2)


