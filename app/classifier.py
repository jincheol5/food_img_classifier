import os
import argparse
import torch
from modules import ModelUtils,DataUtils
from modules import get_system_prompt_for_OCR,get_human_prompt_for_OCR

def get_batch_messages(image_paths:list):
    batch_messages=[]
    for url in image_paths:
        batch_messages.append([
            {
                "role":"system",
                "content":[
                    {"type":"text","text":get_system_prompt_for_OCR()}
                ]
            },
            {
                "role":"user",
                "content":[
                    {"type":"image","url":url}, 
                    {"type":"text","text":get_human_prompt_for_OCR()}
                ]
            }
        ])
    return batch_messages

def run_batch_OCR(model,processor,image_paths):
    """
    """
    batch_messages=get_batch_messages(image_paths=image_paths)
    inputs=processor.apply_chat_template(
            batch_messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            padding=True,
    ).to(model.device)

    with torch.no_grad():
        outputs=model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False,   # OCR은 deterministic이 좋음
        )

    input_len=inputs["input_ids"].shape[1]  # 배치 공통 padded input 길이
    results=[]
    for i,image_path in enumerate(image_paths):
        generated_ids=outputs[i][input_len:]
        text=processor.decode(generated_ids,skip_special_tokens=True).strip()
        results.append({
            "image_name":os.path.splitext(os.path.basename(image_path))[0],
            "text": text,
        })
    return results

def classifier(**kwargs):
    """
    """
    ### get images
    food_list=DataUtils.get_food_list()
    
    idx=0
    while True:
        food_id=food_list[idx]
        image_paths=DataUtils.get_food_images(food_id=food_id)
        if not image_paths:
            idx+=1
            continue
        else:
            break

    model=ModelUtils.load_local_llm(
        model_name=kwargs["model_name"],
        model_type=f"vlm"
    )
    processor=ModelUtils.load_local_processor(
        model_name=kwargs["model_name"]
    )
    results=run_batch_OCR(
        model=model,
        processor=processor,
        image_paths=image_paths
    )

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--app_num",type=int,default=1)
    parser.add_argument("--model_name",type=str,default="Qwen3.5-9B") 
    args=parser.parse_args()
    app_config={
        # app 관련
        "app_num":args.app_num,
        "model_name":args.model_name
    }
    classifier(**app_config)