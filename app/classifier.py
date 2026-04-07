import os
import argparse
import torch
from modules import ModelUtils,DataUtils
from modules import get_system_prompt_for_OCR,get_human_prompt_for_OCR,get_system_prompt_for_classifier,get_human_prompt_for_classifier

def chunk_list(lst,chunk_size):
    """리스트를 chunk_size 단위로 나누기"""
    for i in range(0,len(lst),chunk_size):
        yield lst[i:i+chunk_size]

def get_batch_messages(image_paths: list):
    batch_messages=[]
    for url in image_paths:
        batch_messages.append([
            {
                "role":"system",
                "content":[
                    {"type":"text","text":get_system_prompt_for_classifier()}
                ]
            },
            {
                "role":"user",
                "content":[
                    {"type":"image","url":url},
                    {"type":"text","text":get_human_prompt_for_classifier()}
                ]
            }
        ])
    return batch_messages

def run_batch_OCR(model,processor,image_paths,batch_size=3):
    results=[]

    # batch_size 단위로 나누기
    for batch_paths in chunk_list(image_paths,batch_size):
        batch_messages=get_batch_messages(image_paths=batch_paths)
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
                do_sample=False,
            )

        input_len=inputs["input_ids"].shape[1]

        # batch 내부 index 기준으로 처리
        for i in range(len(batch_paths)):
            generated_ids=outputs[i][input_len:]
            text=processor.decode(
                generated_ids,
                skip_special_tokens=True
            ).strip()
            results.append(text)
    return results


def classifier(**kwargs):
    """
    """
    # 이미지 가져오기
    food_list=DataUtils.get_food_list()
    image_paths=DataUtils.get_food_images(
        food_id=food_list[kwargs["food_num"]]
    )

    model=ModelUtils.load_local_llm(
        model_name=kwargs["model_name"],
        model_type="vlm"
    )

    processor=ModelUtils.load_local_processor(
        model_name=kwargs["model_name"]
    )

    results=run_batch_OCR(
        model=model,
        processor=processor,
        image_paths=image_paths,
        batch_size=3  
    )

    for result in results:
        print(result, end="\n\n")

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--app_num",type=int,default=1)
    parser.add_argument("--model_name",type=str,default="Qwen3.5-9B") 
    parser.add_argument("--food_num",type=int,default=0)
    args=parser.parse_args()
    app_config={
        # app 관련
        "app_num":args.app_num,
        "model_name":args.model_name,
        "food_num":args.food_num
    }
    classifier(**app_config)