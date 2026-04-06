import os
import argparse
import torch
from modules import ModelUtils

def main(app_config:dict):
    match app_config['app_num']:
        case 1:
            """
            App 1.
            Load and save pretrained LLM and its tokenizer or processor
                Qwen3.5
            """
            model_config={
                "torch_dtype":torch.bfloat16,
                "device_map":"auto"
            }
            ModelUtils.save_pretrained_llm_from_HF(
                HF_path=app_config["HF_path"],
                model_name=app_config["model_name"],
                model_type=app_config["model_type"],
                **model_config
            )
            match app_config["model_type"]:
                case "llm":
                    ModelUtils.save_tokenizer_from_HF(
                        HF_path=app_config["HF_path"],
                        model_name=app_config["model_name"]
                    )
                case "vlm":
                    ModelUtils.save_processor_from_HF(
                        HF_path=app_config["HF_path"],
                        model_name=app_config["model_name"]
                    )

        case 2:
            """
            App 2.
            OCR 
            """
            model=ModelUtils.load_local_llm(model_name=app_config["model_name"],model_type=app_config["model_type"])
            processor=ModelUtils.load_local_processor(model_name=app_config["model_name"])

            image_paths=[
                os.path.join("dataset", "images", "2087686040757.png"), # 원재료
                os.path.join("dataset", "images", "8800279679073.png"), # 영양성분
                os.path.join("dataset", "images", "0000088002798.png"), # 식품사진
            ]

            messages=[
                {
                    "role":"system",
                    "content":[
                        {
                            "type":"text",
                            "text":f"You are an OCR system. Extract all visible text from the given image exactly as it appears."
                        }
                    ] 
                },
                {
                    "role":"user",
                    "content":[
                        {
                            "type":"image",
                            "url":image_paths[1]
                        },
                        {
                            "type":"text",
                            "text":f"Extract raw text."
                        }
                    ] 
                }
            ]

            inputs=processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            ).to(model.device)

            with torch.no_grad():
                outputs=model.generate(
                    **inputs,
                    max_new_tokens=512,
                    do_sample=False,   # OCR은 deterministic이 좋음
                )
            
            # 입력 길이 이후만 잘라서 디코딩
            generated_ids=outputs[0][inputs["input_ids"].shape[-1]:]
            result=processor.decode(
                generated_ids,
                skip_special_tokens=True
            )
            print(result)

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--app_num",type=int,default=1)
    parser.add_argument("--HF_path",type=str,default="Qwen/Qwen3.5-9B") 
    parser.add_argument("--model_name",type=str,default="Qwen3.5-9B") 
    parser.add_argument("--model_type",type=str,default="llm") 
    args=parser.parse_args()
    app_config={
        # app 관련
        "app_num":args.app_num,
        "HF_path":args.HF_path,
        "model_name":args.model_name,
        "model_type":args.model_type
    }
    main(app_config=app_config)