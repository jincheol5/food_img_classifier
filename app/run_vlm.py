import os
import argparse
import torch
from modules import ModelUtils

def main(app_config:dict):
    match app_config['app_num']:
        case 1:
            """
            App 1.
            Load and save pretrained LLM and its tokenizer
                Qwen3.5
            """
            model_config={
                "torch_dtype":torch.bfloat16,
                "device_map":"auto"
            }
            ModelUtils.save_pretrained_causal_llm_from_HF(
                HF_path=app_config["HF_path"],
                model_name=app_config["model_name"],
                model_config=model_config
            )
            ModelUtils.save_tokenizer_from_HF(
                HF_path=app_config["HF_path"],
                model_name=app_config["model_name"]
            )

        case 2:
            """
            App 2.

            """
            model_config={
                "torch_dtype":torch.bfloat16,
                "device_map":"auto"
            }
            model=ModelUtils.load_local_causal_llm(
                model_name=app_config["model_name"],
                model_config=model_config
            )
            tokenizer=ModelUtils.load_local_tokenizer(
                model_name=app_config["model_name"],
                model_config=model_config
            )




if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--app_num",type=int,default=1)
    parser.add_argument("--HF_path",type=str,default="Qwen/Qwen3.5-9B") 
    parser.add_argument("--model_name",type=str,default="Qwen3.5-9B") 
    args=parser.parse_args()
    app_config={
        # app 관련
        "app_num":args.app_num,
        "HF_path":args.HF_path,
        "model_name":args.model_name
    }
    main(app_config=app_config)