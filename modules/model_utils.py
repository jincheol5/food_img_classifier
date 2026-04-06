import os
import torch
from transformers import AutoTokenizer,AutoModelForCausalLM,AutoModelForImageTextToText
from transformers import LogitsProcessor

class RestrictTokensProcessor(LogitsProcessor):
    def __init__(self,allowed_token_ids):
        self.allowed_token_ids=allowed_token_ids

    def __call__(self,input_ids,scores):
        masked=torch.full_like(scores,float("-inf"))
        masked[:,self.allowed_token_ids]=scores[:,self.allowed_token_ids]
        return masked

class ModelUtils:
    """
    """
    dir_path=os.path.join("..","data","llm")

    @staticmethod
    def save_pretrained_causal_llm_from_HF(HF_path:str,model_name:str,**kwargs):
        """
        model_config
            torch_dtype: auto or torch.bfloat16 (GPU)
            device_map: "auto", weight를 읽는 순간 바로 GPU/CPU에 분산 배치
        """
        model_path=os.path.join(ModelUtils.dir_path,"pretrained",model_name)
        os.makedirs(model_path,exist_ok=True) # 해당 경로의 모든 폴더 없으면 생성

        model=AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=HF_path,
            torch_dtype=kwargs["torch_dtype"] ,
            device_map=kwargs["device_map"] 
        )
        model.save_pretrained(model_path)
        print(f"Save pretrained causal llm: {model_name} from {HF_path}!")

    @staticmethod
    def save_pretrained_vlm_from_HF(HF_path:str,model_name:str,**kwargs):
        """
        """
        model_path=os.path.join(ModelUtils.dir_path,"pretrained",model_name)
        os.makedirs(model_path,exist_ok=True) # 해당 경로의 모든 폴더 없으면 생성

    @staticmethod
    def save_tokenizer_from_HF(HF_path:str,model_name:str):
        """
        """
        tokenizer_path=os.path.join(ModelUtils.dir_path,"pretrained",model_name)
        os.makedirs(tokenizer_path,exist_ok=True) # 해당 경로의 모든 폴더 없으면 생성
        tokenizer=AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=HF_path
        )
        tokenizer.save_pretrained(tokenizer_path)
        print(f"Save tokenizer for pretrained causal llm: {model_name} from {HF_path}!")
    
    @staticmethod
    def load_local_causal_llm(model_name:str):
        """
        """
        model_path=os.path.join(ModelUtils.dir_path,"pretrained",model_name)
        model=AutoModelForCausalLM.from_pretrained(
            model_path
        )
        return model

    @staticmethod
    def load_local_tokenizer(model_name:str):
        """
        """
        model_path=os.path.join(ModelUtils.dir_path,"pretrained",model_name)
        tokenizer=AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=model_path
        )
        return tokenizer