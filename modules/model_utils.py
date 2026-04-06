import os
from transformers import AutoTokenizer,AutoModelForCausalLM

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
    def load_local_causal_llm(model_name:str,**kwargs):
        """
        model_config
            torch_dtype: Auto or torch.bfloat16 (GPU)
            device_map: "Auto", "cuda:0" or "cpu"
        """
        model_path=os.path.join(ModelUtils.dir_path,"pretrained",model_name)
        model=AutoModelForCausalLM.from_pretrained(
            model_path, 
            torch_dtype=kwargs["torch_dtype"]
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