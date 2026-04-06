from modules import ModelUtils

model=ModelUtils.load_local_causal_llm(model_name=f"Qwen3.5-9B")
tokenizer=ModelUtils.load_local_tokenizer(model_name=f"Qwen3.5-9B")

for label in ["0","1","2"]:
    ids=tokenizer.encode(label,add_special_tokens=False)
    print(label,ids)