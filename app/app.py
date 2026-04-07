import os
import torch
from modules import ModelUtils,RestrictTokensProcessor
from modules import get_system_prompt_for_classifier,get_human_prompt_for_classifier
from transformers import AutoProcessor

"""
    0: 영양성분  
    1: 원재료
    2: others
"""

image_paths=[
    os.path.join("dataset", "images", "2087686040757.png"), # 원재료
    os.path.join("dataset", "images", "8800279679073.png"), # 영양성분
    os.path.join("dataset", "images", "0000088002798.png"), # 식품사진
]

model=ModelUtils.load_local_llm(model_name=f"Qwen3.5-9B",model_type=f"vlm")
processor=AutoProcessor.from_pretrained("Qwen/Qwen3.5-9B")

"""
하나만 추론
"""
messages=[
    {
        "role":"system",
        "content":[
            {
                "type":"text",
                "text":get_system_prompt_for_classifier()
            }
        ] 
    },
    {
        "role":"user",
        "content":[
            {
                "type":"image",
                "url":image_paths[2]
            },
            {
                "type":"text",
                "text":get_human_prompt_for_classifier()
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
    logits=model(**inputs).logits[:,-1,:] # logits.shape=[batch_size,sequence_length,vocab_size], 마지막 예측 토큰에 대한 확률분포 가져오기

scores=logits[0,[15,16,17]]
pred=torch.argmax(scores).item()
print(pred)

"""
batch 추론
"""
# batch_messages=[]
# for url in image_paths:
#     batch_messages.append([
#         {
#             "role": "system",
#             "content": [
#                 {"type": "text", "text": get_system_prompt_for_classifier()}
#             ]
#         },
#         {
#             "role": "user",
#             "content": [
#                 {"type": "image", "url": url}, 
#                 {"type": "text", "text": get_human_prompt_for_classifier()}
#             ]
#         }
#     ])

# # 2. 한 번에 인코딩 (텍스트 + 이미지)
# inputs=processor.apply_chat_template(
#     batch_messages,
#     add_generation_prompt=True,
#     tokenize=True,
#     return_dict=True,
#     return_tensors="pt",
#     padding=True,
# ).to(model.device)

# # 배치 입력에서 각 샘플의 마지막 유효 토큰 위치의 logits만 정확히 뽑아서, 후보 토큰 3개 중 어떤 것이 가장 점수가 높은지 예측하는 코드
# model.eval()
# with torch.no_grad():
#     outputs=model(**inputs)
#     logits=outputs.logits # [B,L,V], B=batch size, L=각 입력 시퀀스 길이(padding 포함), V=vocabulary size

# last_indices=inputs["attention_mask"].sum(dim=1)-1
# batch_indices=torch.arange(logits.size(0),device=logits.device)
# last_logits=logits[batch_indices,last_indices,:]

# scores=last_logits[:,[15,16,17]]
# preds=torch.argmax(scores,dim=1)

# for path,pred in zip(image_paths,preds.tolist()):
#     print(f"{os.path.basename(path)} -> pred: {pred}")
