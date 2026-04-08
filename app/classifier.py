import argparse
from langchain_ollama import ChatOllama
from modules import ModelUtils,DataUtils

def classifier(**kwargs):
    # 이미지 가져오기
    food_list=DataUtils.get_food_list()
    image_paths=DataUtils.get_food_images(
        food_id=food_list[kwargs["food_num"]]
    )

    model=ChatOllama(
        model=app_config['model_name'],
        base_url=f"http://localhost:{app_config['port']}"
    )

    messages=ModelUtils.get_batch_classifier_messages(image_paths=image_paths,isBase64=True)
    responses=model.batch(
        messages,
        config={"max_concurrency":1}
    )
    for response in responses:
        print(response.content,end="\n\n")

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--model_name",type=str,default="Qwen3.5:35B") 
    parser.add_argument("--port",type=int,default=11434)
    parser.add_argument("--food_num",type=int,default=0)
    args=parser.parse_args()
    app_config={
        # app 관련
        "model_name":args.model_name,
        "port":args.port,
        "food_num":args.food_num
    }
    classifier(**app_config)