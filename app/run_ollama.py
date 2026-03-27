import os
import argparse
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage,HumanMessage
from modules import get_system_prompt,get_human_prompt

def main(app_config:dict):
    match app_config['app_num']:
        case 1:
            """
            App-1.
            Run VLM
                Qwen
            """
            vlm=ChatOllama(
                model=app_config['model_name'],
                base_url=f"http://localhost:{app_config['port']}"
            )

            # image_path=os.path.join("dataset","images","8801045516554.png") # 영양성분 이미지
            # image_path=os.path.join("dataset","images","2087686023125.png") # 원재료 이미지
            image_path=os.path.join("dataset","images","1500000017972.png") # 그 외 이미지

            system_msg=SystemMessage(content=get_system_prompt())
            human_msg=HumanMessage(
                content=[
                    {
                        "type":"text",
                        "text":get_human_prompt()
                    },
                    {
                        "type":"image_url",
                        "image_url":image_path
                    }

                ]
            )

            msg=[
                system_msg,
                human_msg
            ]
            response=vlm.invoke(msg)
            print(response.content)

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--app_num",type=int,default=1)
    parser.add_argument("--port",type=int,default=11434)
    parser.add_argument("--model_name",type=str,default="qwen3.5:0.8b") # Qwen3.5-0.8B-Instruct
    args=parser.parse_args()
    app_config={
        # app 관련
        "app_num":args.app_num,
        "port":args.port,
        "model_name":args.model_name
    }
    main(app_config=app_config)