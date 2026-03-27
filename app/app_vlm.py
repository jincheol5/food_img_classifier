import argparse
from langchain_ollama import ChatOllama

def main(app_config:dict):
    match app_config['app_num']:
        case 1:
            """
            model pull
            """
            vlm=ChatOllama(
                model=app_config['model_name'],
                base_url=f"http://localhost:{app_config['port']}"
            )



if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--app_num",type=int,default=1)
    parser.add_argument("--model_name",type=str,default="qwen2.5:1.5b") # Qwen2.5-1.5B-Instruct
    args=parser.parse_args()
    app_config={
        # app 관련
        'app_num':args.app_num,
        'model_name':args.model_name
    }
    main(app_config=app_config)