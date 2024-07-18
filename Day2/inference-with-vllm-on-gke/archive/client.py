import requests

if __name__ == "__main__":
    url = 'http://127.0.0.1:8000/generate'
    
    user_prompt = "I'm new to coding. If you could only recommend one programming language to start with, what would it be and why?"

    req_body = {
        "prompt": "<start_of_turn>user\n${user_prompt}<end_of_turn>\n",
        "temperature": 0.90,
        "top_p": 1.0,
        "max_tokens": 128
    }

    x = requests.post(url, json=req_body)

    print(x.text)