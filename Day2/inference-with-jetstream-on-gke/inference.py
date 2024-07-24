import json
import requests

if __name__ == "__main__":
    url = 'http://127.0.0.1:8000/generate'
    user_prompt = "I'm new to coding. If you could only recommend one programming language to start with, what would it be and why?"

    req_body = {
        "prompt": f"<start_of_turn>user\n${user_prompt}<end_of_turn>\n",
        "max_tokens": 1024
    }

    res = requests.post(url, json=req_body)
    res = json.loads(res.text)
    res = json.dumps(res, indent=4)

    print(res)