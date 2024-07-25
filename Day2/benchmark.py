import json
import requests

if __name__ == "__main__":
    url = 'http://127.0.0.1:8000/generate'
    user_prompt = "I'm new to coding. If you could only recommend one programming language to start with, what would it be and why?"

    req_body = {
        "prompt": f"<start_of_turn>user\n${user_prompt}<end_of_turn>\n",
        "temperature": 0.90,
        "top_p": 1.0,
        "max_tokens": 1024
    }

    NUM_ITER = 50
    PRINT_EVERY = 5
    elapsed_times = []
    num_tokens = []

    for idx in range(NUM_ITER):
        res = requests.post(url, json=req_body)
        res = json.loads(res.text)
        elapsed_times.append(res["benchmark"]["total_elapsed_time"])
        num_tokens.append(res["benchmark"]["total_tokens_generated"])

        if (idx + 1) % PRINT_EVERY == 0:
            print(f"Done: {idx + 1} / {NUM_ITER}")

    tot_num_tokens = sum(num_tokens)
    tot_elapsed_time = sum(elapsed_times)
    avg_tp = tot_num_tokens / tot_elapsed_time
    print("\n===== Result =====")
    print(f"Iterations: {NUM_ITER}")
    print(f"Total Elapsed Time for Generation: {tot_elapsed_time:.2f} seconds")
    print(f"Total Generated Tokens: {tot_num_tokens}")
    print(f"Average Throughput: {avg_tp:.2f} tokens/sec")

