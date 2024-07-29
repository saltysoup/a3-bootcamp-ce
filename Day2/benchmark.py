import asyncio
import aiohttp
import json


async def send_req(session: aiohttp.ClientSession):
    url = 'http://127.0.0.1:8000/generate'
    user_prompt = "I'm new to coding. If you could only recommend one programming language to start with, what would it be and why?"

    req_body = {
        "prompt": f"<start_of_turn>user\n${user_prompt}<end_of_turn>\n",
        "temperature": 0.90,
        "top_p": 1.0,
        "max_tokens": 1024
    }

    async with session.post(url, json=req_body) as response:
        return await response.text()
             
async def main(n_reqs: int):
    async with aiohttp.ClientSession() as session:
        resp_list = await asyncio.gather(
            *[send_req(session) for _ in range(n_reqs)]
        )

    return resp_list


if __name__ == "__main__":
    REQ_CNT = 30

    resp_list = asyncio.run(main(REQ_CNT))

    num_tokens_list = []
    elapsed_time_list = []

    for resp in resp_list:
        resp = json.loads(resp)
        elapsed_time_list.append(resp["benchmark"]["total_elapsed_time"])
        num_tokens_list.append(resp["benchmark"]["total_tokens_generated"])

    tot_elapsed_time = sum(elapsed_time_list)
    tot_num_tokens = sum(num_tokens_list)
    avg_tp = tot_num_tokens / tot_elapsed_time

    print("\n===== Result =====")
    print(f"Request Count: {REQ_CNT}")
    print(f"Total Elapsed Time for Generation: {tot_elapsed_time:.2f} seconds")
    print(f"Total Generated Tokens: {tot_num_tokens}")
    print(f"Average Throughput: {avg_tp:.2f} tokens/sec")
