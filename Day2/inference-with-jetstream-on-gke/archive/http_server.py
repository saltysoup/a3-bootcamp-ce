# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""HTTP Server to interact with JetStream Server."""

import asyncio
import concurrent.futures
import json
import time
import logging
from typing import Optional

import fastapi
import grpc
from jetstream.core.proto import jetstream_pb2
from jetstream.core.proto import jetstream_pb2_grpc
import pydantic


class GenerateRequest(pydantic.BaseModel):
    server: Optional[str] = "127.0.0.1"
    port: Optional[str] = "9000"
    session_cache: Optional[str] = ""
    prompt: Optional[str] = "This is an example prompt"
    priority: Optional[int] = 0
    max_tokens: Optional[int] = 100


app = fastapi.FastAPI()
executor = concurrent.futures.ThreadPoolExecutor(max_workers=1000)

@app.get("/")
def root():
    """Root path for Jetstream HTTP Server."""
    response = {"message": "HTTP Server for JetStream"}
    response = fastapi.Response(
        content=json.dumps(response, indent=4), media_type="application/json"
    )
    return response


@app.post("/generate", status_code=200)
async def generate(request: GenerateRequest):
    """Generate a prompt."""
    try:
        request = jetstream_pb2.DecodeRequest(
            session_cache=request.session_cache,
            text_content=jetstream_pb2.DecodeRequest.TextContent(text=request.prompt),
            priority=request.priority,
            max_tokens=request.max_tokens,
        )

        start_time = time.perf_counter()
        future = executor.submit(generate_prompt, request)
        result = await future.result()

        time_elapsed = time.perf_counter() - start_time
        response = {
        "prediction": result["output"],
        "benchmark": {
                "total_elapsed_time": time_elapsed,
                "total_tokens_generated": result["tot_num_tokens"],
                "throughput": result["tot_num_tokens"] / time_elapsed,
            }
        }

        response = fastapi.Response(
            content=json.dumps(response, indent=4), media_type="application/json"
        )
        return response
    except Exception as e:
        logging.exception("Exception in generate")
        raise fastapi.HTTPException(status_code=500, detail=str(e))


async def generate_prompt(
    request: jetstream_pb2.DecodeRequest,
):
    """Generate a prompt."""
    tokens = []

    options = [("grpc.keepalive_timeout_ms", 10000)]
    async with grpc.aio.insecure_channel("127.0.0.1:9000", options=options) as channel:
        stub = jetstream_pb2_grpc.OrchestratorStub(channel)
        response = stub.Decode(request)
        output = ""
        async for r in response:
            tokens.append(r.stream_content.samples[0].text)
            output += str(r.stream_content.samples[0].text)

        ret = {
            "output": output,
            "tot_num_tokens": len(tokens)
        }
        return ret
    