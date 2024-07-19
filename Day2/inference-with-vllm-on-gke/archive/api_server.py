"""
NOTE: This API server is used only for demonstrating usage of AsyncEngine
and simple performance benchmarks. It is not intended for production use.
For production use, we recommend using our OpenAI compatible server.
We are also not going to accept PRs modifying this file, please
change `vllm/entrypoints/openai/api_server.py` instead.
"""

import argparse
import copy
import json
from http import HTTPStatus
import ssl
import time
from typing import AsyncGenerator, List

import uvicorn
from fastapi import FastAPI, Request, status, HTTPException
from fastapi.responses import JSONResponse, Response, StreamingResponse

from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.sampling_params import SamplingParams
from vllm.usage.usage_lib import UsageContext
from vllm.utils import random_uuid
from vllm.entrypoints.openai.api_server import (
    init_openai_api_server,
    create_chat_completion,
)
from vllm.engine.async_llm_engine import AsyncEngineDeadError
from vllm.entrypoints.openai.cli_args import LoRAParserAction
from vllm.entrypoints.openai.protocol import ChatCompletionRequest
from vllm.lora import lora_disk_lru_cache
from vllm.entrypoints import vertex_gke_integration
from vllm.logger import init_logger
from vllm.model_executor.model_loader.utils import get_model_architecture


logger = init_logger(__name__)
TIMEOUT_KEEP_ALIVE = 5  # seconds.
app = FastAPI()
engine = None
lora_modules_manager = None

def format_output(prompt: str, output: str):
    output = output.strip("\n")
    return f"prompt:\n{prompt.strip()}\noutput:\n{output}"


@app.get("/ping")
async def ping() -> Response:
    return Response(status_code=200)


@app.get("/health")
async def health() -> Response:
    """Health check."""
    return Response(status_code=200)

@app.post("/v1/chat/completions", status_code=400)
async def chat_completions(request: Request) -> Response:
    """Generate chat completions for the request.

    This should follow the OpenAI request format."""
    request_dict = await request.json()
    chat_completion_request = ChatCompletionRequest(**request_dict)
    return await create_chat_completion(chat_completion_request, request)


@app.post("/generate", status_code=400)
async def generate(request: Request) -> Response:
    """Generate completion for the request.

    The request should be a JSON object with the following fields:
    - prompt: the prompt to use for the generation.
    - stream: whether to stream the results or not.
    - other fields: the sampling parameters (See `SamplingParams` for details).
    """
    assert engine is not None
    request_dict = await request.json()
    is_on_vertex = "instances" in request_dict
    if is_on_vertex:
        request_dicts = request_dict["instances"]
    else:
        request_dicts = [request_dict]
    ret = []
    multi_request = len(request_dicts) > 1
    try:
        for request_dict in request_dicts:
            is_chat_completion = request_dict.pop("@requestFormat", "") == "chatCompletions"
            # OpenAI style request
            if is_chat_completion:
                chat_completion_request = ChatCompletionRequest(**request_dict)
                response = await create_chat_completion(chat_completion_request, request)
                ret.append(json.loads(response.body))
            else:
                prompt = request_dict.pop("prompt")
                stream = request_dict.pop("stream", False)
                if multi_request and stream:
                    raise HTTPException(
                        status_code=HTTPStatus.BAD_REQUEST,
                        detail="stream=True can only be used with a single prompt.",
                    )

                dynamic_lora_path = request_dict.pop("dynamic-lora", None)
                raw_response = request_dict.pop("raw_response", False)
                sampling_params = SamplingParams(**request_dict)
                request_id = random_uuid()

                lora_request = None
                if lora_modules_manager is not None and dynamic_lora_path is not None:
                    lora_request = lora_modules_manager.get_dynamic_lora_request(dynamic_lora_path)

                results_generator = engine.generate(prompt, sampling_params, request_id, lora_request=lora_request)

                # Streaming case
                async def stream_results() -> AsyncGenerator[bytes, None]:
                    prior_request_output = None
                    async for request_output in results_generator:
                        prompt = request_output.prompt
                        text_outputs = []
                        for i, output in enumerate(request_output.outputs):
                            if prior_request_output is not None:
                                prior_output = prior_request_output.outputs[i]
                                text_output = output.text[len(prior_output.text) :]
                            else:
                                text_output = output.text
                            text_outputs.append(text_output)
                        ret = {"predictions": text_outputs}
                        if raw_response:
                            output_token_counts = []
                            for i, output in enumerate(request_output.outputs):
                                if prior_request_output is not None:
                                    prior_output = prior_request_output.outputs[i]
                                    output_token_count = len(output.token_ids) - len(
                                        prior_output.token_ids
                                    )
                                else:
                                    output_token_count = len(output.token_ids)
                                output_token_counts.append(output_token_count)
                            cumulative_logprobs = [
                                output.cumulative_logprob
                                for output in request_output.outputs
                            ]
                            ret.update(
                                {
                                    "output_token_counts": output_token_counts,
                                    "cumulative_logprobs": cumulative_logprobs,
                                }
                            )
                        prior_request_output = copy.deepcopy(request_output)
                        yield (json.dumps(ret) + "\0").encode("utf-8")

                if stream:
                    return StreamingResponse(stream_results())

                # Non-streaming case
                start_time = time.perf_counter()
                final_output = None
                async for request_output in results_generator:
                    if await request.is_disconnected():
                        # Abort the request if the client disconnects.
                        await engine.abort(request_id)
                        return Response(status_code=499)
                    final_output = request_output

                # measure overall time for token generation
                time_elapsed = time.perf_counter() - start_time

                assert final_output is not None
                if raw_response:
                    text_outputs = [output.text for output in final_output.outputs]
                    output_token_counts = [
                        len(output.token_ids) for output in final_output.outputs
                    ]
                    cumulative_logprobs = [
                        output.cumulative_logprob for output in final_output.outputs
                    ]
                    ret.append({
                        "predictions": text_outputs,
                        "output_token_counts": output_token_counts,
                        "cumulative_logprobs": cumulative_logprobs,
                    })
                else:
                    prompt = final_output.prompt
                    text_outputs = [
                        format_output(prompt, output.text)
                        for output in final_output.outputs
                    ]

                    # count number of tokens
                    output_token_counts = [
                        len(output.token_ids) for output in final_output.outputs
                    ]
                    total_tokens_generated = sum(output_token_counts)

                    ret.append({
                        "prediction": text_outputs[0],
                        "benchmark": {
                            "total_elapsed_time": time_elapsed,
                            "total_tokens_generated": total_tokens_generated,
                            "throughput": total_tokens_generated / time_elapsed,
                        }
                    })
    except ValueError as e:
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST,
            detail=str(e),
        )
    except AsyncEngineDeadError as e:
        raise HTTPException(
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
            detail=str(e),
        )
    if len(ret) == 1 and not is_chat_completion:
        return JSONResponse(ret[0])
    else:
        return JSONResponse({"predictions": ret})


if __name__ == "__main__":
    logger.info("Starting API server...")
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default=None)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--ssl-keyfile", type=str, default=None)
    parser.add_argument("--ssl-certfile", type=str, default=None)
    parser.add_argument(
        "--ssl-ca-certs", type=str, default=None, help="The CA certificates file"
    )
    parser.add_argument(
        "--ssl-cert-reqs",
        type=int,
        default=int(ssl.CERT_NONE),
        help="Whether client certificate is required (see stdlib ssl module's)",
    )
    parser.add_argument(
        "--root-path",
        type=str,
        default=None,
        help="FastAPI root_path when app is behind a path based routing proxy",
    )
    # OpenAI server arguments.
    parser.add_argument(
        "--lora-modules",
        type=str,
        default=None,
        nargs="+",
        action=LoRAParserAction,
        help="LoRA module configurations in the format name=path. "
        "Multiple modules can be specified.",
    )
    parser.add_argument(
        "--chat-template",
        type=str,
        default=None,
        help="The file path to the chat template, "
        "or the template in single-line form "
        "for the specified model",
    )
    parser.add_argument(
        "--response-role",
        type=str,
        default="assistant",
        help="The role name to return if " "`request.add_generation_prompt=true`.",
    )
    parser.add_argument("--log-level", type=str, default="debug")
    parser = AsyncEngineArgs.add_cli_args(parser)
    args = parser.parse_args()
    engine_args = AsyncEngineArgs.from_cli_args(args)
    engine = AsyncLLMEngine.from_engine_args(
        engine_args, usage_context=UsageContext.API_SERVER)

    if engine.engine.scheduler.lora_enabled:
        model_cls, arch = get_model_architecture(engine.engine.model_config)
        supported_lora_modules = model_cls.supported_lora_modules
        packed_modules_mapping = model_cls.packed_modules_mapping
        expected_lora_modules: List[str] = []
        for module in supported_lora_modules:
            if module in packed_modules_mapping:
                expected_lora_modules.extend(
                    packed_modules_mapping[module])
            else:
                expected_lora_modules.append(module)
        lora_modules_manager = lora_disk_lru_cache.LoRADiskLRUCache(
            max_lora_rank=engine.engine.lora_config.max_lora_rank,
            expected_lora_modules=expected_lora_modules,
        )

    logger.info("Initializing OpenAI API server...")
    init_openai_api_server(args, engine)

    app.root_path = args.root_path

    vertex_gke_integration.maybe_snapshot_on_gke()

    uvicorn.run(app,
                host=args.host,
                port=args.port,
                log_level=args.log_level,
                timeout_keep_alive=TIMEOUT_KEEP_ALIVE,
                ssl_keyfile=args.ssl_keyfile,
                ssl_certfile=args.ssl_certfile,
                ssl_ca_certs=args.ssl_ca_certs,
                ssl_cert_reqs=args.ssl_cert_reqs)