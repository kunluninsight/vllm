"""Stream OpenAI completions HTTP traffic to one DashLLM daemon."""

import argparse
import asyncio
import json
import logging
import time
import uuid

from aiohttp import web

logger = logging.getLogger(__name__)


def create_app(client, model, tokenizer=None):
    app = web.Application(client_max_size=32 * 1024**2)
    active = set()
    counts = {"completed": 0, "failed": 0, "peak_active": 0}

    async def models(request):
        return web.json_response(
            {"object": "list", "data": [{"id": model, "object": "model"}]}
        )

    async def health(request):
        return web.json_response(dict(status="ok", active=len(active), **counts))

    async def completions(request):
        try:
            body = await request.json()
            if not isinstance(body, dict):
                raise TypeError("request body must be a JSON object")
            ids = body.get("prompt")
            if (
                not isinstance(ids, list)
                or not ids
                or any(type(token) is not int or token < 0 for token in ids)
            ):
                raise ValueError(
                    "prompt must be a nonempty array of original token IDs"
                )
            output_len = body.get("max_tokens")
            if type(output_len) is not int or output_len <= 0:
                raise ValueError("max_tokens must be a positive integer")
            if body.get("stream") is not True or body.get("n", 1) != 1:
                raise ValueError("only stream=true and n=1 are supported")
        except (ValueError, TypeError) as exc:
            raise web.HTTPBadRequest(
                text=json.dumps({"error": str(exc)}), content_type="application/json"
            ) from exc
        rid = request.headers.get("x-request-id", uuid.uuid4().hex)
        internal_id = f"{rid}-{uuid.uuid4().hex[:12]}"
        active.add(internal_id)
        counts["peak_active"] = max(counts["peak_active"], len(active))
        started = time.monotonic()
        stream = web.StreamResponse(
            headers={
                "Content-Type": "text/event-stream",
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            }
        )
        sent = 0
        finished = False
        prepared = False
        decoded_ids = []
        decoded_text = ""
        params = {
            key: body[key]
            for key in (
                "temperature",
                "top_p",
                "top_k",
                "ignore_eos",
                "repetition_penalty",
                "presence_penalty",
                "frequency_penalty",
            )
            if key in body
        }
        params["max_tokens"] = output_len

        async def emit(payload):
            await stream.write(("data: " + json.dumps(payload) + "\n\n").encode())

        try:
            async for response in client.async_generate(
                request_id=internal_id,
                prompt={"prompt_token_ids": ids},
                params=params,
                capability={"type": "generate"},
            ):
                if response.is_error_response():
                    raise RuntimeError(response.error_msg)
                tokens = response.token_ids[0] if response.token_ids else []
                if tokens:
                    if not prepared:
                        await stream.prepare(request)
                        prepared = True
                    sent += len(tokens)
                    text = ""
                    if tokenizer is not None:
                        decoded_ids.extend(tokens)
                        full_text = tokenizer.decode(
                            decoded_ids,
                            skip_special_tokens=False,
                            clean_up_tokenization_spaces=False,
                        )
                        text = full_text.removeprefix(decoded_text)
                        decoded_text = full_text
                    await emit(
                        {
                            "id": rid,
                            "object": "text_completion",
                            "model": model,
                            "choices": [
                                {
                                    "index": 0,
                                    "text": text,
                                    "token_ids": list(tokens),
                                    "finish_reason": None,
                                }
                            ],
                        }
                    )
                finished |= response.is_finished()
            if not finished or not sent:
                raise RuntimeError(
                    f"incomplete DashLLM stream: finished={finished}, tokens={sent}"
                )
            # choices must stay empty: vLLM counts every choices-bearing SSE as a token event.
            await emit(
                {
                    "id": rid,
                    "choices": [],
                    "usage": {
                        "prompt_tokens": len(ids),
                        "completion_tokens": sent,
                        "total_tokens": len(ids) + sent,
                    },
                }
            )
            await stream.write(b"data: [DONE]\n\n")
            await stream.write_eof()
            counts["completed"] += 1
            return stream
        except (asyncio.CancelledError, ConnectionError):
            counts["failed"] += 1
            client.cancel_request(internal_id)
            raise
        except Exception as exc:
            counts["failed"] += 1
            client.cancel_request(internal_id)
            logger.exception("request %s failed after %d tokens", rid, sent)
            if not prepared:
                raise web.HTTPBadGateway(
                    text=json.dumps({"error": str(exc)}),
                    content_type="application/json",
                ) from exc
            # Abort partial streams; a clean EOF would make benchmark mark them successful.
            if request.transport is not None:
                request.transport.abort()
            return stream
        finally:
            active.discard(internal_id)
            logger.info(
                "request=%s input_tokens=%d output_tokens=%d elapsed=%.6f finished=%s",
                rid,
                len(ids),
                sent,
                time.monotonic() - started,
                finished,
            )

    app.router.add_get("/health", health)
    app.router.add_get("/v1/models", models)
    app.router.add_post("/v1/completions", completions)
    return app


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dashllm-address", default="127.0.0.1:8001")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18080)
    parser.add_argument("--model", default="dashllm")
    parser.add_argument(
        "--tokenizer",
        help="Optional local tokenizer for response text; input IDs are never re-tokenized",
    )
    parser.add_argument("--request-timeout-ms", type=int, default=3600000)
    args = parser.parse_args()
    from dashserving.client.llm_client import LLMClient
    from dashserving.config import DashClientConfig

    tokenizer = None
    if args.tokenizer:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer, trust_remote_code=True
        )
    client = LLMClient(
        args.dashllm_address,
        capability={"type": "generate"},
        config=DashClientConfig(request_timeout_ms=args.request_timeout_ms),
    )
    logging.basicConfig(level=logging.INFO)
    web.run_app(
        create_app(client, args.model, tokenizer),
        host=args.host,
        port=args.port,
        access_log=None,
    )


if __name__ == "__main__":
    main()
