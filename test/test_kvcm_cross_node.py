"""Validate mixed-size KV transfers on two isolated, running serving nodes.

This test sends requests, invalidates L1, and evicts its own objects. Use only
dedicated debug nodes sharing an isolated KVCM instance and tracker namespace.
Enable the worker's invalidate_kv_cache callback (SCR_NEXT_CONTAINER=1,
SCR_ENABLE=0, DS_LLM_ENABLE_SCR_CROSS_DEPLOYMENT_RESTORE=0).
Run from the serving image with network access to both nodes.
"""

import argparse
import json
import random
import time
import uuid
from urllib.request import Request, urlopen


def api(host, path, body=None, port=9113):
    request = Request(
        f"http://{host}:{port}/{path}",
        data=None if body is None else json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urlopen(request, timeout=60) as response:
        return json.load(response)


def objects(host):
    return {
        item["key"]: item
        for item in api(host, "objects/list", {"tier": "vineyard", "limit": 10000})[
            "objects"
        ]
    }


def settled(host, keys):
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        current = objects(host)
        if keys <= current.keys() and all(not current[key]["leased"] for key in keys):
            return current
        time.sleep(0.2)
    raise AssertionError(f"Objects not settled on {host}: {keys}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hosts", nargs=2, required=True)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--input-tokens", type=int, default=8192)
    parser.add_argument(
        "--expected-peer-hosts",
        nargs=2,
        help="Expected advertised IPv4s, in --hosts order",
    )
    args = parser.parse_args()
    if args.rounds < 1 or args.input_tokens < 1:
        parser.error("rounds and input-tokens must be positive")
    try:
        from dashserving.client.llm_client import LLMClient
    except ImportError:
        from dashserving.client._rust import LLMClient
    from dashserving.config import DashClientConfig

    clients = {
        host: LLMClient(
            host + ":8001",
            capability={"type": "generate"},
            config=DashClientConfig(request_timeout_ms=120000),
        )
        for host in args.hosts
    }
    run = uuid.uuid4().hex[:12]

    def generate(host, prompt, phase):
        rid = f"kvcm-regression-{run}-{phase}"
        finished, count = False, 0
        for response in clients[host].generate(
            request_id=rid,
            prompt={"prompt_token_ids": prompt},
            params={"max_tokens": 1, "ignore_eos": True},
        ):
            assert not response.is_error_response(), response.error_msg
            if response.token_ids:
                count += len(response.token_ids[0])
            finished |= response.is_finished()
        assert finished and count == 1, (rid, finished, count)
        print("REQUEST", rid, host, flush=True)

    for index, host in enumerate(args.hosts):
        peer = api(host, "peer_info")
        assert peer["kvcm_status"]["cache_transfer_ready"]
        if args.expected_peer_hosts:
            assert (
                peer["peer_address"].rsplit(":", 1)[0]
                == args.expected_peer_hosts[index]
            ), peer["peer_address"]
        print("PEER", host, peer["peer_address"], flush=True)
    for iteration in range(args.rounds):
        source, receiver = args.hosts[:: 1 if iteration % 2 == 0 else -1]
        rng = random.Random(run + str(iteration))
        prompt = [rng.randint(100, 10000) for _ in range(args.input_tokens)]
        before = objects(source)
        generate(source, prompt, f"{iteration}-populate")
        deadline = time.monotonic() + 30
        while not (keys := objects(source).keys() - before.keys()):
            assert time.monotonic() < deadline, "No KV objects stored"
            time.sleep(0.2)
        original = settled(source, keys)
        sizes = {key: original[key]["size"] for key in keys}
        assert len(set(sizes.values())) > 1, "Regression requires mixed-size KV groups"
        hot_before = api(receiver, "stats")["hit_stats"]["remote_vineyard_hits"]
        generate(receiver, prompt, f"{iteration}-hot-cross")
        fetched = settled(receiver, keys)
        assert {key: fetched[key]["size"] for key in keys} == sizes, (
            "P2P changed object sizes"
        )
        hot_delta = (
            api(receiver, "stats")["hit_stats"]["remote_vineyard_hits"] - hot_before
        )
        assert hot_delta == len(keys), (hot_delta, len(keys))

        # Evict the P2P-fetched copies first: this exposed the F3 size rejection.
        for host in (receiver, source):
            result = api(host, "objects/evict", {"keys": sorted(keys)})
            print("EVICT", host, json.dumps(result), flush=True)
            assert not (keys & objects(host).keys()), (
                "Eviction did not remove test objects"
            )
            reset = api(host, "invalidate_kv_cache", {}, port=8100)
            assert reset.get("status") == "kv_cache_invalidated", reset
        time.sleep(2)
        stats_before = api(receiver, "stats")["hit_stats"]
        generate(receiver, prompt, f"{iteration}-cold-reload")
        reloaded = settled(receiver, keys)
        assert {key: reloaded[key]["size"] for key in keys} == sizes, (
            "Cold reload changed object sizes"
        )
        stats_after = api(receiver, "stats")["hit_stats"]
        assert (
            stats_after["remote_vineyard_hits"] == stats_before["remote_vineyard_hits"]
        ), "Cold reload unexpectedly used a hot peer"
        assert stats_after["total_hits"] - stats_before["total_hits"] == len(keys), (
            "Cold reload did not hit every object"
        )
        assert (
            stats_after["local_vineyard_hits"] == stats_before["local_vineyard_hits"]
        ), "Cold reload unexpectedly used local KV"
        print(
            "ROUND_PASS",
            iteration,
            "objects",
            len(keys),
            "sizes",
            sorted(set(sizes.values())),
            flush=True,
        )
    print("COMPLETE", run, "rounds", args.rounds, flush=True)


if __name__ == "__main__":
    main()
