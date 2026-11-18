# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Small JSON client for a shared simulator KV broker."""

import json
from urllib.error import HTTPError, URLError
from urllib.request import ProxyHandler, Request, build_opener


class BrokerError(RuntimeError):
    pass


class BrokerClient:
    def __init__(self, url, owner, timeout_s=2):
        self.url = url.rstrip("/")
        self.owner = owner
        self.timeout = timeout_s
        self.ack = 0
        self.broker_id: str | None = None
        self._http = build_opener(ProxyHandler({}))

    def call(self, method, **values):
        data = json.dumps(values).encode()
        request = Request(
            self.url + "/" + method,
            data=data,
            headers={"Content-Type": "application/json"},
        )
        try:
            with self._http.open(request, timeout=self.timeout) as response:
                return json.load(response)
        except HTTPError as error:
            try:
                message = json.load(error)["error"]
            except (ValueError, KeyError):
                message = str(error)
            raise BrokerError(message) from error
        except (URLError, OSError, ValueError) as error:
            raise BrokerError(f"KV broker request failed: {error}") from error

    def poll(self, wait_s=0):
        result = self.call(
            "poll",
            owner=self.owner,
            ack=self.ack,
            broker_id=self.broker_id,
            wait_s=wait_s,
        )
        if result["broker_id"] != self.broker_id:
            self.broker_id = result["broker_id"]
            self.ack = 0
        events = result["events"]
        if events:
            self.ack = max(self.ack, max(event["seq"] for event in events))
        return events
