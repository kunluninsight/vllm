# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in HTTP collection controls for the native GPU worker."""


class CollectorPlugin:
    name = "simulator_collector"
    required_tasks = ("generate",)

    def attach_router(self, app):
        from fastapi import APIRouter, Request

        router = APIRouter()

        @router.post("/start_profile")
        async def start(raw_request: Request):
            state = raw_request.app.state
            result = await state.collector_engine.collective_rpc("start_collection")
            if state.collector_timeline is not None:
                state.collector_timeline.start()
            return result

        @router.post("/stop_profile")
        async def stop(raw_request: Request):
            state = raw_request.app.state
            result = await state.collector_engine.collective_rpc("stop_collection")
            if state.collector_timeline is not None:
                state.collector_timeline.stop()
            return result

        app.include_router(router)

    async def init_state(self, engine_client, state, args):
        if args.worker_cls != "vllm_simulator.collector.worker.CollectorWorker":
            raise ValueError(
                "simulator_collector requires --worker-cls "
                "vllm_simulator.collector.worker.CollectorWorker"
            )
        state.collector_engine = engine_client

        state.collector_timeline = None
        config = engine_client.vllm_config
        options = config.additional_config.get("simulator", {}).get("collector", {})
        if options.get("request_timeline", False):
            import vllm.envs as envs
            from vllm_simulator.collector.timeline import RequestTimeline

            manager = engine_client.logger_manager
            if manager is None or config.parallel_config._api_process_count != 1:
                raise ValueError(
                    "request_timeline requires log stats and one API process"
                )
            if not envs.VLLM_DISABLE_REQUEST_ID_RANDOMIZATION:
                raise ValueError(
                    "Debug request_timeline requires "
                    "VLLM_DISABLE_REQUEST_ID_RANDOMIZATION=1 and unique request IDs"
                )
            timeline = RequestTimeline(config, manager.engine_indexes)
            timeline.attach(engine_client.output_processor)
            manager.stat_loggers.append(timeline)
            state.collector_timeline = timeline
