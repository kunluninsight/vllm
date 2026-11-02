#!/usr/bin/env python3
"""Process startup for the bookkeeping-only PACE MetaService mock."""

import argparse
import json
import logging
import os
import signal
import sys
import threading
import time

from hisim.simulation.hijack.tair_mempool.mock_meta_service import (
    VERSION,
    MockPace,
    Server,
)
from hisim.simulation.hijack.tair_mempool.mock_provider import (
    LOG,
    TIERING_MODES,
    parse_size,
)

# Real Meta command-line options the mock accepts but does not use, as
# (flag, argparse dest, type / bool for a flag, help tail).
PACE_IGNORED_FLAGS = (
    ("-P", "thread_num", int, "server threads"),
    ("-b", "backend", None, "memory/redis/mysql"),
    ("-c", "backend_args", None, "metadata backend connection string"),
    ("-n", "no_delete_network_err_node", bool, ""),
    ("-t", "delete_node_timeout_s", int, ""),
    ("-l", "leader_election", None, "no/zookeeper/redis"),
    ("-z", "zookeeper_args", None, ""),
    ("-i", "poll_interval_s", int, ""),
    ("-m", "monitor_type", None, "local/kmonitor/prometheus/otel"),
    ("-M", "monitor_port", int, ""),
    ("--exclude_network_interface", "exclude_network_interface", None, ""),
    ("--network_interface", "network_interface", None, ""),
    ("--exclude_subnet", "exclude_subnet", None, ""),
    ("--client_io_threads", "client_io_threads", int, ""),
    ("--meta_thread_pool_size", "meta_thread_pool_size", int, ""),
)
PACE_IGNORED_OPTS = tuple(dest for _, dest, _, _ in PACE_IGNORED_FLAGS)

# --byenv log file when TAIR_MEMPOOL_META_LOG_PATH is unset.
DEFAULT_META_LOG_PATH = "/tmp/meta.log"

# WaterMarkLoop knobs the real Provider takes from env, as (argparse dest, default,
# env); the flag is the dest with dashes. See mock_provider for their effect.
PROVIDER_TIERED_OPTS = (
    ("watermark_check_ms", 100, "TAIR_MEMPOOL_TIERED_WATERMARK_CHECK_MS"),
    ("demote_worker_count", 8, "TAIR_MEMPOOL_TIERED_DEMOTE_WORKER_COUNT"),
    ("demote_batch_size", 8, "TAIR_MEMPOOL_TIERED_DEMOTE_BATCH_SIZE"),
    ("demote_batch_mb", 256, "TAIR_MEMPOOL_TIERED_DEMOTE_BATCH_MB"),
)


def build_nodes(app, cfg):
    if cfg.node_specs:
        for spec in cfg.node_specs:
            app.add_provider(
                spec.get("dram", cfg.dram),
                spec.get("ssd", cfg.ssd),
                ip=spec.get("ip"),
                port=spec.get("port"),
                uid=spec.get("id"),
                healthy=spec.get("healthy", True),
                pace_id=spec.get("pace_node_id"),
                tiering_mode=spec.get("tiering_mode"),
            )
    else:
        for _ in range(cfg.nodes):
            app.add_provider(cfg.dram, cfg.ssd)


def make_parser():
    p = argparse.ArgumentParser(
        description="Bookkeeping-only PACE (Tair Mempool) MetaService + Provider mock; accepts "
        "tair_mempool_meta_service command-line options.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        allow_abbrev=False,
    )
    g = p.add_argument_group(
        "tair_mempool_meta_service options (pace_meta_service.cpp)"
    )
    g.add_argument("-p", dest="port", type=int, default=12348, help="listen port")
    g.add_argument("-o", dest="log_file", default="", help="log file, empty for stderr")
    g.add_argument("-L", dest="log_level", default="INFO", help="log level")
    g.add_argument("-v", action="version", version=VERSION)
    for flag, dest, kind, tail in PACE_IGNORED_FLAGS:
        note = "(ignored)" + (" " + tail if tail else "")
        if kind is bool:
            g.add_argument(flag, dest=dest, action="store_true", help=note)
        else:
            g.add_argument(flag, dest=dest, type=kind, default=None, help=note)
    g.add_argument(
        "--byenv",
        action="store_true",
        help="read TAIR_MEMPOOL_META_* like start_metaservice_byenv.sh "
        "(TAIR_MEMPOOL_META_LOG_PATH defaults to %s)" % DEFAULT_META_LOG_PATH,
    )

    m = p.add_argument_group("mock options (env HISIM_TAIR_MEMPOOL_<NAME>)")
    m.add_argument(
        "--config",
        help="JSON config keyed by option dest, plus an optional node_specs list",
    )
    m.add_argument("--host", default="0.0.0.0", help="bind address")
    m.add_argument(
        "--nodes",
        type=int,
        default=3,
        help="number of virtual Providers (node_specs wins)",
    )
    m.add_argument("--dram", default="64GB", help="DRAM pool per Provider")
    m.add_argument(
        "--ssd", default="512GB", help="LocalSSD capacity per Provider, 0 for none"
    )
    m.add_argument(
        "--tiering-mode",
        choices=TIERING_MODES,
        default="transparent",
        help="default Provider tiering mode",
    )
    m.add_argument(
        "--node-ip-base",
        default="127.0.1.1",
        help="first virtual Provider IP (incremented by pace_node_id)",
    )
    m.add_argument("--node-port", type=int, default=12346)
    for dest, default, env_name in PROVIDER_TIERED_OPTS:
        m.add_argument(
            "--" + dest.replace("_", "-"),
            type=int,
            default=default,
            help="env " + env_name,
        )
    m.add_argument(
        "--demote-high-watermark-pct",
        type=int,
        default=85,
        help="Provider --demote_high_watermark_pct (DramSlotPool clamps to <=85)",
    )
    m.add_argument(
        "--demote-low-watermark-pct",
        type=int,
        default=70,
        help="Provider --demote_low_watermark_pct",
    )
    m.add_argument(
        "--alloc-ssd-full-pct",
        type=int,
        default=95,
        help="SSD-full threshold of SyncDemoteForAlloc fail_fast",
    )
    m.add_argument(
        "--auth",
        action="append",
        default=None,
        metavar="USER:PASS",
        help="initial Basic account, default admin:123",
    )
    m.add_argument("--no-auth", action="store_true", help="disable WithAuth")
    m.add_argument(
        "--delay-ms",
        type=float,
        nargs=2,
        metavar=("MIN", "MAX"),
        default=None,
        help="global injected delay",
    )
    m.add_argument(
        "--state-file",
        default=None,
        help="ledger snapshot file, loaded at start-up when present",
    )
    m.add_argument(
        "--snapshot-interval",
        type=float,
        default=60.0,
        help="periodic snapshot interval (s), 0 to disable",
    )
    m.add_argument(
        "--stats-interval",
        type=float,
        default=30.0,
        help="periodic stats log interval (s), 0 to disable",
    )
    m.add_argument("--access-log", action="store_true", help="log every request")
    return p


def expand_byenv(argv, env):
    """Reproduce start_metaservice_byenv.sh: -o LOG "$@" EXACT STORAGE HA env_args."""
    if "--byenv" not in argv:
        return argv
    rest = [a for a in argv if a != "--byenv"]
    # The real script requires it; the mock falls back to a default.
    log_path = env.get("TAIR_MEMPOOL_META_LOG_PATH") or DEFAULT_META_LOG_PATH
    extra = []
    for key in (
        "TAIR_MEMPOOL_META_EXACT_ARGS",
        "TAIR_MEMPOOL_META_STORAGE",
        "TAIR_MEMPOOL_META_HA",
    ):
        extra += env.get(key, "").split()
    for key, opt in (
        ("TAIR_MEMPOOL_META_EXCLUDE_NIC", "--exclude_network_interface"),
        ("TAIR_MEMPOOL_META_NIC", "--network_interface"),
        ("TAIR_MEMPOOL_META_EXCLUDE_SUBNET", "--exclude_subnet"),
        ("TAIR_MEMPOOL_META_BACKEND_ARGS", "-c"),
    ):
        if env.get(key):
            extra += [opt, env[key]]
    # otel_env.sh appends this last when the exporter falls back to local metrics.
    if env.get("TAIR_MEMPOOL_OTEL_FALLBACK_METRIC_TYPE") == "local":
        extra += ["-m", "local"]
    return ["-o", log_path] + rest + extra


def _provider_env_defaults(env):
    """start_server_byenv.sh provider shape -> virtual Provider defaults.

    Only the capacity-relevant arguments are mapped: -s (DRAM MB) plus the tiering
    flags. push_only skips SSD init (and forces transparent), a tiering value other
    than "true" leaves the Provider without an SSD backend. The virtual Provider
    count has no real counterpart (each Provider is its own process), so it comes
    from HISIM_TAIR_MEMPOOL_NODES like the other mock-only options.
    """
    out = {}
    raw = env.get("TAIR_MEMPOOL_SERVER_LOCAL_MEMORY_SIZE", "")
    if raw:
        tokens = raw.split()
        size = tokens[tokens.index("-s") + 1] if "-s" in tokens[:-1] else ""
        if size.isdigit():
            # The real data pool excludes shm metadata (default 1024 MiB) from -s.
            meta_mb = int(env.get("TAIR_MEMPOOL_SHM_META_SIZE_MB", "1024"))
            out["dram"] = str(max(int(size) - meta_mb, 0)) + "MB"
        else:
            LOG.warning(
                "ignore TAIR_MEMPOOL_SERVER_LOCAL_MEMORY_SIZE=%r without -s <MB>", raw
            )
    push_only = env.get("TAIR_MEMPOOL_PUSH_ONLY_MODE") == "true"
    if push_only or "TAIR_MEMPOOL_ENABLE_TRANSPARENT_TIERING" in env:
        transparent = (
            push_only or env.get("TAIR_MEMPOOL_ENABLE_TRANSPARENT_TIERING") == "true"
        )
        out["tiering_mode"] = "transparent" if transparent else "off"
        if transparent and not push_only:
            # The script materializes 10240M when SSD_SIZE_MB is unset.
            out["ssd"] = (env.get("TAIR_MEMPOOL_SSD_SIZE_MB") or "10240") + "MB"
        else:
            out["ssd"] = "0"
    return out


def _mock_env_defaults(parser, env):
    """Real Provider env plus HISIM_TAIR_MEMPOOL_<DEST> -> argparse defaults (mock options only; HISIM_TAIR_MEMPOOL_* wins)."""
    out = _provider_env_defaults(env)
    for dest, _default, env_name in PROVIDER_TIERED_OPTS:
        raw = env.get(env_name, "")
        if raw:
            try:
                out[dest] = int(raw)
            except ValueError:
                LOG.warning("ignore invalid %s=%r", env_name, raw)
    mock_group = next(
        gr for gr in parser._action_groups if gr.title.startswith("mock options")
    )
    for action in mock_group._group_actions:
        dest = action.dest
        raw = env.get("HISIM_TAIR_MEMPOOL_" + dest.upper())
        if not raw or dest == "config":
            continue
        if action.nargs == 0:
            out[dest] = str(raw).strip().lower() in ("1", "true", "yes", "on")
        elif action.nargs == 2:
            out[dest] = [float(x) for x in str(raw).replace(",", " ").split()]
        elif dest == "auth":
            out[dest] = [x for x in str(raw).split(",") if x]
        else:
            value = action.type(raw) if action.type else raw
            if action.choices is not None and value not in action.choices:
                raise SystemExit(
                    "invalid HISIM_TAIR_MEMPOOL_%s=%r" % (dest.upper(), raw)
                )
            out[dest] = value
    return out


def parse_config(argv=None, env=None):
    env = os.environ if env is None else env
    argv = expand_byenv(list(sys.argv[1:] if argv is None else argv), env)
    parser = make_parser()
    pre, _ = parser.parse_known_args(argv)
    node_specs, defaults = None, {}
    config_path = pre.config or env.get("HISIM_TAIR_MEMPOOL_CONFIG")
    if config_path:
        with open(config_path) as fh:
            conf = json.load(fh)
        node_specs = conf.pop("node_specs", None)
        unknown = set(conf) - {a.dest for a in parser._actions}
        if unknown:
            parser.error("unknown config keys: %s" % sorted(unknown))
        defaults.update(conf)
    defaults.update(_mock_env_defaults(parser, env))
    parser.set_defaults(**defaults)
    cfg = parser.parse_args(argv)
    cfg.node_specs = node_specs
    cfg.argv = argv
    cfg.auth = [] if cfg.no_auth else (cfg.auth or ["admin:123"])
    for key in ("dram", "ssd"):
        parse_size(getattr(cfg, key))
    if (
        cfg.demote_worker_count <= 0
        or not 1 <= cfg.demote_batch_size <= 1024
        or cfg.demote_batch_mb <= 0
        or cfg.watermark_check_ms <= 0
    ):
        parser.error(
            "demote worker/batch/batch_mb/watermark_check_ms must be positive (batch_size <= 1024)"
        )
    return cfg


def setup_logging(cfg):
    name = str(cfg.log_level).upper()
    level = {"WARN": logging.WARNING, "FATAL": logging.CRITICAL}.get(
        name, getattr(logging, name, logging.INFO)
    )
    fmt = "%(asctime)s %(levelname)s %(name)s: %(message)s"
    if cfg.log_file:
        if os.path.dirname(cfg.log_file):
            os.makedirs(os.path.dirname(cfg.log_file), exist_ok=True)
        logging.basicConfig(level=level, format=fmt, filename=cfg.log_file)
    else:
        logging.basicConfig(level=level, format=fmt)


def start_background(app):
    """Harness threads: periodic snapshot / stats log, and the Providers' WaterMarkLoop clock."""
    stop = threading.Event()

    def housekeeping():
        last_snap = last_stats = time.monotonic()
        last_line = ""
        while not stop.wait(1.0):
            now = time.monotonic()
            if (
                app.cfg.state_file
                and app.cfg.snapshot_interval > 0
                and now - last_snap >= app.cfg.snapshot_interval
            ):
                last_snap = now
                try:
                    app.save()
                except Exception:
                    LOG.exception("periodic snapshot failed")
            if (
                app.cfg.stats_interval > 0
                and now - last_stats >= app.cfg.stats_interval
            ):
                last_stats = now
                line = app.summary_line()
                if line != last_line:
                    LOG.info("stats: %s", line)
                    last_line = line

    def watermark_loop():
        interval = app.cfg.watermark_check_ms / 1000.0
        while not stop.wait(interval):
            try:
                app.tiering_tick_all()
            except Exception:
                LOG.exception("tiering tick failed")

    threading.Thread(target=housekeeping, name="mock-pace-bg", daemon=True).start()
    threading.Thread(
        target=watermark_loop, name="mock-pace-tiering", daemon=True
    ).start()
    return stop


def serve(cfg):
    app = MockPace(cfg)
    if cfg.state_file and os.path.exists(cfg.state_file):
        app.load(cfg.state_file)
    else:
        build_nodes(app, cfg)
    server = Server((cfg.host, cfg.port), app)
    stop = start_background(app)

    def on_signal(signum, _frame):
        LOG.info("signal %d received, shutting down", signum)
        app.meta.shutting_down = True
        stop.set()
        threading.Thread(target=server.shutdown, daemon=True).start()

    signal.signal(signal.SIGTERM, on_signal)
    signal.signal(signal.SIGINT, on_signal)
    ignored = [
        "%s=%s" % (d, "(hidden)" if d == "backend_args" else getattr(cfg, d))
        for d in PACE_IGNORED_OPTS
        if getattr(cfg, d, None) not in (None, False)
    ]
    if ignored:
        LOG.info(
            "pace meta options accepted but not used by mock: %s", ", ".join(ignored)
        )
    LOG.info(
        "mock pace listening on http://%s:%d auth=%s providers=%d ga_alloc_config=%s",
        cfg.host,
        server.server_address[1],
        "on" if app.meta.users else "off",
        len(app.meta.provider_list()),
        app.meta.ga_cfg.get(),
    )
    LOG.info("stats: %s", app.summary_line())
    print(
        "mock pace listening on http://%s:%d (log: %s)"
        % (cfg.host, server.server_address[1], cfg.log_file or "stderr"),
        flush=True,
    )
    try:
        server.serve_forever(poll_interval=0.5)
    finally:
        server.server_close()
        if cfg.state_file:
            try:
                app.save()
            except Exception:
                LOG.exception("final snapshot failed")


def main(argv=None):
    cfg = parse_config(argv)
    setup_logging(cfg)
    serve(cfg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
