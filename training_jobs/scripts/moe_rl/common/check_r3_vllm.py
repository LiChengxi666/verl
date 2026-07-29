#!/usr/bin/env python

import sys


def supports_r3_router_replay(async_engine_args_cls: type) -> bool:
    fields = getattr(async_engine_args_cls, "__dataclass_fields__", {})
    return "enable_return_routed_experts" in fields


def main() -> int:
    from vllm.engine.arg_utils import AsyncEngineArgs

    if supports_r3_router_replay(AsyncEngineArgs):
        return 0

    print(
        "R3 requires a Router-Replay-enabled vLLM with "
        "AsyncEngineArgs.enable_return_routed_experts",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
