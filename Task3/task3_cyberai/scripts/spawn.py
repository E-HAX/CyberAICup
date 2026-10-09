"""Fire-and-forget launcher for the deployed Modal app.

`modal run` creates an ephemeral app that is torn down the moment the local
entrypoint returns, which cancels anything it spawned. Deploying the app once
and spawning against the deployment instead makes long sweeps independent of
this machine entirely: the call keeps running whether or not the laptop is on.

    modal deploy src/modal_app.py
    python scripts/spawn.py search_stage '{"stage": "all_trees"}'
    python scripts/spawn.py --result fc-XXXX
"""

from __future__ import annotations

import json
import sys

import modal

APP = "rtc-cyberai"


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__)
        return 1
    if argv[0] == "--result":
        call = modal.FunctionCall.from_id(argv[1])
        try:
            print(json.dumps(call.get(timeout=5), indent=2, default=str))
        except TimeoutError:
            print("still running")
        return 0

    name = argv[0]
    kwargs = json.loads(argv[1]) if len(argv) > 1 else {}
    fn = modal.Function.from_name(APP, name)
    call = fn.spawn(**kwargs)
    print(call.object_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
