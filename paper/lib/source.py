import wandb
from typing import List
from .config import get_config


def get_runs(names: List[str]) -> List[wandb.apis.public.Run]:
    api = wandb.Api()
    config = get_config()
    res = list(api.runs(config["wandb_project"], {"config.name": {"$in": [f"sweep_{n}" for n in names]}}))
    assert all(r.state == "finished" for r in res)
    print(f"Querying runs {names}: {len(res)} runs loaded")
    assert len(res) > 0
    return res
