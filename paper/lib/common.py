from typing import List, Dict
from .stat_tracker import StatTracker


def construct_name(config_names: List[str], config: Dict[str, any]) -> str:
    return "/".join([f"{c}_{config[c]}" for c in config_names])


def group(runs, config_names: List[str]) -> Dict[str, any]:
    res = {}
    for r in runs:
        cval = construct_name(config_names, r.config)
        print(cval)
        if cval not in res:
            res[cval] = []

        res[cval].append(r)

    return res

def calc_stat(group_of_runs: Dict[str, List[any]], filter) -> Dict[str, Dict[str, StatTracker]]:
    all_stats = {}

    for k, rn in group_of_runs.items():
        if k not in all_stats:
            all_stats[k] = {}

        stats = all_stats[k]

        for r in rn:
            for k, v in r.summary.items():
                if not filter(k):
                    continue

                if k not in stats:
                    stats[k] = StatTracker()

                stats[k].add(v)

    return all_stats