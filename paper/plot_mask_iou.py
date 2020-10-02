import lib
from lib import StatTracker

runs = lib.get_runs(["cifar10_mask_stability"])

stat = StatTracker()
for r in runs:
    for k, v in r.summary.items():
        if k.startswith("masks_stability/") and "/pair_" in k:
            stat.add(v)

print(stat)
print("Count", stat.n)