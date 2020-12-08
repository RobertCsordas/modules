import lib
from collections import OrderedDict
import matplotlib.pyplot as plt
import os

def get_stats(runs, accuracy_key="validation/iid/accuracy/total"):
    avg_normal = lib.StatTracker()
    avg_inv = lib.StatTracker()
    for r in runs:
        for k, v in r.summary.items():
            if not k.startswith("half_mask_test/") or not k.endswith("iid/accuracy"):
                continue

            hist = r.history(keys=[k], pandas=False)
            assert len(hist)==1
            hist = hist[0]

            h = list(r.scan_history(keys=[accuracy_key], min_step=hist["_step"]-1,
                                    max_step=hist["_step"]+1))
            assert len(h)==1
            ref_accuracy = h[0][accuracy_key]

            val = (ref_accuracy - hist[k])*100

            # Fast hack for editing the plot
            # val = (1-v)*100

            (avg_normal if k.startswith("half_mask_test/normal") else avg_inv).add(val)

    return avg_normal.get(), avg_inv.get()

stats = OrderedDict()
stats["SCAN trafo"] = get_stats(lib.get_runs(["trafo_scan_halfmask_inv"]))
stats["$+/*$ FNN"] = get_stats(lib.get_runs(["addmul_feedforward_halfmask_inv"], filters={"config.layer_sizes": "800,800,800,800"}), accuracy_key="validation/iid/accuracy")
stats["$+/+$ LSTM"] = get_stats([r for r in lib.get_runs(["tuple_rnn_halfmask_inv"]) if r.config["tuple.mode"] == "together"])
stats["$+/+$ FNN Big"] = get_stats(lib.get_runs(["tuple_feedforward_big_halfmask_inv"]))
stats["$+/+$ FNN Small"] = get_stats(lib.get_runs(["tuple_feedforward_small_halfmask_inv"]))
stats["CIFAR 10 simple"] = get_stats(lib.get_runs(["cifar10_halfmask_inv"]), accuracy_key="validation/iid/accuracy")

for k, v in stats.items():
    for i in range(2):
        assert v[i].n==10, f"Invalid number of elements for {k}: {v[i].n}"

fig = plt.figure(figsize=[6, 1.2])
for j in range(2):
    plt.bar([i*2.5+j for i in range(len(stats))], [s[j].mean for s in stats.values()], yerr=[s[j].std for s in stats.values()], align='center')
plt.xticks([2.5*i + 0.5 for i in range(len(stats))], stats.keys())
plt.ylabel("Accuracy drop [\\%]")
plt.ylim(0,100)
for tick in fig.axes[0].xaxis.get_major_ticks()[1::2]:
    tick.set_pad(15)
fig.axes[0].yaxis.set_label_coords(-0.085, 0.45)
plt.legend(["Early masked", "Late masked"])

os.makedirs("out", exist_ok=True)
fig.savefig(f"out/half_mask.pdf", bbox_inches='tight', pad_inches = 0.01)
