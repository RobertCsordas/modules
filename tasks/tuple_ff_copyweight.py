from .tuple_ff import TupleTaskFeedforward
import torch


class TupleFeedforwardCopyweightTask(TupleTaskFeedforward):
    def prepare_model_for_analysis(self):
        super().prepare_model_for_analysis()

        with torch.no_grad():
            w = self.model.model_parameters["layers_0_weight"]
            w[:, : w.shape[1] // 2] = w[:, w.shape[1] // 2:]

            last_layer = len(self.helper.opt.layer_sizes)
            for n in self.model.model_parameters.keys():
                if n.startswith(f"layers_{last_layer}_"):
                    w = self.model.model_parameters[n]
                    w[:w.shape[0] // 2] = w[w.shape[0] // 2:]
