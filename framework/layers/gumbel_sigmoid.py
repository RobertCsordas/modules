import torch


def gumbel_sigmoid(logits: torch.Tensor, tau: float = 1, hard: bool = False, eps: float = 1e-10) -> torch.Tensor:
    uniform = logits.new_empty([2]+list(logits.shape)).uniform_(0,1)

    noise = -((uniform[1] + eps).log() / (uniform[0] + eps).log() + eps).log()
    res = torch.sigmoid((logits + noise) / tau)

    if hard:
        res = ((res > 0.5).type_as(res) - res).detach() + res

    return res


def sigmoid(logits: torch.Tensor, mode: str = "simple", tau: float = 1, eps: float = 1e-10):
    if mode=="simple":
        return torch.sigmoid(logits)
    elif mode in ["soft", "hard"]:
        return gumbel_sigmoid(logits, tau, hard=mode=="hard", eps=eps)
    else:
        assert False, "Invalid sigmoid mode: %s" % mode
