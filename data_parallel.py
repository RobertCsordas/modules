import torch
from torch.cuda._utils import _get_device_index
from itertools import chain
from torch.nn.parallel.scatter_gather import scatter_kwargs, gather
from torch.nn.parallel.parallel_apply import parallel_apply
from typing import Callable, List

Replicator = Callable[[torch.nn.Module, List[int]], List[torch.nn.Module]]


# Copy of data_parallel from pytorch, but with custom replicate.
def data_parallel(module, inputs, replicate: Replicator, device_ids=None, output_device=None, dim=0,
                  module_kwargs=None) -> torch.Tensor:
    r"""Evaluates module(input) in parallel across the GPUs given in device_ids.
    This is the functional version of the DataParallel module.
    Args:
        module (Module): the module to evaluate in parallel
        inputs (Tensor): inputs to the module
        device_ids (list of int or torch.device): GPU ids on which to replicate module
        output_device (list of int or torch.device): GPU location of the output  Use -1 to indicate the CPU.
            (default: device_ids[0])
    Returns:
        a Tensor containing the result of module(input) located on
        output_device
    """
    if not isinstance(inputs, tuple):
        inputs = (inputs,)

    if device_ids is None:
        device_ids = list(range(torch.cuda.device_count()))

    if output_device is None:
        output_device = device_ids[0]

    device_ids = list(map(lambda x: _get_device_index(x, True), device_ids))
    output_device = _get_device_index(output_device, True)
    src_device_obj = torch.device("cuda:{}".format(device_ids[0]))

    for t in chain(module.parameters(), module.buffers()):
        if t.device != src_device_obj:
            raise RuntimeError("module must have its parameters and buffers "
                               "on device {} (device_ids[0]) but found one of "
                               "them on device: {}".format(src_device_obj, t.device))

    inputs, module_kwargs = scatter_kwargs(inputs, module_kwargs, device_ids, dim)
    if len(device_ids) == 1:
        return module(*inputs[0], **module_kwargs[0])
    used_device_ids = device_ids[:len(inputs)]
    replicas = replicate(module, used_device_ids)
    outputs = parallel_apply(replicas, inputs, module_kwargs, used_device_ids)
    return gather(outputs, output_device, dim)