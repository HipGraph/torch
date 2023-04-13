import torch
import torch_geometric
import numpy as np
import sys
import inspect
from torch.utils.tensorboard import SummaryWriter as SW
from torch.utils.tensorboard import _pytorch_graph


# Setup str -> function maps for all PyTorch and PYG functions/classes needed for model definition.
#   These maps allow users to call, for example, the LSTM module using layer_fn_map["LSTM"]().
#   This allows for higher-level model definitions that are agnostic of chosen layers, activations, etc.
#   For example, we can now define a cell-agnostic RNN model by passing a layer argument (e.g. layer="LSTM")
#       that specifies the specific layer to use. Thus, this model would define a general RNN architecture,
#       such as sequence-to-sequence, with the specific layer type as a hyper-parameter.

#   init activation -> function map
act_fn_map = {"identity": lambda x:x}
for name, fn in inspect.getmembers(torch.nn.functional, inspect.isfunction):
    if not name.startswith("_"):
        act_fn_map[name] = fn

#   init optimization -> class constructor function map
opt_fn_map = {}
for name, fn in inspect.getmembers(torch.optim, inspect.isclass):
    opt_fn_map[name] = fn

#   init scheduler -> function map
sched_fn_map = {}
for name, fn in inspect.getmembers(torch.optim.lr_scheduler, inspect.isclass):
    sched_fn_map[name] = fn

#   init initialization -> function map
init_fn_map = {}
for name, fn in inspect.getmembers(torch.nn.init, inspect.isfunction):
    if not name.startswith("_"):
        init_fn_map[name] = fn

#   init loss -> class constructor function map
loss_fn_map = {}
for name, fn in inspect.getmembers(torch.nn.modules.loss, inspect.isclass):
    if not name.startswith("_"):
        loss_fn_map[name] = fn

#   init layer -> class constructor function map
layer_fn_map = {}
for name, fn in inspect.getmembers(torch.nn.modules, inspect.isclass):
    if not name.startswith("_") and not "Loss" in name:
        layer_fn_map[name] = fn


class EarlyStopper:

    def __init__(self, patience, init_steps=0):
        self.patience = patience
        self.n_plateau_steps = init_steps
        self.min_loss = sys.float_info.max

    def step(self, loss):
        if loss < self.min_loss:
            self.min_loss = loss
            self.n_plateau_steps = 0
        else:
            self.n_plateau_steps += 1
        return self.stop()

    def stop(self):
        return self.patience > 0 and self.n_plateau_steps >= self.patience


# Override add_graph() from tensorboard.SummaryWriter to allow the passage of paramters to trace()
#   This is needed to pass strict=False so that models with dictionary outputs may be logged with add_graph()
class SummaryWriter(SW):

    def add_graph(self, model, input_to_model=None, verbose=False, trace_kwargs={}):
        self._get_file_writer().add_graph(graph(model, input_to_model, verbose, trace_kwargs))


# Override graph() from tensorboard._pytorch_graph to allow the passage of paramters to trace()
#   This is needed to pass strict=False so that models with dictionary outputs may be logged with add_graph()
def graph(model, args, verbose=False, trace_kwargs={}):
    with torch.onnx.select_model_mode_for_export(model, torch.onnx.TrainingMode.EVAL):
        try:
            trace = torch.jit.trace(model, args, **trace_kwargs)
            graph = trace.graph
            torch._C._jit_pass_inline(graph)
        except RuntimeError as e:
            print(e)
            print('Error occurs, No graph saved')
            raise e
        if verbose:
            print(graph)
    list_of_nodes = _pytorch_graph.parse(graph, trace, args)
    stepstats = _pytorch_graph.RunMetadata(
        step_stats=_pytorch_graph.StepStats(
            dev_stats=[_pytorch_graph.DeviceStepStats(device="/device:CPU:0")]
        )
    )
    graph_def = _pytorch_graph.GraphDef(
        node=list_of_nodes,
        versions=_pytorch_graph.VersionDef(producer=22)
    )
    return graph_def, stepstats


def l1_reg_grad(weight, factor, weight_target=0):
    return factor * torch.sign(weight - weight_target)


def l2_reg_grad(weight, factor, weight_target=0):
    return factor * (weight - weight_target)


def unravel_index(indices, shape):
    """Converts flat indices into unraveled coordinates in a target shape. This is a `torch` implementation of `numpy.unravel_index`.

    Arguments
    ---------
    indices : LongTensor with shape=(?, N)
    shape : tuple with shape=(D,)

    Returns
    -------
    coord : LongTensor with shape=(?, N, D)

    Source
    ------
    author : francois-rozet @ https://github.com/pytorch/pytorch/issues/35674#issuecomment-739492875

    """
    shape = torch.tensor(shape)
    indices = indices % shape.prod()  # prevent out-of-bounds indices
    coord = torch.zeros(indices.size() + shape.size(), dtype=indices.dtype, device=indices.device)
    for i, dim in enumerate(reversed(shape)):
        coord[..., i] = indices % dim
        indices = torch.div(indices, dim, rounding_mode="trunc")
    return coord.flip(-1)


def batch_sampler_collate(batch):
    return batch[0]


def bcast_safe_view(x, y, matched_dim=-1):
    """ Reshapes x to match dimensionality of y by adding dummy axes so that any broadcasted operation on x and y is deterministic

    Arguments
    ---------
    x : tensor with len(x.shape) <= len(y.shape)
        the input data to be reshaped
    y : tensor
        the target data that x will be reshaped to
    matched_dim : int or tuple of ints
        dimensions/axis in which x and y have one-to-one correpsondence. input x will be broadcasted to all other dimensions/axes of y 
    """
    if isinstance(matched_dim, int):
        matched_dim = [matched_dim]
    bcast_safe_shape = list(1 for _ in range(len(y.shape)))
    for dim in matched_dim:
        bcast_safe_shape[dim] = y.shape[dim]
    return x.view(bcast_safe_shape)


def align_dims(x, y, x_dim, y_dim):
    if len(x.shape) > len(y.shape):
        raise ValueError(
            "Input x must have fewer dimensions than y to be aligned. Received x.shape=%s and y.shape=%s" % (
                x.shape, y.shape
            )
        )
    elif len(x.shape) == len(y.shape):
        return x
    if x_dim < 0:
        x_dim = len(x.shape) + x_dim
    if y_dim < 0:
        y_dim = len(y.shape) + y_dim
    new_shape = [1 for _ in y.shape]
    start = y_dim - x_dim
    end = start + len(x.shape)
    new_shape[start:end] = x.shape
    return x.view(new_shape)


def align(inputs, dims):
    if not (isinstance(inputs, tuple) or isinstance(inputs, list)) or not all([isinstance(inp, torch.Tensor) for inp in inputs]):
        raise ValueError("Argumet inputs must be tuple or list of tensors")
    if len(inputs) < 2:
        return inputs
    if isinstance(dims, int):
        dims = tuple(dims for inp in inputs)
    elif not isinstance(dim, tuple):
        raise ValueError("Argument dim must be int or tuple of ints")
    input_dims = [inp.dim() for inp in inputs]
    idx = input_dims.index(max(input_dims))
    y = inputs[idx]
    y_dim = dims[idx]
    return [align_dims(inp, y, dim, y_dim) for inp, dim in zip(inputs, dims)]


def maybe_expand_then_cat(tensors, dim=0):
    debug = 0
    if debug:
        for tensor in tensors:
            print(tensor.shape)
    dims = [tensor.dim() for tensor in tensors]
    if debug:
        print(dims)
    idx = dims.index(max(dims))
    if debug:
        print(idx)
    shape = list(tensors[idx].shape)
    if debug:
        print(shape)
    shape[dim] = -1
    if debug:
        print(shape)
    tensors = align(tensors, dim)
    for i, tensor in enumerate(tensors):
        tensors[i] = tensor.expand(shape)
    return torch.cat(tensors, dim)


def squeeze_not(x, dim):
    debug = 0
    if isinstance(dim, int):
        dim = (dim,)
    elif not isinstance(dim, tuple):
        raise TypeError(dim)
    orig_dim = x.dim()
    dims = list(range(orig_dim))
    for i in range(len(dim)):
        _dim = dim[i]
        if _dim < 0:
            _dim = orig_dim + _dim
        dims.remove(_dim)
    if debug:
        print("squeeze_dims =", dims)
        print(x.size())
    for i, _dim in enumerate(dims):
        _dim -= (orig_dim - x.dim())
        x = torch.squeeze(x, _dim)
        if debug:
            print("_dim =", _dim, "x =", x.size())
    if debug:
        input()
    return x
