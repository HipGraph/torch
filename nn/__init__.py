import torch
import torch.nn.functional as F

from .. import util as pyt_util

import Utility as util


class gLSTMCell(torch.nn.Module):

    debug = 0

    def __init__(self, in_size, out_size, **kwargs):
        super(gLSTMCell, self).__init__()
        n_groups = kwargs.get("n_groups", 1)
        xs_size = kwargs.get("xs_size", 0)
        arch = kwargs.get("arch", 0)
        # weights for input x
        self.weight_ih = torch.nn.Parameter(torch.ones((4, n_groups, out_size, in_size)))
        self.bias_ih = torch.nn.Parameter(torch.ones((4, n_groups, out_size)))
        # weights for input x_s
        if arch > 0:
            self.weight_sh = torch.nn.Parameter(torch.ones((4, n_groups, out_size, xs_size)))
            self.bias_sh = torch.nn.Parameter(torch.ones((4, n_groups, out_size)))
        # weights for input h
        self.weight_hh = torch.nn.Parameter(torch.ones((4, n_groups, out_size, out_size)))
        self.bias_hh = torch.nn.Parameter(torch.ones((4, n_groups, out_size)))
        # vars
        self.in_size = in_size
        self.out_size = out_size
        self.n_groups = n_groups
        self.xs_size = xs_size
        self.arch = arch

    def forward(self, x, x_s=None, group_index=None, hx=None, **kwargs):
        node_dim = kwargs.get("node_dim", -2)
        if self.debug:
            print(util.make_msg_block("gLSTM Forward"))
            print("x =", x.shape)
        if self.debug and not x_s is None:
            print("x_s =", x_s.shape)
        if self.debug and not group_index is None:
            print("group_index =", group_index.shape)
        if self.debug and not hx is None:
            print("hx =", (hx[0].shape, hx[1].shape))
        if 0:
            if group_index is None:
                n_nodes = x.shape[node_dim]
                group_index = torch.arange(n_nodes, dtype=torch.long, device=x.device) % self.n_groups
        if hx is None:
            zeros = torch.zeros(x.shape[:2]+(self.out_size,), dtype=x.dtype, device=x.device)
            hx = (zeros, zeros)
        h, c = hx
        # unpack weights for input x
        w_ih = torch.index_select(self.weight_ih, 1, group_index)
        b_ih = torch.index_select(self.bias_ih, 1, group_index)
        w_ii, w_if, w_ig, w_io = w_ih
        b_ii, b_if, b_ig, b_io = b_ih
        if self.debug:
            print("w_ih =", w_ih.shape)
            print("b_ih =", b_ih.shape)
        # unpack weights for input x_s
        if self.arch > 0:
            w_sh = torch.index_select(self.weight_sh, 1, group_index)
            b_sh = torch.index_select(self.bias_sh, 1, group_index)
            w_si, w_sf, w_sg, w_so = w_sh
            b_si, b_sf, b_sg, b_so = b_sh
            if self.debug:
                print("w_sh =", w_sh.shape)
                print("b_sh =", b_sh.shape)
        # unpack weights for input h
        w_hh = torch.index_select(self.weight_hh, 1, group_index)
        b_hh = torch.index_select(self.bias_hh, 1, group_index)
        w_hi, w_hf, w_hg, w_ho = w_hh
        b_hi, b_hf, b_hg, b_ho = b_hh
        if self.debug:
            print("w_hh =", w_hh.shape)
            print("b_hh =", b_hh.shape)
        # Input Gate Forward
        if not x_s is None and self.arch & 1:
            i = torch.sigmoid(
                self.linear_x(x, w_ii, b_ii), self.linear_x(x_s, w_si, b_si)[None,:,:] + self.linear_h(h, w_hi, b_hi)
            )
        else:
            i = torch.sigmoid(self.linear_x(x, w_ii, b_ii) + self.linear_h(h, w_hi, b_hi))
        # Forget Gate Forward
        if not x_s is None and self.arch & 2:
            f = torch.sigmoid(
                self.linear_x(x, w_if, b_if), self.linear_x(x_s, w_sf, b_sf)[None,:,:] + self.linear_h(h, w_hf, b_hf)
            )
        else:
            f = torch.sigmoid(self.linear_x(x, w_if, b_if) + self.linear_h(h, w_hf, b_hf))
        # ? Gate Forward
        if not x_s is None and self.arch & 4:
            g = torch.tanh(
                self.linear_x(x, w_ig, b_ig), self.linear_x(x_s, w_sg, b_sg)[None,:,:] + self.linear_h(h, w_hg, b_hg)
            )
        else:
            g = torch.tanh(self.linear_x(x, w_ig, b_ig) + self.linear_h(h, w_hg, b_hg))
        # Ouput Gate Forward
        if not x_s is None and self.arch & 8:
            o = torch.sigmoid(
                self.linear_x(x, w_io, b_io), self.linear_x(x_s, w_so, b_so)[None,:,:] + self.linear_h(h, w_ho, b_ho)
            )
        else:
            o = torch.sigmoid(self.linear_x(x, w_io, b_io) + self.linear_h(h, w_ho, b_ho))
        if self.debug:
            print("i =", i.shape)
            print("f =", f.shape)
            print("g =", g.shape)
            print("o =", o.shape)
        c_ = f * c + i * g
        h_ = o * torch.tanh(c_)
        if self.debug:
            print("hx =", (c_.shape, h_.shape))
        return (h_, c_)

    def linear_x(self, x, w, b):
        # Dimensions:
        #   N - n_samples (batch size)
        #   V - n_nodes (graph size)
        #   I - n_inputs (input size)
        #   O - n_outputs (output size)
        # Inputs:
        #   x.shape=(V, I) | x.shape=(N, V, I)
        #   w.shape=(V, O, I)
        #   b.shape=(V, O)
        # Returns:
        #   y.shape=(V, O) | y.shape=(N, V, O)
        if self.debug:
            print("### gLSTM.linear_x() Forward ###")
            print("x =", x.shape)
            print("w =", w.shape)
            print("b =", b.shape)
        if len(x.shape) == 2:
            y = torch.einsum("VI,VOI->VO", x, w) + b
        elif len(x.shape) == 3:
            y = torch.einsum("NVI,VOI->NVO", x, w) + b[None,:,:]
        else:
            raise ValueError(x.shape)
        return y

    def linear_h(self, h, w, b):
        # Dimensions:
        #   N - n_samples (batch size)
        #   V - n_nodes (graph size)
        #   O - n_outputs (output size)
        # Inputs:
        #   h.shape=(N, V, O)
        #   w.shape=(V, O, O)
        #   b.shape=(V, O)
        # Returns:
        #   y.shape=(N, V, O)
        if self.debug:
            print("### gLSTM.linear_h() Forward ###")
            print("h =", h.shape)
            print("w =", w.shape)
            print("b =", b.shape)
        return torch.einsum("NVO,VOO->NVO", h, w) + b[None,:,:]

    def add(self, x, x_s):
        return (x.view(-1, x_s.shape[0], x.shape[-1]) + x_s).view(x.shape)
        return torch.reshape(torch.reshape(x, (-1, x_s.shape[0], x.shape[-1])) + x_s, x.shape)

    def reset_parameters(self):
        stdv = 1.0 / self.out_size**(1/2) if self.out_size > 0 else 0
        for weight in self.parameters():
            torch.nn.init.uniform_(weight, -stdv, stdv)


class mLSTMCell(torch.nn.Module):

    debug = 0

    def __init__(self, in_size, out_size, **kwargs):
        super(mLSTMCell, self).__init__()
        xs_size = kwargs.get("xs_size", 0)
        arch = kwargs.get("arch", 0)
        # weights for input x
        self.weight_ih = torch.nn.Parameter(torch.ones((4, out_size, in_size)))
        self.bias_ih = torch.nn.Parameter(torch.ones((4, out_size)))
        # weights for input x_s
        if arch > 0:
            self.weight_sh = torch.nn.Parameter(torch.ones((4, out_size, xs_size)))
            self.bias_sh = torch.nn.Parameter(torch.ones((4, out_size)))
        # weights for input h
        self.weight_hh = torch.nn.Parameter(torch.ones((4, out_size, out_size)))
        self.bias_hh = torch.nn.Parameter(torch.ones((4, out_size)))
        # vars
        self.in_size = in_size
        self.out_size = out_size
        self.xs_size = xs_size
        self.arch = arch

    def forward(self, x, x_s, hx):
        if self.debug:
            print(util.make_msg_block("mLSTM Forward"))
            print("x =", x.shape)
        if self.debug and not x_s is None:
            print("x_s =", x_s.shape)
        if hx is None:
            zeros = torch.zeros((x.size(0), self.out_size), dtype=x.dtype, device=x.device)
            hx = (zeros, zeros)
        h, c = hx
        if self.debug:
            print("hx =", (h.shape, c.shape))
        # unpack weights for input x
        w_ii, w_if, w_ig, w_io = self.weight_ih
        b_ii, b_if, b_ig, b_io = self.bias_ih
        if self.debug:
            print(self.weight_ih.shape)
            print(self.bias_ih.shape)
        # unpack weights for input x_s
        w_si, w_sf, w_sg, w_so = self.weight_sh
        b_si, b_sf, b_sg, b_so = self.bias_sh
        if self.debug:
            print(self.weight_sh.shape)
            print(self.bias_sh.shape)
        # unpack weights for input h
        w_hi, w_hf, w_hg, w_ho = self.weight_hh
        b_hi, b_hf, b_hg, b_ho = self.bias_hh
        if self.debug:
            print(self.weight_hh.shape)
            print(self.bias_hh.shape)
        # Input Gate Forward
        if not x_s is None and self.arch & 1:
            i = torch.sigmoid(self.add(F.linear(x, w_ii, b_ii), F.linear(x_s, w_si, b_si)) + F.linear(h, w_hi, b_hi))
        else:
            i = torch.sigmoid(F.linear(x, w_ii, b_ii) + F.linear(h, w_hi, b_hi))
        # Forget Gate Forward
        if not x_s is None and self.arch & 2:
            f = torch.sigmoid(self.add(F.linear(x, w_if, b_if), F.linear(x_s, w_sf, b_sf)) + F.linear(h, w_hf, b_hf))
        else:
            f = torch.sigmoid(F.linear(x, w_if, b_if) + F.linear(h, w_hf, b_hf))
        # ? Gate Forward
        if not x_s is None and self.arch & 4:
            g = torch.tanh(self.add(F.linear(x, w_ig, b_ig), F.linear(x_s, w_sg, b_sg)) + F.linear(h, w_hg, b_hg))
        else:
            g = torch.tanh(F.linear(x, w_ig, b_ig) + F.linear(h, w_hg, b_hg))
        # Ouput Gate Forward
        if not x_s is None and self.arch & 8:
            o = torch.sigmoid(self.add(F.linear(x, w_io, b_io), F.linear(x_s, w_so, b_so)) + F.linear(h, w_ho, b_ho))
        else:
            o = torch.sigmoid(F.linear(x, w_io, b_io) + F.linear(h, w_ho, b_ho))
        if self.debug:
            print(i.shape, f.shape, g.shape, o.shape)
        c_ = f * c + i * g
        h_ = o * torch.tanh(c_)
        if self.debug:
            print(c_.shape, h_.shape)
        return (h_, c_)

    def add(self, x, x_s):
        return (x.view(-1, x_s.shape[0], x.shape[-1]) + x_s).view(x.shape)
        return torch.reshape(torch.reshape(x, (-1, x_s.shape[0], x.shape[-1])) + x_s, x.shape)

    def reset_parameters(self):
        stdv = 1.0 / self.out_size**(1/2) if self.out_size > 0 else 0
        for weight in self.parameters():
            torch.nn.init.uniform_(weight, -stdv, stdv)


class gLinear(torch.nn.Module):

    debug = 0

    def __init__(self, in_size, out_size, n_groups=1, bias=True):
        super(gLinear, self).__init__()
        self.weight = torch.nn.Parameter(torch.ones((n_groups, out_size, in_size)))
        if bias:
            self.bias = torch.nn.Parameter(torch.ones((n_groups, out_size)))
        else:
            self.bias = None
        self.in_size = in_size
        self.out_size = out_size
        self.n_groups = n_groups

    def forward(self, x, group_index=None):
        w = torch.index_select(self.weight, 0, group_index)
        if not self.bias is None:
            b = torch.index_select(self.bias, 0, group_index)
        if self.debug:
            print("x =", x.shape)
            print("w =", w.shape)
            if not self.bias is None:
                print("b =", b.shape)
        if len(x.shape) == 2:
            if self.bias is None:
                y = torch.einsum("VI,VOI->VO", x, w)
            else:
                y = torch.einsum("VI,VOI->VO", x, w) + b
        elif len(x.shape) == 3:
            if self.bias is None:
                y = torch.einsum("NVI,VOI->NVO", x, w)
            else:
                y = torch.einsum("NVI,VOI->NVO", x, w) + b[None,:,:]
        elif len(x.shape) == 4:
            if self.bias is None:
                y = torch.einsum("NTVI,VOI->NTVO", x, w)
            else:
                y = torch.einsum("NTVI,VOI->NTVO", x, w) + b[None,None,:,:]
        else:
            raise ValueError(x.shape)
        return y

    def reset_parameters(self):
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_size), 1/sqrt(in_size)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        torch.nn.init.kaiming_uniform_(self.weight, a=5**(1/2))
        if not self.bias is None:
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / fan_in**(1/2) if fan_in > 0 else 0
            torch.nn.init.uniform_(self.bias, -bound, bound)


class _pcLinear(torch.nn.Module):

    debug = 0

    def __init__(self, period_size, n_heads=1, bias=True):
        super(pcLinear, self).__init__()
        if n_heads < 1:
            raise ValueError("Input n_heads must be positive. Received n_heads=%d" % (n_heads))
        elif n_heads == 1:
            self.w = torch.nn.Parameter(torch.ones((period_size,)))
            if bias:
                self.b = torch.nn.Parameter(torch.zeros((period_size,)))
        else:
            self.w = torch.nn.Parameter(torch.ones((period_size, n_heads)))
            if bias:
                self.b = torch.nn.Parameter(torch.zeros((period_size, n_heads)))
        self.period_size = period_size
        self.n_heads = n_heads
        self.bias = bias

    def forward(self, x, periodic_indices):
#        self.debug = 2
        if self.debug:
            print(util.make_msg_block("pcLinear Forward"))
        if self.debug:
            print("x =", x.shape)
            if self.debug > 1:
                print(x)
        if self.debug:
            print("periodic_indices =", periodic_indices.shape)
            if self.debug > 1:
                print(periodic_indices)
        if self.n_heads == 1:
            w = self.w[periodic_indices][:,:,None,None]
            if self.bias:
                b = self.b[periodic_indices][:,:,None,None]
        else:
            w = self.w[periodic_indices,:][:,:,None,:]
            if self.bias:
                b = self.b[periodic_indices,:][:,:,None,:]
        if self.debug:
            print("w =", w.shape)
            if self.debug > 1:
                print(w)
        if self.debug and self.bias:
            print("b =", w.shape)
            if self.debug > 1:
                print(b)
        z = w * x + b if self.bias else w * x
        if self.debug:
            print("z =", z.shape)
            if self.debug > 1:
                print(z)
        return z


class pcLinear(torch.nn.Module):

    debug = 0

    def __init__(self, in_size, out_size, period_size, bias=True):
        super(pcLinear, self).__init__()
        self.weight = torch.nn.Parameter(torch.empty((period_size, out_size, in_size)))
        if bias:
            self.bias = torch.nn.Parameter(torch.empty(period_size, out_size))
        else:
            self.register_parameter("bias", None)
        self.in_size = in_size
        self.out_size = out_size
        self.period_size = period_size
        self.reset_parameters()

    def forward(self, x, periodic_index):
#        self.debug = 2
        if self.debug:
            print(util.make_msg_block("pcLinear Forward"))
        if self.debug:
            print("x =", x.shape) # shape=(n_sample, n_temporal, n_spatial, in_size)
            if self.debug > 1:
                print(x)
        if self.debug:
            print("periodic_index =", periodic_index.shape)
            if self.debug > 1:
                print(periodic_index)
        w = self.weight[periodic_index,:,:]
        if not self.bias is None:
            b = self.bias[periodic_index,:]
        if self.debug:
            print("w =", w.shape) # shape=(n_sample, n_temporal, out_size, in_size)
            if self.debug > 1:
                print(w)
        if self.debug and not self.bias is None:
            print("b =", w.shape) # shape=(n_sample, n_temporal, out_size)
            if self.debug > 1:
                print(b)
        a = torch.einsum("ntoi,ntsi->ntso", w, x)
        if not self.bias is None:
            a += b[:,:,None,:]
        if self.debug:
            print("a =", a.shape)
            if self.debug > 1:
                print(a)
        return a

    def reset_parameters(self):
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_size), 1/sqrt(in_size)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        torch.nn.init.kaiming_uniform_(self.weight, a=5**(1/2))
        if not self.bias is None:
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / fan_in**(1/2) if fan_in > 0 else 0
            torch.nn.init.uniform_(self.bias, -bound, bound)

    def extra_repr(self):
        return "in_size=%d, out_size=%d, period_size=%d, bias=%s" % (
            self.in_size, self.out_size, self.period_size, self.bias is not None
        )
