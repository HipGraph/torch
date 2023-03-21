import torch
import torch.nn.functional as F

from .. import util as pyt_util

import Utility as util


class __Linear__(torch.nn.Module):

    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            torch.nn.init.uniform_(self.bias, -bound, bound)


class __RNNCell__(torch.nn.Module):

    def reset_parameters(self) -> None:
        stdv = 1.0 / math.sqrt(self.hidden_size) if self.hidden_size > 0 else 0
        for weight in self.parameters():
            torch.nn.init.uniform_(weight, -stdv, stdv)


class kLinear(__Linear__):

    debug = 0

    def __init__(self, in_size, out_size, bias=True, k=1):
        super(kLinear, self).__init__()
        self.weight = torch.nn.Parameter(torch.empty((k, out_size, in_size)))
        if bias:
            self.bias = torch.nn.Parameter(torch.empty((out_size,)))
        else:
            self.bias = self.register_parameter("bias", None)
        self.in_size = in_size
        self.out_size = out_size
        self.k = k

    def forward(self, x):
        w, b = self.weight, self.bias
        # x.shape=(?, K, |V|, I)
        # w.shape=(K, O, I)
        # b.shape=(O,)
        if self.debug:
            print("x =", x.shape)
            print("w =", w.shape)
            if not b is None:
                print("b =", b.shape)
        x = torch.einsum("...KVI,KOI->...KVO", x, w) # shape=(?, K, |V|, O)
        if not b is None:
            x = x + b # shape=(?, |V|, O)
        return x


class gLinear(__Linear__):

    debug = 0

    def __init__(self, in_size, out_size, bias=True, n_groups=1):
        super(gLinear, self).__init__()
        self.weight = torch.nn.Parameter(torch.empty((n_groups, out_size, in_size)))
        if bias:
            self.bias = torch.nn.Parameter(torch.empty((n_groups, out_size)))
        else:
            self.bias = self.register_parameter("bias", None)
        self.in_size = in_size
        self.out_size = out_size
        self.n_groups = n_groups

    def forward(self, x, group_index):
        w = torch.index_select(self.weight, 0, group_index)
        b = None if self.bias is None else torch.index_select(self.bias, 0, group_index)
        # x.shape=(?, |V|, I)
        # group_index.shape=(?, |V|) E {0, 1, ..., n_groups-1}
        # w.shape=(|V|, O, I)
        # b.shape=(|V|, O)
        if self.debug:
            print("x =", x.shape)
            print("group_index =", group_index.shape)
            print("w =", w.shape)
            if not b is None:
                print("b =", b.shape)
        x = torch.einsum("...VI,VOI->...VO", x, w)
        if not b is None:
            x = x + b
        return x


class kgLinear(__Linear__):

    debug = 0

    def __init__(self, in_size, out_size, bias=True, k=1, n_groups=1):
        super(kgLinear, self).__init__()
        self.weight = torch.nn.Parameter(torch.empty((k, n_groups, out_size, in_size)))
        if bias:
            self.bias = torch.nn.Parameter(torch.empty((n_groups, out_size)))
        else:
            self.bias = self.register_parameter("bias", None)
        self.in_size = in_size
        self.out_size = out_size
        self.k = k
        self.n_groups = n_groups

    def forward(self, x, group_index):
        w = torch.index_select(self.weight, 1, group_index)
        b = None if self.bias is None else torch.index_select(self.bias, 0, group_index)
        # x.shape=(?, K, |V|, I)
        # group_index.shape=(?, |V|) E {0, 1, ..., n_groups-1}
        # w.shape=(K, |V|, O, I)
        # b.shape=(|V|, O)
        if self.debug:
            print("x =", x.shape)
            print("w =", w.shape)
            if not b is None:
                print("b =", b.shape)
        x = torch.einsum("...KVI,KVOI->...KVO", x, w) # shape=(?, |V|, O)
        if not b is None:
            x = x + b # shape=(?, |V|, O)
        return x


class vwLinear(__Linear__):

    debug = 0

    def __init__(self, in_size, out_size, bias=True, embed_size=10):
        super(vwLinear, self).__init__()
        self.weight = torch.nn.Parameter(torch.empty((embed_size, out_size, in_size)))
        if bias:
            self.bias = torch.nn.Parameter(torch.empty((embed_size, out_size)))
        else:
            self.bias = self.register_parameter("bias", None)
        self.in_size = in_size
        self.out_size = out_size
        self.embed_size = embed_size

    def forward(self, x, embedding):
        w = torch.einsum("...D,DOI->...OI", embedding, self.weight)
        b = None if self.bias is None else torch.einsum("...D,DO->...O", embedding, self.bias)
        # x.shape=(?, |V|, I)
        # embedding.shape=(?, |V|, D)
        # w.shape=(|V|, O, I)
        # b.shape=(|V|, O)
        if self.debug:
            print("x =", x.shape)
            print("embedding =", embedding.shape)
            print("w =", w.shape)
            if not b is None:
                print("b =", b.shape)
        x = torch.einsum("...VI,VOI->...VO", x, w) # shape=(?, |V|, O)
        if not b is None:
            x = x + b # shape=(?, |V|, O)
        return x


class kvwLinear(__Linear__):

    debug = 0

    def __init__(self, in_size, out_size, bias=True, k=1, embed_size=10):
        super(kvwLinear, self).__init__()
        self.weight = torch.nn.Parameter(torch.empty((k, embed_size, out_size, in_size)))
        if bias:
            self.bias = torch.nn.Parameter(torch.empty((embed_size, out_size)))
        else:
            self.bias = self.register_parameter("bias", None)
        self.in_size = in_size
        self.out_size = out_size
        self.k = k
        self.embed_size = embed_size

    def forward(self, x, embedding):
        w = torch.einsum("...VD,KDOI->...KVOI", embedding, self.weight)
        b = None if self.bias is None else torch.einsum("...D,DO->...O", embedding, self.bias)
        # x.shape=(?, K, |V|, I)
        # embedding.shape=(?, |V|, D)
        # w.shape=(K, |V|, O, I)
        # b.shape=(|V|, O)
        if self.debug:
            print("x =", x.shape)
            print("embedding =", embedding.shape)
            print("w =", w.shape)
            if not b is None:
                print("b =", b.shape)
        x = torch.einsum("...KVI,KVOI->...KVO", x, w) # shape=(?, |V|, O)
        if not b is None:
            x = x + b # shape=(?, |V|, O)
        return x


class chebvwLinear(__Linear__):

    debug = 0

    def __init__(self, in_size, out_size, bias=True, cheb_k=2, embed_size=10):
        super(chebvwLinear, self).__init__()
        print("chebvwLinear :", in_size, out_size, bias, cheb_k, embed_size)
        arch = 1
        if arch == 1:
            self.weight = torch.nn.Parameter(torch.empty((embed_size, cheb_k, in_size, out_size)))
        else:
            self.weight = torch.nn.Parameter(torch.empty((embed_size, cheb_k, out_size, in_size)))
        if bias:
            self.bias = torch.nn.Parameter(torch.empty((embed_size, out_size)))
        else:
            self.bias = self.register_parameter("bias", None)
        self.in_size = in_size
        self.out_size = out_size
        self.cheb_k = cheb_k
        self.embed_size = embed_size
        self.arch = arch

    def forward(self, x, A, embedding):
        if self.arch == 1:
            w = torch.einsum("nd,dkio->nkio", embedding, self.weight)
            b = None if self.bias is None else torch.einsum("nd,do->no", embedding, self.bias)
        elif self.arch == 2:
            w = torch.einsum("nd,dkoi->nkoi", embedding, self.weight)
            b = None if self.bias is None else torch.einsum("nd,do->no", embedding, self.bias)
        else:
            w = torch.einsum("...D,DKOI->...KOI", embedding, self.weight)
            b = None if self.bias is None else torch.einsum("...D,DO->...O", embedding, self.bias)
        # x.shape=(?, |V|, I)
        # A.shape=(K, |V|, |V|)
        # embedding.shape=(?, |V|, D)
        # w.shape=(K, |V|, O, I)
        # b.shape=(|V|, O)
        if self.debug:
            print("x =", x.shape)
            if self.debug > 2:
                print(x)
            print("A =", A.shape)
            if self.debug > 2:
                print(A)
            print("w =", w.shape)
            if self.debug > 2:
                print(w)
            if not b is None:
                print("b =", b.shape)
                if self.debug > 2:
                    print(b)
        V = A.shape[-2]
        As = [torch.eye(V).to(x.device)]
        if self.cheb_k > 1:
            As.append(A)
        for k in range(2, self.cheb_k):
            As.append(torch.matmul(2 * A, As[-1]) - As[-2])
        A = torch.stack(As, 0)
        if self.arch == 1:
            x_g = torch.einsum("knm,bmc->bknc", A, x)
            x_g = x_g.permute(0, 2, 1, 3)
            x = torch.einsum("bnki,nkio->bno", x_g, w)
        elif self.arch == 2:
            x_g = torch.einsum("knm,bmc->bknc", A, x)
            x_g = x_g.permute(0, 2, 1, 3)
            x = torch.einsum("bnki,nkoi->bno", x_g, w)
        else:
            x_g = torch.einsum("KVM,...MI->...KVI", A, x)
            x = torch.einsum("...KVI,VKOI->...VO", x_g, w)
        if not b is None:
            x = x + b
        if self.debug:
            print("A =", A.shape)
            if self.debug > 2:
                print(A)
            print("x =", x.shape)
            if self.debug > 2:
                print(x)
                input()
        return x


class gGRUCell(__RNNCell__):

    debug = 0

    def __init__(self, in_size, out_size, **kwargs):
        super(gGRUCell, self).__init__()
        n_groups = kwargs.get("n_groups", 1)
        xs_size = kwargs.get("xs_size", 0)
        arch = kwargs.get("arch", 0)
        # weights for input x
        self.weight_ih = torch.nn.Parameter(torch.ones((3, n_groups, out_size, in_size)))
        self.bias_ih = torch.nn.Parameter(torch.ones((3, n_groups, out_size)))
        # weights for input x_s
        if arch > 0:
            self.weight_sh = torch.nn.Parameter(torch.ones((3, n_groups, out_size, xs_size)))
            self.bias_sh = torch.nn.Parameter(torch.ones((3, n_groups, out_size)))
        # weights for input h
        self.weight_hh = torch.nn.Parameter(torch.ones((3, n_groups, out_size, out_size)))
        self.bias_hh = torch.nn.Parameter(torch.ones((3, n_groups, out_size)))
        # vars
        self.in_size = in_size
        self.out_size = out_size
        self.n_groups = n_groups
        self.xs_size = xs_size
        self.arch = arch

    def forward(self, x, x_s=None, group_index=None, hx=None, **kwargs):
        node_dim = kwargs.get("node_dim", -2)
        if self.debug:
            print(util.make_msg_block("gGRUCell Forward"))
            print("x =", x.shape)
        if self.debug and not x_s is None:
            print("x_s =", x_s.shape)
        if self.debug and not group_index is None:
            print("group_index =", group_index.shape)
        if self.debug and not hx is None:
            print("hx =", hx.shape)
        if 0:
            if group_index is None:
                n_nodes = x.shape[node_dim]
                group_index = torch.arange(n_nodes, dtype=torch.long, device=x.device) % self.n_groups
        if hx is None:
            zeros = torch.zeros(x.shape[:2]+(self.out_size,), dtype=x.dtype, device=x.device)
            hx = zeros
        h = hx
        # unpack weights for input x
        w_ih = torch.index_select(self.weight_ih, 1, group_index)
        b_ih = torch.index_select(self.bias_ih, 1, group_index)
        w_ir, w_iz, w_in = w_ih
        b_ir, b_iz, b_in = b_ih
        if self.debug:
            print("w_ih =", w_ih.shape)
            print("b_ih =", b_ih.shape)
        # unpack weights for input x_s
        if self.arch > 0:
            w_sh = torch.index_select(self.weight_sh, 1, group_index)
            b_sh = torch.index_select(self.bias_sh, 1, group_index)
            w_sr, w_sz, w_sn = w_sh
            b_sr, b_sz, b_sn = b_sh
            if self.debug:
                print("w_sh =", w_sh.shape)
                print("b_sh =", b_sh.shape)
        # unpack weights for input h
        w_hh = torch.index_select(self.weight_hh, 1, group_index)
        b_hh = torch.index_select(self.bias_hh, 1, group_index)
        w_hr, w_hz, w_hn = w_hh
        b_hr, b_hz, b_hn = b_hh
        if self.debug:
            print("w_hh =", w_hh.shape)
            print("b_hh =", b_hh.shape)
        # Reset Gate Forward
        if not x_s is None and self.arch & 1:
            r = torch.sigmoid(
                self.linear_x(x, w_ir, b_ir) + self.linear_x(x_s, w_sr, b_sr)[None,:,:] + self.linear_h(h, w_hr, b_hr)
            )
        else:
            r = torch.sigmoid(self.linear_x(x, w_ir, b_ir) + self.linear_h(h, w_hr, b_hr))
        # Update Gate Forward
        if not x_s is None and self.arch & 2:
            z = torch.sigmoid(
                self.linear_x(x, w_iz, b_iz) + self.linear_x(x_s, w_sz, b_sz)[None,:,:] + self.linear_h(h, w_hz, b_hz)
            )
        else:
            z = torch.sigmoid(self.linear_x(x, w_iz, b_iz) + self.linear_h(h, w_hz, b_hz))
        # New Gate Forward
        if not x_s is None and self.arch & 4:
            n = torch.tanh(
                self.linear_x(x, w_in, b_in) + self.linear_x(x_s, w_sn, b_sn)[None,:,:] + r * self.linear_h(h, w_hn, b_hn)
            )
        else:
            n = torch.tanh(self.linear_x(x, w_in, b_in) + r * self.linear_h(h, w_hn, b_hn))
        if self.debug:
            print("r =", r.shape)
            print("z =", z.shape)
            print("n =", n.shape)
        _h = (1 - z) * n + z * h
        if self.debug:
            print("hx =", _h.shape)
        return _h

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
            print("### gGRUCell.linear_x() Forward ###")
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
            print("### gGRUCell.linear_h() Forward ###")
            print("h =", h.shape)
            print("w =", w.shape)
            print("b =", b.shape)
        return torch.einsum("NVO,VOO->NVO", h, w) + b[None,:,:]

    def add(self, x, x_s):
        return (x.view(-1, x_s.shape[0], x.shape[-1]) + x_s).view(x.shape)
        return torch.reshape(torch.reshape(x, (-1, x_s.shape[0], x.shape[-1])) + x_s, x.shape)


class gLSTMCell(__RNNCell__):

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
            print(util.make_msg_block("gLSTMCell Forward"))
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
                self.linear_x(x, w_ii, b_ii) + self.linear_x(x_s, w_si, b_si)[None,:,:] + self.linear_h(h, w_hi, b_hi)
            )
        else:
            i = torch.sigmoid(self.linear_x(x, w_ii, b_ii) + self.linear_h(h, w_hi, b_hi))
        # Forget Gate Forward
        if not x_s is None and self.arch & 2:
            f = torch.sigmoid(
                self.linear_x(x, w_if, b_if) + self.linear_x(x_s, w_sf, b_sf)[None,:,:] + self.linear_h(h, w_hf, b_hf)
            )
        else:
            f = torch.sigmoid(self.linear_x(x, w_if, b_if) + self.linear_h(h, w_hf, b_hf))
        # Cell Gate Forward
        if not x_s is None and self.arch & 4:
            g = torch.tanh(
                self.linear_x(x, w_ig, b_ig) + self.linear_x(x_s, w_sg, b_sg)[None,:,:] + self.linear_h(h, w_hg, b_hg)
            )
        else:
            g = torch.tanh(self.linear_x(x, w_ig, b_ig) + self.linear_h(h, w_hg, b_hg))
        # Ouput Gate Forward
        if not x_s is None and self.arch & 8:
            o = torch.sigmoid(
                self.linear_x(x, w_io, b_io) + self.linear_x(x_s, w_so, b_so)[None,:,:] + self.linear_h(h, w_ho, b_ho)
            )
        else:
            o = torch.sigmoid(self.linear_x(x, w_io, b_io) + self.linear_h(h, w_ho, b_ho))
        if self.debug:
            print("i =", i.shape)
            print("f =", f.shape)
            print("g =", g.shape)
            print("o =", o.shape)
        _c = f * c + i * g
        _h = o * torch.tanh(_c)
        if self.debug:
            print("hx =", (_h.shape, _c.shape))
        return (_h, _c)

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
            print("### gLSTMCell.linear_x() Forward ###")
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
            print("### gLSTMCell.linear_h() Forward ###")
            print("h =", h.shape)
            print("w =", w.shape)
            print("b =", b.shape)
        return torch.einsum("NVO,VOO->NVO", h, w) + b[None,:,:]

    def add(self, x, x_s):
        return (x.view(-1, x_s.shape[0], x.shape[-1]) + x_s).view(x.shape)
        return torch.reshape(torch.reshape(x, (-1, x_s.shape[0], x.shape[-1])) + x_s, x.shape)


class mLSTMCell(__RNNCell__):

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

    def forward(self, x, x_s, hx=None):
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
