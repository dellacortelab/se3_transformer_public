import torch
from torch import nn
import numpy as np
from dgl import DGLGraph
import dgl

from se3_transformer.external import *


class PointCloud:
    '''Represents a point cloud in R^3. This class calculates and stores relevant
    geometric information such as the vectors, distances, directions, and
    spherical harmonics of the vectors.'''
    def __init__(self, pos=None, graph=None, cutoff=8.0, J_max=4):
        if graph is None:
            edges = self._find_edges_(pos, cutoff)
            self.graph = dgl.graph(edges)
            self.graph.ndata['pos'] = pos
        else:
            self.graph = graph
        self._calc_edge_info_(J_max)
        self.w_j = dict()

    def _find_edges_(self, pos, cutoff):
        # Use positions to create graph. Need to improve! Currently O(n^2)
        vec_mat = pos[:,None,:]-pos[None,:,:]
        dist_mat = torch.sqrt(torch.sum((vec_mat)**2,axis=-1))
        u = []
        v = []
        for j in range(len(pos)):
            for i in range(j):
                if dist_mat[i,j] < cutoff:
                    u.append(i)
                    v.append(j)
        u, v = torch.tensor(u+v,dtype=torch.long), torch.tensor(v+u,dtype=torch.long)
        return (u,v)

    def _calc_edge_info_(self, J_max):
        # Calculate and store position and angle information
        u,v = self.graph.edges()[0], self.graph.edges()[1]
        pos = self.graph.ndata['pos']
        vec = pos[u]-pos[v]
        self.graph.edata['vec'] = vec
        r_ij = get_spherical_from_cartesian_torch(vec)
        self.graph.edata['r_ij'] = r_ij
        self.Y = precompute_sh(r_ij, J_max)
        self.graph.edata['dist'] = torch.norm(vec, dim=1)
 
    def get_sh(self, J):
        # Returns spherical harmonic of order J.
        if not J in self.Y.keys(): # If J <= J_max this is false.
            r_ij = self.graph.edata['r_ij']
            Y_new = precompute_sh(r_ij,J)
            for key in Y_new.keys():
                if not key in self.Y.keys():
                    self.Y[key] = Y_new[key].to(torch.float32).to(self.graph.device)
        return self.Y[J]

    def get_w_j(self, l, k):
        # Returns basis kernel
        if not (l,k) in self.w_j.keys():
            num_edgs = self.graph.number_of_edges()
            w_j_list = []
            for j, J in enumerate(range(abs(k-l), k+l+1)):
                Y_J = self.get_sh(J)
                Q_J = basis_transformation_Q_J(J, k, l).to(torch.float32).to(self.graph.device)
                w_j = torch.matmul(Y_J,Q_J.T).reshape(num_edgs, 2*l+1, 2*k+1)
                w_j_list.append(w_j.unsqueeze(0))
            w_j = torch.cat(w_j_list, dim=0)
            self.w_j[(l,k)] = w_j.transpose(0,1)
        return self.w_j[(l,k)]

    def to(self, dev):
        self.graph = self.graph.to(dev)
        for key, value in self.w_j.items():
            self.w_j[key] = value.to(dev)
        return self

# Dictionary for indices
# e: edges
# o: c_out
# i: c_in
# l: output tensor representation
# k: input tensor representation
# j: hidden tensor representation

class WLayer(nn.Module):
    def __init__(self, k, l, c_in=1, c_out=1, d_max=1):
        super(WLayer, self).__init__()
        self.k = k
        self.l = l
        self.c_in = c_in
        self.c_out = c_out
        self.d_max = d_max

        J_size = k+l-abs(k-l)+1
        self.J_size = J_size
        r_size = J_size * c_out * c_in

        self.radial = nn.Sequential(
                nn.Linear(c_in+1, 16),
                nn.LayerNorm(16),
                nn.ReLU(),
                nn.Linear(16, 16),
                nn.LayerNorm(16),
                nn.ReLU(),
                nn.Linear(16, r_size)
        )
        nn.init.xavier_normal_(self.radial[0].weight)
        nn.init.xavier_normal_(self.radial[3].weight)
        nn.init.xavier_normal_(self.radial[6].weight)
    
    def prep_input(self, pc, f):
        u, v = pc.graph.edges()
        zero = torch.sum(f[0][u]*f[0][v], dim=2)
        dist = pc.graph.edata['dist']
        vector = torch.cat((zero, dist.unsqueeze(-1)), dim=1)
        return vector

    def forward(self, pc, f):
        w_j = pc.get_w_j(self.l, self.k)
        input_vector = self.prep_input(pc, f)

        size = (pc.graph.number_of_edges(), self.J_size, self.c_out, self.c_in)
        R = self.radial(input_vector).view(*size)
        w = torch.einsum('ejoi,ejlk->eoilk', R, w_j)
        return w

class AttnBlock(nn.Module):
    def __init__(self, d_in, d_out, c_in=1, c_out=1, rdc=torch.sum):
        super(AttnBlock, self).__init__()
        self.d_in = d_in
        self.d_out = d_out

        self.c_in = c_in
        self.c_out = c_out

        self.rdc = rdc

        self.wq = nn.Parameter(torch.randn(d_in+1, c_out, c_in))
        nn.init.xavier_normal_(self.wq)

        self.wk_layers = nn.ModuleList(
            [
            nn.ModuleList(
                [
                WLayer(k, l, c_in, c_out, d_max=d_in)
                for k in range(d_in+1)
                ]
            )
            for l in range(d_out+1)
            ]
        )

    def forward(self, pc, f):
        pc.graph.ndata['q'] = self.calc_q(pc, f)

        for l in range(self.d_out+1):
            for k in range(self.d_in+1):
                pc.graph.edata[(k,l)] = self.wk_layers[l][k](pc, f)

        for key, value in f.items():
            pc.graph.ndata[key] = value
        pc.graph.update_all(self.attn_msg, self.attn_rdc)
        lse = pc.graph.ndata['lse']
        dot = pc.graph.edata['dot']
        a = torch.exp(dot-lse[pc.graph.edges()[1]])
        return a

    def attn_msg(self, edges):
        k = self.calc_k(edges)
        q = edges.dst['q']
        dot = torch.einsum('eol,eol->e',q,k)
        edges.data['dot'] = dot
        return {'dot': dot}

    def attn_rdc(self, nodes):
        # does sum over j'
        dot = nodes.mailbox['dot']
        lse = torch.logsumexp(dot, dim=1)
        return {'lse': lse}

    def calc_q(self, pc, f):
        ql = []
        for k in range(min(self.d_in,self.d_out)+1):
            sum = torch.einsum('oi,nik->nok',self.wq[k],f[k])
            ql.append(sum)
        q = torch.cat(ql, dim=2)
        return q

    def calc_k(self, edges):
        kl = []
        for l in range(min(self.d_in,self.d_out)+1):
            wks = []
            for k in range(self.d_in+1):
                wk = torch.einsum(
                         'eoilk,eik->eol',
                         edges.data[(k,l)],
                         edges.dst[k]
                )
                wks.append(wk.unsqueeze(3))
            stack = torch.cat(wks, dim=3)
            s = torch.sum(stack, dim=3)
            kl.append(s)
        k = torch.cat(kl, dim=2)
        return k

class GConv(nn.Module):
    def __init__(self, d_in, d_out, c_in=1, c_out=1, attention=True, self_interaction=True, rdc=torch.sum):
        super(GConv, self).__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.c_in = c_in
        self.c_out = c_out
        self.attn = attention
        self.si = self_interaction
        self.rdc = rdc

        self.wv_layers = nn.ModuleList(
                [
                nn.ModuleList(
                        [
                        WLayer(k, l, c_in=c_in, c_out=c_out, d_max=d_in)
                        for l in range(d_out+1)
                        ]
                )
                for k in range(d_in+1)
                ]
        )

        if self.attn:
            self.block = AttnBlock(d_in, d_out, c_in=c_in, c_out=c_out)

        if self.si:
            si_nets = []
            for _ in range(min(d_in, d_out)+1):
                si_net = nn.Sequential(
                        nn.Linear(c_in**2, c_in*c_out),
                        nn.LayerNorm(c_in*c_out),
                        nn.ReLU(),
                        nn.Linear(c_in*c_out, c_in*c_out),
                        nn.LayerNorm(c_in*c_out),
                        nn.ReLU(),
                        nn.Linear(c_in*c_out, c_in*c_out)
                )
                nn.init.xavier_normal_(si_net[0].weight)
                nn.init.xavier_normal_(si_net[3].weight)
                nn.init.xavier_normal_(si_net[6].weight)
                si_nets.append(si_net)
      
            self.si_nets = nn.ModuleList(si_nets)

    def forward(self, pc, f):
        if self.attn:
            pc.graph.edata['a'] = self.block(pc, f)
        for k in range(self.d_in+1):
            pc.graph.ndata[k] = f[k]
            for l in range(self.d_out+1):
                pc.graph.edata[(k,l)] = self.wv_layers[k][l](pc, f)
        pc.graph.update_all(self.msg_func, self.rdc_func)
        f_out = dict()
        for l in range(self.d_out+1):
            f_out[l] = pc.graph.ndata[l]
        if self.si:
            si = self.calc_si(pc, f)
            for l in si.keys():
                f_out[l] += si[l]
        return f_out

    def msg_func(self, edges):
        vls = dict()
        for l in range(self.d_out+1):
            vl = self.calc_vl(edges, l)
            vls[l] = vl
        return vls

    def rdc_func(self, nodes):
        f = dict()
        for key, value in nodes.mailbox.items():
            f[key] = self.rdc(value, dim=1)
        return f

    def calc_vl(self, edges, l):
        if self.attn:
            a = edges.data['a']
        vlks = []
        for k in range(self.d_in+1):
            wk = edges.data[(k,l)]
            f = edges.src[k]
            if self.attn:
                vlk = torch.einsum('e,eoilk,eik->eol', a, wk, f)
            else:
                vlk = torch.einsum('eoilk,eik->eol', wk, f)
            vlks.append(vlk.unsqueeze(3))
        vlk = torch.cat(vlks, dim=3)
        vl = torch.sum(vlk, dim=3)
        return vl

    def calc_si(self, pc, f):
        si = dict()
        num_nodes = pc.graph.num_nodes()
        c_in = self.c_in
        c_out = self.c_out
        size_in = (num_nodes, c_in*c_in)
        size_out = (num_nodes, c_out, c_in)
        for l in range(min(self.d_in,self.d_out)+1):
            f_l = f[l]
            inner = torch.einsum('ncl,ndl->ncd',f_l,f_l)
            inner = inner.view(*size_in)
            si_w = self.si_nets[l](inner).view(*size_out)
            si[l] = torch.einsum('noi,nil->nol', si_w, f[l])
        return si


class GLinear(nn.Module):
    def __init__(self, d_in, c_in, c_out):
        super(GLinear, self).__init__()
        self.d_in = d_in
        self.c_in = c_in
        self.c_out = c_out
        self.ws = nn.Parameter(torch.randn(d_in+1, c_out, c_in))
        nn.init.xavier_normal_(self.ws)

    def forward(self, f):
        out = dict()
        for i in range(self.d_in+1):
            out[i] = torch.einsum('oi,nim->nom', self.ws[i], f[i])
        return out

class MultiHead(nn.Module):
    def __init__(self, d_in=1, d_out=1, c_in=1, c_hid=None, c_out=1, heads=1, attention=True, self_interaction=True, rdc=torch.sum):
        super(MultiHead, self).__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.c_in = c_in
        if c_hid is None:
            c_hid = c_in
        self.c_hid = c_hid
        self.c_out = c_out
        self.heads = heads
        self.rdc = rdc
        layers = [
                 GConv(d_in, d_out, c_in, c_hid, attention, self_interaction, rdc)
                 for _ in range(heads)
        ]
        self.layers = nn.ModuleList(layers)
        self.linear = GLinear(self.d_out, heads*c_hid, c_out)

    def forward(self, pc, f):
        outs = [self.layers[i](pc, f) for i in range(self.heads)]
        out = self.merge_fs(outs)
        out = self.linear(out)
        return out

    # Put chunks back together
    @staticmethod
    def merge_fs(fs):
        f = dict()
        for l in fs[0].keys():
            f_ls = [f_i[l] for f_i in fs]
            f[l] = torch.cat(f_ls, dim=1)
        return f


class GNonlinear(nn.Module):
    def __init__(self, d_in, c_in=1, c_out=1):
        super(GNonlinear, self).__init__()
        self.d_in = d_in
        self.c_in = c_in
        self.c_out = c_out
        self.lns = nn.ModuleList([nn.LayerNorm(c_in) for _ in range(d_in+1)])
        self.relu = nn.ReLU()

    def forward(self, pc, f):
        out = dict()
        for l in range(self.d_in+1):
            f_l = f[l]
            norm = torch.norm(f[l],dim=2,keepdim=True)
            out[l] = self.relu(self.lns[l](norm[...,0])).unsqueeze(-1) * (f_l/norm)
        return out

