import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from model import *
#损失
# class SLCrossEntropyLoss(nn.Module):
#     def __init__(self):
#         super(SLCrossEntropyLoss, self).__init__()
#
#     def forward(self, p, target):
#         log_p = F.log_softmax(p, dim=1)  # softmax + log
#         loss = -1 * torch.sum(target * log_p, 1)
#         return loss.mean()
#
#
# class LSCrossEntropyLoss(nn.Module):
#     def __init__(self, label_smooth=None, cls=None):
#         super(LSCrossEntropyLoss, self).__init__()
#         self.label_smooth = label_smooth
#         self.cls = cls
#
#     def forward(self, p, target):
#         if self.label_smooth is not None:
#             log_p = F.log_softmax(p, dim=1)  # softmax + log
#             target = F.one_hot(target, self.cls)  # one-hot
#             target = torch.clamp(target.float(), min=self.label_smooth / (self.cls - 1),
#                                  max=1.0 - self.label_smooth)
#             loss = -1 * torch.sum(target * log_p, 1)
#         else:
#             loss = F.cross_entropy(p, target, reduction='mean', ignore_index=-100)
#         return loss.mean()

class EncoderLayer(nn.Module):
    ''' Compose with two layers '''

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(self, enc_input, slf_attn_mask=None):
        enc_output, enc_slf_attn = self.slf_attn(
            enc_input, enc_input, enc_input, mask=slf_attn_mask)
        enc_output = self.pos_ffn(enc_output)
        return enc_output, enc_slf_attn

class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()

        self.dropout = dropout

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)

        self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5, attn_dropout = self.dropout)

        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)


    def forward(self, q, k, v, mask=None):

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        residual = q
        q = self.layer_norm(q)

        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        # Transpose for attention dot product: b x n x lq x dv
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        
        if mask is not None:
            mask = mask.unsqueeze(1)   # For head axis broadcasting.

        q, attn = self.attention(q, k, v, mask=mask)

        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        q = q.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        q = F.dropout(self.fc(q), self.dropout, training=self.training)
        q += residual

        return q, attn


class PositionwiseFeedForward(nn.Module):
    ''' A two-feed-forward-layer module '''

    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_in, d_hid) # position-wise
        self.w_2 = nn.Linear(d_hid, d_in) # position-wise
        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        self.dropout = dropout

    def forward(self, x):

        residual = x
        x = self.layer_norm(x)

        x = self.w_2(F.relu(self.w_1(x)))
        x = F.dropout(x, self.dropout, training=self.training)
        x += residual

        return x


class PositionalEncoding(nn.Module):

    def __init__(self, d_hid, n_position=200):
        super(PositionalEncoding, self).__init__()

        # Not a parameter
        self.register_buffer('pos_table', self._get_sinusoid_encoding_table(n_position, d_hid))

    def _get_sinusoid_encoding_table(self, n_position, d_hid):
        ''' Sinusoid position encoding table '''
        # TODO: make it with torch instead of numpy

        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

        return torch.FloatTensor(sinusoid_table).unsqueeze(0)

    def forward(self, x):
        return x + self.pos_table[:, :x.size(1)].clone().detach()


class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = attn_dropout

    def forward(self, q, k, v, mask=None):

        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)

        #print(F.softmax(attn, dim=-1))

        attn =  F.dropout(F.softmax(attn, dim=-1), self.dropout, training=self.training)
        output = torch.matmul(attn, v)

        return output, attn

#搭建GAT网络
class Aggregator(nn.Module):
    def __int__(self, batch_size, dim, dropout, act, name = None):
        super(Aggregator, self).__int__()
        self.dropout = dropout
        self.act = act
        self.batch_size = batch_size
        self.dim = dim
    def forward(self):
        pass

class LocalAggregator(nn.Module):
    # def __init__(self, num_node, opt):
    #     super(LocalAggregator, self).__init__()
    #     self.opt = opt
    #     self.pos_len = opt.pos_len
    #     self.num_node = num_node
    #     self.batch_size = opt.batch_size
    #     self.dropout = opt.dropout
    #     self.hidden_size = opt.hiddenSize
    #     self.lr = opt.lr
    #     self.step_size = opt.lr_step
    #     self.gamma = opt.lr_dc
    #     self.alpha = opt.alpha
    #     self.leaky_relu = nn.LeakyReLU(self.alpha)
    #     self.kernel_size = opt.kernel_size
    #     self.label_smooth = opt.smooth
    #
    #     self.tr_layer = TRLayer(opt)
    #     self.embedding = nn.Embedding(self.num_node, self.hidden_size, padding_idx=0)
    #     self.position_embedding = nn.Embedding(self.pos_len, self.hidden_size)
    #     self.w_2 = nn.Parameter(torch.Tensor(self.hidden_size, 1))
    #     self.linear1 = nn.Linear(self.hidden_size, self.hidden_size)
    #     self.linear2 = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
    #     self.linear3 = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
    #     self.linear4 = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
    #     self.b_2 = nn.Parameter(torch.Tensor(self.hidden_size))
    #
    #     self.conv_q = nn.Conv1d(in_channels=self.hidden_size, out_channels=self.hidden_size, kernel_size=self.kernel_size, padding=self.kernel_size//2, groups=self.hidden_size)
    #     self.conv_k = nn.Conv1d(in_channels=self.hidden_size, out_channels=self.hidden_size, kernel_size=self.kernel_size, padding=self.kernel_size//2, groups=self.hidden_size)
    #     self.conv_v = nn.Conv1d(in_channels=self.hidden_size, out_channels=self.hidden_size, kernel_size=self.kernel_size, padding=self.kernel_size//2, groups=self.hidden_size)
    #
    #     self.loss_function = LSCrossEntropyLoss(label_smooth=self.label_smooth, cls=self.num_node-1)
    #     self.sl_loss_function = SLCrossEntropyLoss()
    #     self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
    #     self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=self.step_size, gamma=self.gamma)
    #     self.init_parameters()
    #
    # def init_parameters(self):
    #     stand = 1.0 / math.sqrt(self.hidden_size)
    #
    #
    #     for weight in self.parameters():
    #         weight.data.uniform_(-stand, stand)
    #
    # def get_score(self, intent):
    #     b = self.embedding.weight[1:]
    #     scores = torch.matmul(intent, b.transpose(1, 0))
    #     return scores
    #
    # def forward(self, mask, adj, items, alias_inputs, pos, graph_mask):
    #     items_h = self.embedding(items)
    #     hidden = self.tr_layer(items_h, adj, alias_inputs)#
    #
    #     s_len = hidden.shape[1]
    #     pos_emb = self.position_embedding(pos)
    #     pos_emb = pos_emb.permute(0, 2, 1)
    #     p_q, p_k, p_v = self.conv_q(pos_emb), self.conv_k(pos_emb), self.conv_v(pos_emb)
    #     p_q, p_k, p_v = p_q.permute(0, 2, 1), p_k.permute(0, 2, 1), p_v.permute(0, 2, 1)
    #     p_q, p_k, p_v = self.leaky_relu(p_q), self.leaky_relu(p_k), self.leaky_relu(
    #         p_v)
    #     pos_mask = graph_mask
    #     dk = pos_emb.size()[-1]
    #     p_alpha = p_q.matmul(p_k.transpose(-2, -1)) / math.sqrt(dk)
    #     p_alpha = p_alpha.masked_fill(pos_mask == 0, -9e15)
    #     p_alpha = F.softmax(p_alpha, dim=-1)
    #     pos_emb = p_alpha.matmul(p_v)
    #
    #     nh = hidden + F.dropout(pos_emb, self.dropout)
    #     h_n = nh[torch.arange(mask.shape[0]).long(), torch.sum(mask, 1) - 1]
    #     h_n = h_n.unsqueeze(1).repeat(1, s_len, 1)
    #     mask = mask.float().unsqueeze(-1)
    #     hs = torch.sum(nh * mask, -2) / torch.sum(mask, 1)
    #     hs = hs.unsqueeze(-2).repeat(1, s_len, 1)
    #     nh = self.leaky_relu(self.linear1(nh) + self.linear2(hs) + self.linear3(h_n) + self.linear4(torch.where(hs > h_n, hs, h_n)))
    #     nh = F.dropout(nh, self.dropout)
    #     beta = torch.matmul(nh, self.w_2) + self.b_2
    #     beta = beta * mask
    #     intent = torch.sum(beta * hidden, 1)
    #     return intent

    def __init__(self, dim, alpha, dropout = 0., concat=True):
        super(LocalAggregator, self).__init__()
        self.dim = dim
        self.dropout = dropout
        self.concat = concat

        self.a_0 = nn.Parameter(torch.Tensor(self.dim, 1))
        self.a_1 = nn.Parameter(torch.Tensor(self.dim, 1))
        self.a_2 = nn.Parameter(torch.Tensor(self.dim, 1))
        self.a_3 = nn.Parameter(torch.Tensor(self.dim, 1))
        self.bias = nn.Parameter(torch.Tensor(self.dim))

        self.leakyrelu = nn.LeakyReLU(alpha)

    def forward(self, hidden, adj, mask_item=None):
        h = hidden
        batch_size = h.shape[0]
        N = h.shape[1]

        a_input = (h.repeat(1, 1, N).view(batch_size, N * N, self.dim)
                   * h.repeat(1, N, 1)).view(batch_size, N, N, self.dim)
        e_0 = torch.matmul(a_input, self.a_0)
        e_1 = torch.matmul(a_input, self.a_1)
        e_2 = torch.matmul(a_input, self.a_2)
        e_3 = torch.matmul(a_input, self.a_3)

        e_0 = self.leakyrelu(e_0).squeeze(-1).view(batch_size, N, N)
        e_1 = self.leakyrelu(e_1).squeeze(-1).view(batch_size, N, N)
        e_2 = self.leakyrelu(e_2).squeeze(-1).view(batch_size, N, N)
        e_3 = self.leakyrelu(e_3).squeeze(-1).view(batch_size, N, N)

        mask = -9e15 * torch.ones_like(e_0)
        alpha = torch.where(adj.eq(1), e_0, mask)
        alpha = torch.where(adj.eq(2), e_1, alpha)
        alpha = torch.where(adj.eq(3), e_2, alpha)
        alpha = torch.where(adj.eq(4), e_3, alpha)
        alpha = torch.softmax(alpha, dim=-1)

        output = torch.matmul(alpha, h)

        return output


def trans_to_cuda(variable):
    if torch.cuda.is_available():
        return variable.cuda()
    else:
        return variable

