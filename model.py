import torch
import numpy as np, torch.nn as nn, torch.nn.functional as F
from utils import cal_accuracy


class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, **kwargs):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k)
        self.w_ks = nn.Linear(d_model, n_head * d_k)
        self.w_vs = nn.Linear(d_model, n_head * d_v)
        nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_v)))

        self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5),
                                                   dropout=kwargs['dr_sdp'])
        self.layer_norm = nn.LayerNorm(d_model)
        self.layer_norm_in = nn.LayerNorm((3, n_head * d_v))

        self.fc = nn.Linear(n_head * d_v, d_model)
        nn.init.xavier_normal_(self.fc.weight)

        self.dropout = nn.Dropout(kwargs['dr_mha'])

    def forward(self, q, k, v, mask=None):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head

        sz_b, len_q, _ = q.size()
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()

        residual = q

        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        q = q.transpose(1, 2).contiguous().view(-1, len_q, d_k)  # (n*b) x lq x dk
        k = k.transpose(1, 2).contiguous().view(-1, len_k, d_k)  # (n*b) x lk x dk
        v = v.transpose(1, 2).contiguous().view(-1, len_v, d_v)  # (n*b) x lv x dv

        output = self.attention(q, k, v)

        output = output.view(sz_b, n_head, len_q, d_v)
        output = output.transpose(1, 2).contiguous().view(sz_b, len_q, -1)  # b x lq x (n*dv)

        output = self.dropout(self.fc(output))
        output = self.layer_norm(output + residual)

        return output


class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, dropout=0.2):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v):
        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn / self.temperature

        attn = self.softmax(attn)
        attn = self.dropout(attn)
        output = torch.bmm(attn, v)

        return output


class PositionwiseFeedForwardUseConv(nn.Module):
    def __init__(self, d_in, d_hid, dropout=0.3, decode_method='twomult'):
        super(PositionwiseFeedForwardUseConv, self).__init__()
        self.w_1 = nn.Conv1d(d_in, d_hid, 1)
        nn.init.kaiming_uniform_(self.w_1.weight, mode='fan_out', nonlinearity='relu')
        self.w_2 = nn.Conv1d(d_hid, d_in, 1)
        nn.init.kaiming_uniform_(self.w_2.weight, mode='fan_in', nonlinearity='relu')
        self.layer_norm = nn.LayerNorm(d_in)
        self.dropout = nn.Dropout(dropout)
        self.decode_method = decode_method

    def forward(self, x):
        residual = x
        output = x.transpose(1, 2)
        output = self.w_2(F.relu(self.w_1(output)))
        output = output.transpose(1, 2)
        output = self.dropout(output)
        if self.decode_method == 'tucker':
            output = output + residual
        else:
            output = self.layer_norm(output + residual)
        return output


class EncoderLayer(nn.Module):
    def __init__(self, d_model, d_inner, n_head, d_k, d_v, **kwargs):
        super(EncoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(
            n_head, d_model, d_k, d_v, **kwargs)
        self.pos_ffn = PositionwiseFeedForwardUseConv(
            d_model, d_inner, dropout=kwargs['dr_pff'], 
            decode_method=kwargs['decoder'])

    def forward(self, enc_input):
        enc_output = self.slf_attn(enc_input, enc_input, enc_input)
        enc_output = self.pos_ffn(enc_output)

        return enc_output


class Encoder(nn.Module):
    def __init__(self, d_input, n_layers, n_head, d_k, d_v,
                 d_model, d_inner, **kwargs):
        super(Encoder, self).__init__()
        # parameters
        self.d_input = d_input
        self.n_layers = n_layers
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v
        self.d_model = d_model
        self.d_inner = d_inner
        self.linear_in = nn.Linear(d_input, d_model)
        self.layer_norm_in = nn.LayerNorm(d_model)
        self.bn = nn.BatchNorm2d(1)
        self.dropout = nn.Dropout(kwargs['dr_enc'])

        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v, **kwargs)
            for _ in range(n_layers)])

    def forward(self, edges):
        enc_output = self.dropout(edges)

        for enc_layer in self.layer_stack:
            enc_output = enc_layer(enc_output)

        return enc_output


class TuckER(nn.Module):
    def __init__(self, d1, d2, **kwargs):
        super(TuckER, self).__init__()
        self.W = torch.nn.Parameter(
            torch.tensor(np.random.uniform(-0.05, 0.05, (d2, d1, d1)), dtype=torch.float, requires_grad=True)
        )
        self.bn0 = torch.nn.BatchNorm1d(d1)

    def forward(self, e1, r):
        x = self.bn0(e1)
        x = x.view(-1, 1, e1.size(1))

        W_mat = torch.mm(r, self.W.view(r.size(1), -1))
        W_mat = W_mat.view(-1, e1.size(1), e1.size(1))

        x = torch.bmm(x, W_mat)
        x = x.view(-1, e1.size(1))

        return x


class SAttLE(nn.Module):

    def __init__(
            self, num_nodes,
            num_rels, n_layers,
            d_embd, d_k, d_v,
            d_model, d_inner, n_head,
            label_smoothing=0.1,
            **kwargs
    ):
        super(SAttLE, self).__init__()

        self.encoder = Encoder(d_embd, n_layers, n_head, d_k, d_v, d_model, d_inner, **kwargs)
        self.register_parameter('b', nn.Parameter(torch.zeros(num_nodes)))
        self.num_nodes, self.num_rels = num_nodes, num_rels
        self.d_embd = d_embd
        self.init_parameters()
        if kwargs['decoder'] == 'tucker':
            print('decoding with tucker mode!')
            self.decode_method = 'tucker'
            self.tucker = TuckER(d_embd, d_embd)
        else:
            self.decode_method = 'twomult'
        self.ent_bn = nn.BatchNorm1d(d_embd)
        self.rel_bn = nn.BatchNorm1d(d_embd)
        self.label_smoothing = label_smoothing

    def init_parameters(self):
        self.tr_ent_embedding = nn.Parameter(torch.Tensor(self.num_nodes, self.d_embd))
        self.rel_embedding = nn.Parameter(torch.Tensor(2 * self.num_rels, self.d_embd))
        nn.init.xavier_normal_(self.tr_ent_embedding.data)
        nn.init.xavier_normal_(self.rel_embedding.data)

    def cal_loss(self, scores, edges, label_smooth=True):
        labels = torch.zeros_like((scores), device='cuda')
        labels.scatter_(1, edges[:, 2][:, None], 1.)
        if self.label_smoothing:
            labels = ((1.0 - self.label_smoothing) * labels) + (1.0 / labels.size(1))
        pred_loss = F.binary_cross_entropy_with_logits(scores, labels)

        return pred_loss

    def cal_score(self, edges, mode='train'):
        h = self.tr_ent_embedding[edges[:, 0]]
        r = self.rel_embedding[edges[:, 1]]
        h = self.ent_bn(h)[:, None]
        r = self.rel_bn(r)[:, None]
        feat_edges = torch.hstack((h, r))

        embd_edges = self.encoder(feat_edges)
        if self.decode_method == 'tucker':
            src_edges = self.tucker(embd_edges[:, 0, :], embd_edges[:, 1, :])
        else:
            src_edges = embd_edges[:, 1, :]
        scores = torch.mm(src_edges, self.tr_ent_embedding.transpose(0, 1))
        scores += self.b.expand_as(scores)
        return scores

    def forward(self, feat_edges):
        scores = self.cal_score(feat_edges, mode='train')
        return scores

    def predict(self, edges):
        labels = edges[:, 2]
        edges = edges[0][None, :]
        scores = self.cal_score(edges, mode='test')
        scores = scores[:, labels.view(-1)].view(-1)
        scores = torch.sigmoid(scores)
        acc = 0.0
        return scores, acc
