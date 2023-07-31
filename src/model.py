#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    if activation == "leakyrelu":
        return F.leaky_relu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


class BasicConv1d(nn.Module):

    def __init__(self, in_channels, out_channels, dropout, **kwargs):
        super(BasicConv1d, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm1d(out_channels)
        self.activation = nn.GELU()
        # self.dp = nn.Dropout(dropout)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return self.activation(x)


class InceptionA(nn.Module):

    def __init__(self, in_channels, out_channels, dropout):
        super(InceptionA, self).__init__()
        self.branch1x1 = BasicConv1d(in_channels, out_channels, dropout, kernel_size=1)

        self.branch3x3_1 = BasicConv1d(in_channels, out_channels, dropout, kernel_size=1)
        self.branch3x3_2 = BasicConv1d(out_channels, out_channels, dropout, kernel_size=3, padding=1)

        self.branch3x3dbl_1 = BasicConv1d(in_channels, out_channels, dropout, kernel_size=1)
        self.branch3x3dbl_2 = BasicConv1d(out_channels, out_channels, dropout, kernel_size=3, padding=1)
        self.branch3x3dbl_3 = BasicConv1d(out_channels, out_channels, dropout, kernel_size=3, padding=1)

        self.branch_pool = BasicConv1d(in_channels, out_channels, dropout, kernel_size=1)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch3x3 = self.branch3x3_1(x)
        branch3x3 = self.branch3x3_2(branch3x3)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)

        branch_pool = F.avg_pool1d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch3x3, branch3x3dbl, branch_pool]
        return torch.cat(outputs, 1)


class DSEModel(nn.Module):
    def __init__(self, args, fpt_feature, mpnn_feature, weave_feature, afp_feature, nf_feature, vec_feature):
        super(DSEModel, self).__init__()
        self.D_n = args.D_n
        self.S_n = args.S_n
        self.args = args
        self.dropout = args.dropout
        self.hid_dim = args.hid_dim  # 64
        self.fpt_dim = args.fpt_dim  # 128
        self.vec_dim = args.vec_dim  # 300
        self.vec_len = args.vec_len  # 100
        self.kge_dim = args.kge_dim  # 400
        self.gnn_dim = args.gnn_dim  # 617
        self.gnn_num = 3
        self.bio_dim = args.bio_dim  # 768
        self.hid_len = 4

        self.fpts = fpt_feature
        self.mpnns = mpnn_feature
        self.weaves = weave_feature
        self.afps = afp_feature
        self.nfs = nf_feature
        self.vecs = vec_feature
        se_feature = nn.Parameter(torch.Tensor(self.S_n, 8 * self.hid_dim))
        se_feature.data.normal_(0, 0.1)
        self.sees = se_feature

        self.softmax = nn.Softmax(dim=2)
        self.conv_ds = nn.Sequential(
            InceptionA(in_channels=self.hid_dim, out_channels=self.hid_dim * 4, dropout=args.dropout))
        self.rnn_mol = nn.GRU(self.vec_dim, self.hid_dim // 2, num_layers=2, bidirectional=True)
        self.rnn_fpt = nn.GRU(self.fpt_dim, self.hid_dim // 2, num_layers=2, bidirectional=True)
        self.rnn_gnn = nn.GRU(self.gnn_dim, self.hid_dim // 2, num_layers=1, bidirectional=True)
        self.self_attn1 = nn.MultiheadAttention(embed_dim=self.hid_dim, num_heads=2, dropout=args.dropout)
        self.norm1 = nn.LayerNorm(self.hid_dim)

        self.Wd = nn.Sequential(nn.Linear(self.hid_dim * 64, self.hid_dim * 32),
                                nn.BatchNorm1d(self.hid_dim * 32),
                                nn.GELU(),
                                nn.Linear(self.hid_dim * 32, self.hid_dim * 8))
        self.Wa = nn.Sequential(nn.Linear(self.hid_dim, self.hid_dim * 4),
                                nn.GELU(),
                                nn.Linear(self.hid_dim * 4, self.hid_dim * 8))


    def forward(self):
        vecs, _ = self.rnn_mol(self.vecs.unsqueeze(0))

        mpnns, _ = self.rnn_gnn(self.mpnns.unsqueeze(0))
        afps, _ = self.rnn_gnn(self.afps.unsqueeze(0))
        weaves, _ = self.rnn_gnn(self.weaves.unsqueeze(0))
        nfs, _ = self.rnn_gnn(self.nfs.unsqueeze(0))
        cses = torch.concat((mpnns, afps, weaves, nfs), dim=0)
        cses = torch.mean(cses, dim=0, keepdim=True)

        fpts, _ = self.rnn_fpt(self.fpts.unsqueeze(0))

        mat = torch.concat((vecs, cses, fpts), 0)
        mm = torch.mean(mat, dim=0, keepdim=True)
        mcn = torch.concat((mat, mm), 0).permute(1, 2, 0)

        ds = self.conv_ds(mcn)  # batch * 4hid_dim * len
        ds = ds.flatten(1)

        da = self.self_attn1(mm, mat, mat)[0]
        da = self.norm1(da)
        da = da.squeeze(0)

        drug_embeddings = (self.Wd(ds) + self.Wa(da)) / 2
        se_embeddings = self.sees

        outputs = torch.mm(drug_embeddings, se_embeddings.t())
        return outputs
