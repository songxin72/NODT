import torch.nn as nn
import torch
from torch.autograd import Variable
from torch.nn import Parameter
import numpy as np
from .utils import series_decomp_multi


class AttentionLayer(nn.Module):
    def __init__(self, model_dim, num_heads=8, mask=False):
        super().__init__()

        self.model_dim = model_dim
        self.num_heads = num_heads
        self.mask = mask

        self.head_dim = model_dim // num_heads
        self.input_proj = nn.Linear(10, 1)

        self.input_dim = num_heads

    def forward(self, query, key, value):
        batch_size = query.shape[0]
        # Qhead, Khead, Vhead (num_heads * batch_size, length, head_dim)
        query = torch.stack(torch.split(query, [self.head_dim] * self.input_dim, 2), dim=0)  # 按照多头的个数分割[16, 128, 20, 20]
        key = torch.stack(torch.split(key, [self.head_dim] * self.input_dim, 2), dim=0)
        value = torch.stack(torch.split(value, [self.head_dim] * self.input_dim, 2), dim=0)

        key = key.transpose(
            -1, -2
        )  # (num_heads * batch_size, head_dim, src_length) [16, 128, 20, 20]

        key = key.sum(3).unsqueeze(dim=2)

        attn_score = torch.tanh((
            query * key
        ) / self.head_dim**0.5)  # (num_heads * batch_size, tgt_length, src_length) [64, 307, 12, 12]

        attn_score = torch.softmax(attn_score, dim=-1)
        t_out = attn_score * value  # (num_heads * batch_size, tgt_length, head_dim) 64, 307, 12, 38]

        # t_out = torch.cat(
        #     torch.split(t_out, batch_size, dim=0), dim=-1
        # )  # (batch_size, head_dim * num_heads = model_dim) [16, 307, 12, 152]

        v_query = t_out
        v_key = t_out
        v_value = t_out
        v_query = v_query.sum(2)#.sum(2).unsqueeze(dim=2)  # [num_head, batch_size, head_dim]
        v_key = v_key.sum(2)# .sum(2).unsqueeze(dim=2) / self.head_dim
        v_value = v_value.sum(3).permute(1, 0, 2)  # [ batch_size, num_head, time_step]

        v_attn_score = torch.tanh(self.input_proj(v_query * v_key))
        v_attn_score = torch.softmax(v_attn_score, dim=-1).permute(1, 0, 2)
        out = v_value * v_attn_score

        return out


class SelfAttentionLayer(nn.Module):
    def __init__(
        self, model_dim, feed_forward_dim=2048, num_heads=8, dropout=0, mask=False
    ):
        super().__init__()

        self.attn = AttentionLayer(model_dim, num_heads, mask)

    def forward(self, x, dim=-2):  # [16, 12, 307, 152]
        # x = x.transpose(dim, -2)
        out = self.attn(x, x, x)  # (batch_size, ..., length, model_dim)
        return out


class AttentionLayer2(nn.Module):
    def __init__(self, model_dim, num_heads=8, mask=False):
        super().__init__()

        self.model_dim = model_dim
        self.num_heads = num_heads
        self.mask = mask

        self.head_dim = 10
        self.input_dim = num_heads

    def forward(self, query, key, value):  # i:[128, 340, 320] head_dim=20, input_dim=16
        batch_size = query.shape[0]

        # Qhead, Khead, Vhead (num_heads * batch_size, length, head_dim) o:[16, 128, 340, 20] [28, 128, 20, 10]
        query = torch.stack(torch.split(query, [self.head_dim] * self.input_dim, 2), dim=0)  # 按照多头的个数分割[64, 12, 307, 38]
        key = torch.stack(torch.split(key, [self.head_dim] * self.input_dim, 2), dim=0)
        value = torch.stack(torch.split(value, [self.head_dim] * self.input_dim, 2), dim=0)
        key = key.transpose(
            -1, -2
        )  # (num_heads * batch_size, head_dim, src_length) [64, 307, 38, 12]

        # key = key.sum(3).unsqueeze(dim=2)
        # first_out = torch.zeros(query.shape).cuda()
        # attq_shape = key.shape[3]
        # for i in range(query.shape[3]):
        #     temp = 0
        #     for j in range(key.shape[3]):
        #         temp += query[:, :, :, i] * key[:, :, i, j]
        #     first_out[:, :, :, i] = temp / attq_shape

        attn_score = torch.tanh((
            query * key  # query * key
        ) / self.head_dim**0.5)  # (num_heads * batch_size, ..., tgt_length, src_length) [64, 307, 12, 12]

        attn_score = torch.softmax(attn_score, dim=-1)
        t_out = attn_score * value  # (num_heads * batch_size, tgt_length, head_dim) 64, 307, 12, 38]

        # t_out = torch.cat(
        #     torch.split(t_out, batch_size, dim=0), dim=-1
        # )  # (batch_size, ..., tgt_length, head_dim * num_heads = model_dim) [16, 307, 12, 152]

        v_query = t_out
        v_key = t_out
        v_value = t_out
        v_query = v_query.sum(2)  # [num_head, batch_size, head_dim]
        v_key = v_key.sum(2)
        v_value = v_value.sum(3).permute(1, 0, 2)  # 默认为3 [28, 128, 20]

        v_attn_score = torch.tanh((v_query * v_key)).permute(1, 0, 2)  # [28, 128, 10]
        v_attn_score = torch.softmax(v_attn_score, dim=-1)

        # out = torch.zeros(v_value.shape).cuda()
        # attn_shape = v_attn_score.shape[2]
        # for i in range(v_value.shape[2]):
        #     out[:, :, i] = (v_attn_score * v_value[:, :, i].unsqueeze(dim=2)).sum(2) / attn_shape

        out = v_value * v_attn_score

        return out


class SelfAttentionLayer2(nn.Module):
    def __init__(
        self, model_dim, feed_forward_dim=2048, num_heads=8, dropout=0, mask=False
    ):
        super().__init__()

        self.attn = AttentionLayer3(model_dim, num_heads, mask)

    def forward(self, x):  # i: [batch_size, time_step, fea_dim]
        out = self.attn(x, x, x)  # (batch_size, ..., length, model_dim)
        return out


class GRUModel_former(nn.Module):
    def __init__(self, hidden_dim, input_dim, layer_dim, gate_type, qz_para_dim):
        super(GRUModel_former, self).__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        num_heads = 7  # energy:26
        num_layers = 1
        dropout = 0.1
        self.attn_dim = hidden_dim
        self.in_steps = 20
        self.attn_layers_t = nn.ModuleList(
            [
                SelfAttentionLayer(self.attn_dim, self.attn_dim, num_heads, dropout)
                for _ in range(num_layers)
            ]
        )

        self.fc_q = nn.Linear(hidden_dim, hidden_dim * 2)
        self.input_proj = nn.Linear(1, self.hidden_dim)
        self.num_nodes = input_dim

        self.model_dim = hidden_dim
        self.output_proj = nn.Linear(
            input_dim, input_dim * 2
        )
        self.input_dim = input_dim
        self.head_dim = hidden_dim // num_heads
        self.att_w_temp = Parameter(torch.zeros([input_dim, 1, 1, self.head_dim], dtype=torch.float32), requires_grad=True)
        self.bias_temp = Parameter(torch.ones([input_dim, 1, 1], dtype=torch.float32), requires_grad=True)
        self.aug_w_vari = Parameter(torch.zeros([1, 1, self.head_dim], dtype=torch.float32), requires_grad=True)
        self.bias_vari = Parameter(torch.ones(1, dtype=torch.float32), requires_grad=True)

    def forward(self, x):
        for attn in self.attn_layers_t:  # 使用自注意力
            outs = attn(x, dim=1)

        # x = torch.split(x, [self.head_dim] * self.input_dim, 2)
        # tmph = torch.stack(x, dim=0)  # [num_head, batch_size, time_step, head_dim]
        # logit: [num_head, batch_size, time_step]
        # temp_logit = torch.tanh((tmph * self.att_w_temp).sum(3) + self.bias_temp)
        # [num_head, batch_size, time_step]
        # temp_weight = torch.softmax(temp_logit, dim=-1)
        # [num_head, batch_size, time_step, head_dim]
        # outs = tmph * temp_weight.unsqueeze(-1)

        # 自注意力2
        # h_temp = outs.sum(2)  # [num_head, batch_size, head_dim]
        # v_temp = outs.sum(3).permute(1, 0, 2)  # [batch_size, num_head, time_step]
        # [batch_size, num_head, 1]
        # vari_logits = torch.tanh(((h_temp * self.aug_w_vari).sum(2, keepdim=True) + self.bias_vari).permute(1, 0, 2))
        # [batch_size, num_head, 1]
        # vari_weight = torch.softmax(vari_logits, dim=1)  # 计算权重
        # c_t = (v_temp * vari_weight).sum(2)  # o:[128, 16]

        c_t = outs.sum(2)
        out = self.output_proj(c_t)

        return out


class GRUModel_former2(nn.Module):
    def __init__(self, hidden_dim, input_dim, layer_dim, gate_type, qz_para_dim):
        super(GRUModel_former2, self).__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        num_heads = 28  # energy: elec:17
        num_layers = 1
        dropout = 0.1
        self.attn_dim = hidden_dim + 20
        self.in_steps = 20
        self.attn_layers_t = nn.ModuleList(
            [
                SelfAttentionLayer2(self.attn_dim, self.attn_dim, num_heads, dropout)
                for _ in range(num_layers)
            ]
        )

        self.num_nodes = input_dim

        self.model_dim = hidden_dim
        out_dim = 28  # elec:17
        self.output_proj = nn.Linear(
            out_dim, input_dim * 2
        )
        self.input_dim = input_dim
        self.head_dim = hidden_dim // num_heads

    def forward(self, x, att1):  # i:[batch_size, time_step, input_dim]
        att0 = torch.FloatTensor([]).cuda()
        for i in range(att1.shape[0]):
            att = np.corrcoef(att1[i].cpu().detach().numpy())  # (time_step, time_step)
            att = np.reshape(att, (1, 20, 20))
            att = torch.Tensor(att).cuda()

            # att = x[i] - (x[i].permute(1, 0)@att).permute(1, 0)
            # att = att.unsqueeze(dim=0)
            val = np.corrcoef(x[i].cpu().detach().numpy())
            val = np.reshape(val, (1, 20, 20))
            val = torch.Tensor(val).cuda()
            att = val - att
            # attn_score = torch.softmax(att, dim=-1)
            # att2 = attn_score @ x[i]

            att0 = torch.cat([att0, att])
        h = torch.cat((x, att0), dim=2)

        for attn in self.attn_layers_t:  # 使用自注意力
            outs = attn(h)

        c_t = outs.sum(2)
        out = self.output_proj(c_t)

        return out


class Model_fusion(nn.Module):
    def __init__(self, hidden_dim, input_dim, layer_dim, gate_type, qz_para_dim):
        super(Model_fusion, self).__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        num_heads = 28  # energy: elec:17
        num_layers = 1
        dropout = 0.1
        self.attn_dim = hidden_dim
        self.in_steps = 20
        self.attn_layers_t = nn.ModuleList(
            [
                SelfAttentionLayer2(self.attn_dim, self.attn_dim, num_heads, dropout)
                for _ in range(num_layers)
            ]
        )

        self.fc_q = nn.Linear(hidden_dim, hidden_dim * 2)
        self.input_proj = nn.Linear(1, self.hidden_dim)
        self.num_nodes = input_dim

        self.model_dim = hidden_dim
        out_dim = 17  # energy: elec:17
        self.output_proj = nn.Linear(
            input_dim * 2, input_dim * 2
        )
        self.input_dim = input_dim
        self.head_dim = hidden_dim // num_heads

    def forward(self, x, att1):  # i:[128, 32]
        att0 = torch.FloatTensor([]).cuda()
        for i in range(att1.shape[0]):
            att = np.corrcoef(att1[i].unsqueeze(dim=1).cpu().detach().numpy())  # (32, 32)
            # att = np.reshape(att, (1, 32, 32))
            att = torch.Tensor(att).cuda()

            att = torch.tanh((torch.matmul(x[i], att)))
            attn_score = torch.softmax(att, dim=-1)
            att = attn_score * x[i]
            att = att.unsqueeze(dim=0)
            # val = np.corrcoef(x[i].cpu().detach().numpy())
            # val = np.reshape(val, (1, 20, 20))
            # val = torch.Tensor(val).cuda()
            # att = val - att

            att0 = torch.cat([att0, att])
        h = att0

        return h


class AttentionLayer3(nn.Module):
    def __init__(self, model_dim, num_heads=8, mask=False):
        super().__init__()

        self.model_dim = model_dim
        self.num_heads = 10
        self.mask = mask

        self.head_dim = model_dim // num_heads
        self.input_proj = nn.Linear(self.num_heads, 1)

        self.input_dim = num_heads

    def forward(self, query, key, value):  # [16, 12, 307, 152]
        # Qhead, Khead, Vhead (num_heads * batch_size, length, head_dim)
        query = torch.stack(torch.split(query, [self.head_dim] * self.input_dim, 2), dim=0)  # 按照多头的个数分割[16, 128, 20, 20]
        key = torch.stack(torch.split(key, [self.head_dim] * self.input_dim, 2), dim=0)
        value = torch.stack(torch.split(value, [self.head_dim] * self.input_dim, 2), dim=0)

        key = key.transpose(
            -1, -2
        )  # (num_heads * batch_size, head_dim, src_length) [16, 128, 20, 20]

        key = key.sum(3).unsqueeze(dim=2)
        attn_score = torch.tanh((
            query * key
        ) / self.head_dim**0.5)  # energy:0.5, pm25:0.05

        attn_score = torch.softmax(attn_score, dim=-1)
        t_out = attn_score * value  # (num_heads * batch_size, tgt_length, head_dim) 64, 307, 12, 38]

        v_query = t_out
        v_key = t_out
        v_value = t_out
        v_query = v_query.sum(2)
        v_key = v_key.sum(2)
        v_value = v_value.sum(3).permute(1, 0, 2)

        v_attn_score = torch.tanh(self.input_proj(v_query * v_key))
        v_attn_score = torch.softmax(v_attn_score, dim=-1).permute(1, 0, 2)
        out = v_value * v_attn_score

        return out


class GRUModel_former3(nn.Module):
    def __init__(self, hidden_dim, input_dim, layer_dim, gate_type, qz_para_dim):
        super(GRUModel_former3, self).__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        num_heads = 28  # pm:9 energy:28 elec:17
        num_layers = 1
        dropout = 0.1
        self.attn_dim = hidden_dim + 20
        self.in_steps = 20
        self.attn_layers_t = nn.ModuleList(
            [
                SelfAttentionLayer2(self.attn_dim, self.attn_dim, num_heads, dropout)
                for _ in range(num_layers)
            ]
        )

        self.num_nodes = input_dim

        self.model_dim = hidden_dim
        out_dim = num_heads  # energy:28 elec:17
        self.output_proj = nn.Linear(
            out_dim, input_dim * 2
        )
        self.input_dim = input_dim
        self.head_dim = self.attn_dim // num_heads
        self.num_head = num_heads

        self.aug_w_vari = Parameter(torch.zeros([1, 1, self.head_dim], dtype=torch.float32), requires_grad=True)
        self.bias_vari = Parameter(torch.ones(1, dtype=torch.float32), requires_grad=True)

    def forward(self, x, att1):  # i:[batch_size, time_step, input_dim]
        att0 = torch.FloatTensor([]).cuda()
        for i in range(att1.shape[0]):
            att = np.corrcoef(att1[i].cpu().detach().numpy())  # (time_step, time_step)
            att = np.reshape(att, (1, 20, 20))
            att = torch.Tensor(att).cuda()

            val = np.corrcoef(x[i].cpu().detach().numpy())
            val = np.reshape(val, (1, 20, 20))
            val = torch.Tensor(val).cuda()
            att = val - att

            att0 = torch.cat([att0, att])
        h = torch.cat((x, att0), dim=2)

        for attn in self.attn_layers_t:  # 使用自注意力
            outs = attn(h)

        # 自注意力2
        # h_temp = outs.sum(2)  # [num_head, batch_size, head_dim]
        # v_temp = outs.sum(3).permute(1, 0, 2)  # [batch_size, num_head, time_step]
        # vari_logits = torch.tanh(((h_temp * self.aug_w_vari).sum(2, keepdim=True) + self.bias_vari).permute(1, 0, 2))
        # vari_weight = torch.softmax(vari_logits, dim=1)  # 计算权重
        # c_t = (v_temp * vari_weight).sum(2)  # o:[128, 16]

        c_t = outs.sum(2)
        out = self.output_proj(c_t)

        return out


class AttentionLayer4(nn.Module):
    def __init__(self, model_dim, num_heads=8, mask=False):
        super().__init__()

        self.model_dim = model_dim
        self.num_heads = 10
        self.mask = mask

        self.head_dim = model_dim // num_heads
        self.input_proj = nn.Linear(self.num_heads, 1)

        self.input_dim = num_heads

    def forward(self, query, key, value):  # [16, 12, 307, 152]
        # Qhead, Khead, Vhead (num_heads * batch_size, length, head_dim)
        query = torch.stack(torch.split(query, [self.head_dim] * self.input_dim, 2), dim=0)  # 按照多头的个数分割[16, 128, 20, 20]
        key = torch.stack(torch.split(key, [self.head_dim] * self.input_dim, 2), dim=0)
        value = torch.stack(torch.split(value, [self.head_dim] * self.input_dim, 2), dim=0)

        key = key.transpose(
            -1, -2
        )  # (num_heads * batch_size, head_dim, src_length) [16, 128, 20, 20]

        key = key.sum(3).unsqueeze(dim=2)
        attn_score = torch.tanh((
            query * key
        ) / self.head_dim**0.5)  # energy:0.5, pm25:0.05

        attn_score = torch.softmax(attn_score, dim=-1)
        t_out = attn_score * value  # (num_heads * batch_size, tgt_length, head_dim) 64, 307, 12, 38]

        return t_out


class GRUModel_former4(nn.Module):
    def __init__(self, hidden_dim, input_dim, layer_dim, gate_type, qz_para_dim):
        super(GRUModel_former4, self).__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        num_heads = 28  # pm:9 energy:28 elec:17
        num_layers = 1
        dropout = 0.1
        self.attn_dim = hidden_dim + 20
        self.in_steps = 20
        self.attn_layers_t = nn.ModuleList(
            [
                SelfAttentionLayer2(self.attn_dim, self.attn_dim, num_heads, dropout)
                for _ in range(num_layers)
            ]
        )

        self.num_nodes = input_dim

        self.model_dim = hidden_dim
        out_dim = num_heads  # energy:28 elec:17
        self.output_proj = nn.Linear(
            out_dim, input_dim * 2
        )
        self.input_dim = input_dim
        self.head_dim = self.attn_dim // num_heads
        self.num_head = num_heads

        self.aug_w_vari = Parameter(torch.zeros([1, 1, self.head_dim], dtype=torch.float32), requires_grad=True)
        self.bias_vari = Parameter(torch.ones(1, dtype=torch.float32), requires_grad=True)

    def forward(self, x, att1):  # i:[batch_size, time_step, input_dim]

        att0 = torch.FloatTensor([]).cuda()
        for i in range(att1.shape[0]):
            att = np.corrcoef(att1[i].cpu().detach().numpy())  # (time_step, time_step)
            att = np.reshape(att, (1, 20, 20))
            att = torch.Tensor(att).cuda()

            val = np.corrcoef(x[i].cpu().detach().numpy())
            val = np.reshape(val, (1, 20, 20))
            val = torch.Tensor(val).cuda()
            att = val - att

            att0 = torch.cat([att0, att])
        h = torch.cat((x, att0), dim=2)

        for attn in self.attn_layers_t:  # 使用自注意力
            outs = attn(h)
        c_t = outs.permute(1, 0, 2, 3).sum(2).sum(2)
        out = self.output_proj(c_t)

        return out