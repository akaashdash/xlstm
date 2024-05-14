import torch
import torch.nn as nn
import torch.nn.functional as F

# https://discuss.pytorch.org/t/causal-convolution/3456/3
class CausalConv1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1, **kwargs):
        super(CausalConv1D, self).__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=self.padding, dilation=dilation, **kwargs)

    def forward(self, x):
        x = self.conv(x)
        return x[:, :, :-self.padding]

# https://github.com/myscience/x-lstm/blob/main/xlstm/utils.py
class BlockDiagonal(nn.Module):
    def __init__(self, in_features, out_features, num_blocks):
        super(BlockDiagonal, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_blocks = num_blocks
        
        assert in_features % num_blocks == 0
        assert out_features % num_blocks == 0
        
        block_in_features = in_features // num_blocks
        block_out_features = out_features // num_blocks
        
        self.blocks = nn.ModuleList([
            nn.Linear(block_in_features, block_out_features)
            for _ in range(num_blocks)
        ])
        
    def forward(self, x):
        x = x.chunk(self.num_blocks, dim=-1)
        x = [block(x_i) for block, x_i in zip(self.blocks, x)]
        x = torch.cat(x, dim=-1)
        return x

class sLSTMBlock(nn.Module):
    def __init__(self, input_size, hidden_size, num_heads, projection_factor=4/3):
        super(sLSTMBlock, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_size = hidden_size // num_heads
        self.projection_factor = projection_factor

        self.layer_norm = nn.LayerNorm(input_size)
        self.causal_conv = CausalConv1D(1, 1, 4)

        self.Wz = BlockDiagonal(input_size, hidden_size, num_heads)
        self.Wi = BlockDiagonal(input_size, hidden_size, num_heads)
        self.Wf = BlockDiagonal(input_size, hidden_size, num_heads)
        self.Wo = BlockDiagonal(input_size, hidden_size, num_heads)

        self.Rz = BlockDiagonal(hidden_size, hidden_size, num_heads)
        self.Ri = BlockDiagonal(hidden_size, hidden_size, num_heads)
        self.Rf = BlockDiagonal(hidden_size, hidden_size, num_heads)
        self.Ro = BlockDiagonal(hidden_size, hidden_size, num_heads)

        self.group_norm = nn.GroupNorm(num_heads, hidden_size)

        self.up_proj_left = nn.Linear(hidden_size, int(hidden_size * projection_factor))
        self.up_proj_right = nn.Linear(hidden_size, int(hidden_size * projection_factor))
        self.down_proj = nn.Linear(int(hidden_size * projection_factor), input_size)

    def forward(self, x, h_prev, c_prev, n_prev, m_prev):
        x_norm = self.layer_norm(x)
        x_conv = F.silu(self.causal_conv(x_norm.unsqueeze(1)).squeeze(1))

        z = torch.tanh(self.Wz(x) + self.Rz(h_prev))
        o = torch.sigmoid(self.Wo(x) + self.Ro(h_prev))
        i_tilde = self.Wi(x_conv) + self.Ri(h_prev)
        f_tilde = self.Wf(x_conv) + self.Rf(h_prev)

        m_t = torch.max(f_tilde + m_prev, i_tilde)
        i = torch.exp(i_tilde - m_t)
        f = torch.exp(f_tilde + m_prev - m_t)

        c_t = f * c_prev + i * z
        n_t = f * n_prev + i
        h_t = o * c_t / n_t

        output = h_t
        output_norm = self.group_norm(output)
        output_left = self.up_proj_left(output_norm)
        output_right = self.up_proj_right(output_norm)
        output_gated = F.gelu(output_right)
        output = output_left * output_gated
        output = self.down_proj(output)
        final_output = output + x

        return final_output, h_t, c_t, n_t, m_t

class mLSTMBlock(nn.Module):
    def __init__(self, input_size, hidden_size, num_heads, projection_factor=2):
        super(mLSTMBlock, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_size = hidden_size // num_heads
        self.projection_factor = projection_factor

        self.layer_norm = nn.LayerNorm(input_size)
        self.up_proj_left = nn.Linear(input_size, int(input_size * projection_factor))
        self.up_proj_right = nn.Linear(input_size, hidden_size)
        self.down_proj = nn.Linear(hidden_size, input_size)

        self.causal_conv = CausalConv1D(1, 1, 4)
        self.skip_connection = nn.Linear(int(input_size * projection_factor), hidden_size)

        self.Wq = BlockDiagonal(int(input_size * projection_factor), hidden_size, num_heads)
        self.Wk = BlockDiagonal(int(input_size * projection_factor), hidden_size, num_heads)
        self.Wv = BlockDiagonal(int(input_size * projection_factor), hidden_size, num_heads)
        self.Wi = nn.Linear(int(input_size * projection_factor), hidden_size)
        self.Wf = nn.Linear(int(input_size * projection_factor), hidden_size)
        self.Wo = nn.Linear(int(input_size * projection_factor), hidden_size)

        self.group_norm = nn.GroupNorm(num_heads, hidden_size)

    def forward(self, x, h_prev, c_prev, n_prev, m_prev):
        x_norm = self.layer_norm(x)
        x_up_left = self.up_proj_left(x_norm)
        x_up_right = self.up_proj_right(x_norm)

        x_conv = F.silu(self.causal_conv(x_up_left.unsqueeze(1)).squeeze(1))
        x_skip = self.skip_connection(x_conv)

        q = self.Wq(x_conv)
        k = self.Wk(x_conv) / (self.head_size ** 0.5)
        v = self.Wv(x_up_left)

        i_tilde = self.Wi(x_conv)
        f_tilde = self.Wf(x_conv)
        o = torch.sigmoid(self.Wo(x_up_left))

        m_t = torch.max(f_tilde + m_prev, i_tilde)
        i = torch.exp(i_tilde - m_t)
        f = torch.exp(f_tilde + m_prev - m_t)

        c_t = f * c_prev + i * (v * k) # v @ k.T
        n_t = f * n_prev + i * k
        h_t = o * (c_t * q) / torch.max(torch.abs(n_t.T @ q), 1)[0] # o * (c @ q) / max{|n.T @ q|, 1}

        output = h_t
        output_norm = self.group_norm(output)
        output = output_norm + x_skip
        output = output * F.silu(x_up_right)
        output = self.down_proj(output)
        final_output = output + x

        return final_output, h_t, c_t, n_t, m_t

class xLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_heads, layer_order, num_copies=1, projection_factor_slstm=4/3, projection_factor_mlstm=2):
        super(xLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.layer_order = layer_order
        self.num_copies = num_copies
        self.projection_factor_slstm = projection_factor_slstm
        self.projection_factor_mlstm = projection_factor_mlstm

        self.layers = nn.ModuleList()
        for _ in range(num_copies):
            for layer_type in layer_order:
                if layer_type == 's':
                    layer = sLSTMBlock(input_size, hidden_size, num_heads, projection_factor=projection_factor_slstm)
                elif layer_type == 'm':
                    layer = mLSTMBlock(input_size, hidden_size, num_heads, projection_factor=projection_factor_mlstm)
                else:
                    raise ValueError(f"Invalid layer type: {layer_type}. Choose 's' for sLSTM or 'm' for mLSTM.")
                self.layers.append(layer)

    def forward(self, x, h_prev, c_prev, n_prev, m_prev):
        assert x.size(-1) == self.input_size
        for layer in self.layers:
            x, h_prev, c_prev, n_prev, m_prev = layer(x, h_prev, c_prev, n_prev, m_prev)
        return x, h_prev, c_prev, n_prev, m_prev