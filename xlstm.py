import torch
import torch.nn as nn

class sLSTM(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(sLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.Wz = nn.Linear(input_size, hidden_size)
        self.Wi = nn.Linear(input_size, hidden_size)
        self.Wf = nn.Linear(input_size, hidden_size)
        self.Wo = nn.Linear(input_size, hidden_size)
        
        self.Rz = nn.Linear(hidden_size, hidden_size)
        self.Ri = nn.Linear(hidden_size, hidden_size)
        self.Rf = nn.Linear(hidden_size, hidden_size)
        self.Ro = nn.Linear(hidden_size, hidden_size)

    def forward(self, x, h_prev, c_prev, n_prev):
        z = torch.tanh(self.Wz(x) + self.Rz(h_prev))
        i = torch.exp(self.Wi(x) + self.Ri(h_prev))
        f = torch.sigmoid(self.Wf(x) + self.Rf(h_prev)) # could also be exp
        o = torch.sigmoid(self.Wo(x) + self.Ro(h_prev))
        
        c_t = f * c_prev + i * z
        n_t = f * n_prev + i
        h_t = o * c_t / n_t
        
        return h_t, c_t, n_t
    
class mLSTM(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(mLSTM, self).__init__()
        self.hidden_size = hidden_size
        
        self.Wq = nn.Linear(input_size, hidden_size)
        self.Wk = nn.Linear(input_size, hidden_size)
        self.Wv = nn.Linear(input_size, hidden_size)
        
        self.Wi = nn.Linear(input_size, hidden_size)
        self.Wf = nn.Linear(input_size, hidden_size)
        self.Wo = nn.Linear(input_size, hidden_size)

    def forward(self, x, h_prev, c_prev, n_prev):
        q = self.Wq(x)
        k = self.Wk(x) / (self.hidden_size ** 0.5)
        v = self.Wv(x)
        
        i = torch.exp(self.Wi(x))
        f = torch.sigmoid(self.Wf(x))  # could also be exp
        o = torch.sigmoid(self.Wo(x))
        
        c_t = f * c_prev + i * torch.outer(v, k)
        n_t = f * n_prev + i * k
        h_t = o * torch.matmul(c_t, q) / torch.max(torch.abs(torch.matmul(n_t.T, q)), 1)
        
        return h_t, c_t, n_t
    
class xLSTMBlock(nn.Module):
    def __init__(self, input_size, hidden_size, num_heads, block_type='post'):
        super(xLSTMBlock, self).__init__()
        self.block_type = block_type
        
        if block_type == 'post':
            self.sLSTM = sLSTM(input_size, hidden_size, num_heads)
            self.ff = nn.Sequential(
                nn.Linear(hidden_size, 4 * hidden_size),
                nn.GELU(),
                nn.Linear(4 * hidden_size, hidden_size)
            )
        elif block_type == 'pre':
            self.mLSTM = mLSTM(input_size, hidden_size)
            self.ff1 = nn.Linear(input_size, 2 * hidden_size)
            self.ff2 = nn.Linear(hidden_size, input_size)
        else:
            raise ValueError('Invalid block type.')
        
    def forward(self, x, h_prev=None):
        if self.block_type == 'post':
            h_t, c_t, n_t= self.sLSTM(x, h_prev)
            h_t = self.ff(h_t)
        elif self.block_type == 'pre':
            x = self.ff1(x)
            h_t, c_t, n_t = self.mLSTM(x)
            h_t = self.ff2(h_t)
        
        return h_t, c_t, n_t
    
class xLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_heads, block_ratio):
        super(xLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.block_ratio = block_ratio
        
        self.blocks = nn.ModuleList()
        for i in range(num_layers):
            if i < block_ratio * num_layers:
                block = xLSTMBlock(input_size, hidden_size, num_heads, block_type='post')
            else:
                block = xLSTMBlock(input_size, hidden_size, num_heads, block_type='pre')
            self.blocks.append(block)

    def forward(self, x):
        h_= x
        for block in self.blocks:
            h, c, n = block(h)
        return h