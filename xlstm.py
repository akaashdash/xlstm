import torch
import torch.nn as nn

class sLSTMLayer(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(sLSTMLayer, self).__init__()
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

    def forward(self, x, h_prev, c_prev, n_prev, m_prev):
        z = torch.tanh(self.Wz(x) + self.Rz(h_prev))
        o = torch.sigmoid(self.Wo(x) + self.Ro(h_prev))

        i_tilde = self.Wi(x) + self.Ri(h_prev)
        f_tilde = self.Wf(x) + self.Rf(h_prev)
        m_t = torch.max(f_tilde + m_prev, i_tilde)
        i = torch.exp(i_tilde - m_t)
        f = torch.exp(f_tilde + m_prev - m_t)
        
        c_t = f * c_prev + i * z
        n_t = f * n_prev + i
        h_t = o * c_t / n_t # A.2 says n_t^{-1} ??
        
        return h_t, c_t, n_t, m_t
    
class sLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1):
        super(sLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.layers = nn.ModuleList([sLSTMLayer(input_size, hidden_size) for _ in range(num_layers)])

    def forward(self, x, h_0=None, c_0=None, n_0=None, m_0=None):
        assert x.dim() == 2
        assert x.shape(1) == self.input_size
        h = torch.zeros(self.num_layers, self.hidden_size) if h_0 is None else h_0
        c = torch.zeros(self.num_layers, self.hidden_size) if c_0 is None else c_0
        n = torch.zeros(self.num_layers, self.hidden_size) if n_0 is None else n_0
        m = torch.zeros(self.num_layers, self.hidden_size) if m_0 is None else m_0
        output = []
        for t in range(len(x)):
            for i in range(self.num_layers):
                h[i], c[i], n[i], m[i] = self.layers[i](x[t], h[i], c[i], n[i], m[i])
            output.append(h[-1])

        output = torch.stack(output)
        return output, h
    
class mLSTMLayer(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(mLSTMLayer, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        self.Wq = nn.Linear(input_size, hidden_size)
        self.Wk = nn.Linear(input_size, hidden_size)
        self.Wv = nn.Linear(input_size, hidden_size)
        
        self.Wi = nn.Linear(input_size, hidden_size)
        self.Wf = nn.Linear(input_size, hidden_size)
        self.Wo = nn.Linear(input_size, hidden_size)

    def forward(self, x, h_prev, c_prev, n_prev, m_prev): # Parallelize according to A.3?
        q = self.Wq(x)
        k = self.Wk(x) / (self.hidden_size ** 0.5)
        v = self.Wv(x)
        
        o = torch.sigmoid(self.Wo(x))

        i_tilde = self.Wi(x)
        f_tilde = self.Wf(x)
        m_t = torch.max(f_tilde + m_prev, i_tilde)
        i = torch.exp(i_tilde - m_t)
        f = torch.exp(f_tilde + m_prev - m_t)
        
        c_t = f * c_prev + i * torch.outer(v, k)
        n_t = f * n_prev + i * k
        h_t = o * torch.matmul(c_t, q) / torch.max(torch.abs(torch.matmul(n_t.T, q)), 1)
        
        return h_t, c_t, n_t, m_t
    
class mLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1):
        super(mLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.layers = nn.ModuleList([mLSTMLayer(input_size, hidden_size) for _ in range(num_layers)])

    def forward(self, x, h_0=None, c_0=None, n_0=None, m_0=None):
        assert x.dim() == 2
        assert x.shape(1) == self.input_size
        h = torch.zeros(self.num_layers, self.hidden_size) if h_0 is None else h_0
        c = torch.zeros(self.num_layers, self.hidden_size) if c_0 is None else c_0
        n = torch.zeros(self.num_layers, self.hidden_size) if n_0 is None else n_0
        m = torch.zeros(self.num_layers, self.hidden_size) if m_0 is None else m_0
        output = []
        for t in range(len(x)):
            for i in range(self.num_layers):
                h[i], c[i], n[i], m[i] = self.layers[i](x[t], h[i], c[i], n[i], m[i])
            output.append(h[-1])

        output = torch.stack(output)
        return output, h
    
class xLSTMBlock(nn.Module):
    def __init__(self, input_size, hidden_size, block_type='post'):
        super(xLSTMBlock, self).__init__()
        self.block_type = block_type
        
        if block_type == 'post':
            self.norm1 = nn.LayerNorm(input_size)
            self.sLSTMLayer = sLSTMLayer(input_size, hidden_size)
            self.norm2 = nn.LayerNorm(hidden_size)
            self.ff = nn.Sequential(
                nn.Linear(hidden_size, 4 * hidden_size),
                nn.GELU(),
                nn.Linear(4 * hidden_size, hidden_size)
            )
        elif block_type == 'pre':
            self.norm = nn.LayerNorm(input_size)
            self.ff1 = nn.Linear(input_size, 2 * hidden_size)
            self.mLSTMLayer = mLSTMLayer(2 * hidden_size, hidden_size)
            self.ff2 = nn.Linear(hidden_size, hidden_size)
        else:
            raise ValueError('Invalid block type.')
        
    def forward(self, x, h_prev, c_prev, n_prev, m_prev):
        if self.block_type == 'post':
            x = self.norm1(x)
            h_t, c_t, n_t, m_t = self.sLSTMLayer(x, h_prev, c_prev, n_prev, m_prev)
            h_t = self.norm2(h_t)
            h_t = self.ff(h_t)
        elif self.block_type == 'pre':
            x = self.norm(x)
            x = self.ff1(x)
            h_t, c_t, n_t, m_t = self.mLSTMLayer(x, h_prev, c_prev, n_prev, m_prev)
            h_t = self.ff2(h_t)
        return h_t, c_t, n_t, m_t
    
class xLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, block_ratio):
        super(xLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.block_ratio = block_ratio
        
        self.blocks = nn.ModuleList()
        for i in range(num_layers):
            if i < block_ratio * num_layers:
                block = xLSTMBlock(input_size, hidden_size, block_type='post')
            else:
                block = xLSTMBlock(input_size, hidden_size, block_type='pre')
            self.blocks.append(block)

    def forward(self, x):
        # This makes sense to me, input size must match block size for continuous production
        # Not sure how this forward pass aligns with the paper
        assert len(x) == len(self.blocks)
        # No idea how to initialize these tbh
        h = [torch.zeros(self.hidden_size) for _ in range(len(x)+1)]
        c = [torch.zeros(self.hidden_size) for _ in range(len(x)+1)]
        n = [torch.zeros(self.hidden_size) for _ in range(len(x)+1)]
        m = [torch.zeros(self.hidden_size) for _ in range(len(x)+1)]
        for i in range(1, len(x)+1):
            h[i], c[i], n[i], m[i] = self.blocks[i-1](x[i-1], h[i-1], c[i-1], n[i-1], m[i-1])
        return h[-1]