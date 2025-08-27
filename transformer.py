import numpy as np

class MultiHeadAttention:
    def __init__(self, d_model, n_heads):
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        # combined weight matrix for q, k, v
        self.w_qkv = np.random.randn(d_model, d_model * 3) * 0.02
        self.w_o = np.random.randn(d_model, d_model) * 0.02
    
    def forward(self, x, mask=None):
        batch_size, seq_len, d_model = x.shape
        
        # get q, k, v in one go
        qkv = np.dot(x, self.w_qkv)
        q, k, v = np.split(qkv, 3, axis=-1)
        
        # reshape for multi-head
        q = q.reshape(batch_size, seq_len, self.n_heads, self.d_k).transpose(0, 2, 1, 3)
        k = k.reshape(batch_size, seq_len, self.n_heads, self.d_k).transpose(0, 2, 1, 3)
        v = v.reshape(batch_size, seq_len, self.n_heads, self.d_k).transpose(0, 2, 1, 3)
        
        # attention scores
        scores = np.matmul(q, k.transpose(-1, -2)) / np.sqrt(self.d_k)
        
        # apply mask
        if mask is not None:
            scores = np.where(mask == 0, -1e9, scores)
        
        # softmax
        scores = scores - np.max(scores, axis=-1, keepdims=True)
        attn_weights = np.exp(scores) / np.sum(np.exp(scores), axis=-1, keepdims=True)
        
        # apply to values
        out = np.matmul(attn_weights, v)
        
        # concat heads
        out = out.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, d_model)
        
        # output projection
        return np.dot(out, self.w_o)

class FeedForward:
    def __init__(self, d_model, d_ff):
        self.w1 = np.random.randn(d_model, d_ff) * 0.02
        self.w2 = np.random.randn(d_ff, d_model) * 0.02
    
    def forward(self, x):
        # linear -> relu -> linear
        hidden = np.maximum(0, np.dot(x, self.w1))
        return np.dot(hidden, self.w2)

class TransformerBlock:
    def __init__(self, d_model, n_heads, d_ff):
        self.attention = MultiHeadAttention(d_model, n_heads)
        self.ffn = FeedForward(d_model, d_ff)
    
    def layer_norm(self, x):
        # simple layer norm
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)
        return (x - mean) / np.sqrt(var + 1e-6)
    
    def forward(self, x, mask=None):
        # attention with residual
        attn_out = self.attention.forward(x, mask)
        x = self.layer_norm(x + attn_out)
        
        # feedforward with residual  
        ff_out = self.ffn.forward(x)
        x = self.layer_norm(x + ff_out)
        
        return x

class Transformer:
    def __init__(self, vocab_size, d_model, n_heads, n_layers, d_ff, max_len=512):
        self.d_model = d_model
        
        # embeddings
        self.token_emb = np.random.randn(vocab_size, d_model) * 0.02
        
        # positional encoding
        pos = np.arange(max_len).reshape(-1, 1)
        div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
        pe = np.zeros((max_len, d_model))
        pe[:, 0::2] = np.sin(pos * div_term)
        pe[:, 1::2] = np.cos(pos * div_term)
        self.pos_emb = pe
        
        # transformer blocks
        self.blocks = [TransformerBlock(d_model, n_heads, d_ff) for _ in range(n_layers)]
        
        # output head
        self.lm_head = np.random.randn(d_model, vocab_size) * 0.02
    
    def forward(self, input_ids):
        batch_size, seq_len = input_ids.shape
        
        # embeddings
        x = self.token_emb[input_ids]
        x += self.pos_emb[:seq_len]
        
        # causal mask
        mask = np.tril(np.ones((seq_len, seq_len)))
        mask = mask.reshape(1, 1, seq_len, seq_len)
        
        # transformer blocks
        for block in self.blocks:
            x = block.forward(x, mask)
        
        # output projection
        return np.dot(x, self.lm_head)

# simple usage
if __name__ == "__main__":
    # small model
    model = Transformer(
        vocab_size=1000,
        d_model=256, 
        n_heads=8,
        n_layers=4,
        d_ff=1024
    )
    
    # test input
    input_ids = np.random.randint(0, 1000, (2, 20))
    logits = model.forward(input_ids)
    
    print(f"input: {input_ids.shape}")
    print(f"output: {logits.shape}")
