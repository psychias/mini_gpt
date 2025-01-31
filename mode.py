import torch 
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass



# metadata
@dataclass
class Config:
    hidden_size: int = 128
    learning_rate: float = 0.001
    batch_size: int = 64
    num_epochs: int = 2
    dropout: float = 0.2
    embedding_dim: int = 100
    block_size: int = 128 # max length of input text
    max_iters: int = 1000
    num_layers: int = 6
    num_heads: int = 6
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'


with open('data.txt', 'r') as f:
    text = f.read()

# create vocab
vocab = list(set(text))
vocab_size = len(vocab)
stoi = {char: i for i, char in enumerate(vocab)}
itos = {i: char for i, char in enumerate(vocab)}
encode = lambda x: [stoi[char] for char in x]
decode = lambda x: ''.join([itos[i] for i in x])

data= torch.tensor(encode(text),dtype=torch.long)


# split the data 
train_dataset = text[:int(len(data)*0.7)]
test_dataset = text[int(len(data)*0.7):int(len(data)*0.8)]
valid_dataset = text[int(len(data)*0.8):]


def create_dataset(data, block_size=128,batch_size=64):

    ix = torch.randint(len(data)-block_size, (batch_size,))
    xb = torch.concat([data[i:i+block_size] for i in ix])
    yb = torch.concat([data[i+1:i+block_size+1] for i in ix])
    return xb, yb


class Transformer(nn.Module):
    def __init__(self, vocab_size, hidden_size, num_layers, num_heads, dropout, block_size, device):
        super(Transformer, self).__init__()
        self.device = device
        self.block_size = block_size
        self.tok_emb = nn.Embedding(vocab_size, hidden_size)
        self.pos_emb = nn.Embedding(block_size, hidden_size)
        self.emb_drop = nn.Dropout(dropout)
        self.multihead_attn = MultiheadAttention(hidden_size, num_heads, dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(hidden_size)
        self.ff = FeedForward(hidden_size, dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.out = nn.Linear(hidden_size, vocab_size)


    def forward(self, x):
        tok_emb = self.tok_emb(x)
        pos_emb = self.pos_emb(torch.arange(self.block_size).to(self.device))
        emb = pos_emb + tok_emb

        return self.emb_drop(tok_emb + pos_emb)


class Attention:
    
    def __init__(self,hidden_size, num_heads, dropout):
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = self.head_dim ** 0.5
        self.qkv = nn.Linear(hidden_size, hidden_size*3)
        self.fc = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        pass

    
class MultiheadAttention:
    pass

class FeedForward:
    pass

