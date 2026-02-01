import torch
import torch.nn as nn
from torch.nn import functional as F

# Hyperparamters-----
batch_size = 32
block_size = 8
device = "cuda" if torch.cuda.is_available() else "cpu"
max_iters = 3000
eval_interval = 300
lr = 1e-3
eval_iters = 200
#---------------------

torch.manual_seed(1337)

# Read input file
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# Get all unique characters to form our vocab
chars = sorted(list(set(text)))
vocab_size = len(chars)

# Create character to int mapping
stoi = {ch:i for i,ch in enumerate(chars)}
itos = {i:ch for i,ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]         # Takes a string and converts it into a list of integers
decode = lambda l: ''.join(itos[i] for i in l)  # Takes a list of integers and creates an appended string

# Train-test split
data = torch.tensor(encode(text), dtype=torch.long)
train_percentage = 90 # Percentage of data that goes into training
train_size = 0.01*train_percentage*len(data)
train_data = data[:train_size]
val_data = data[train_size:]

def get_batch(split : str):
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    xb = torch.stack([data[i:i+block_size] for i in ix])
    yb = torch.stack([data[i+1:i+block_size+1] for i in ix])
    xb, yb = xb.to(device), yb.to(device)
    #Note that the batch loader outputs are now on the GPU if it's there
    return xb, yb

class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)
        
    def forward(self, idx, target = None):
        logits = self.token_embedding_table(idx)
        if target is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T,C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            logits, _ = self(idx)
            logits = logits[:,-1,:]
            p = F.softmax(p, dim=-1)
            idx_next = torch.multinomial(p, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)

        return idx
    
model = BigramLanguageModel(vocab_size)
model = model.to(device) # Send model to GPU if it's there
optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)

# Training loop
#---------------------
for iter in range(max_iters):
    xb, yb = get_batch("train")
    logits, loss = model(xb, yb)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

context = "Hello"
input = torch.tensor(encode(context)).unsqueeze(0).to(device)
print(decode(encode(context)))
