import torch
import torch.nn as nn
import torch.nn.functional as F
import time

with open("input.txt", "r") as f:
    text = f.read()

charset = list(sorted(list(set(text))))    
vocab_size = len(charset)
stoi = {c:i for i, c in enumerate(charset)}
itos = {i:c for i, c in enumerate(charset)}
encode = lambda x: [stoi[c] for c in x]
decode = lambda x: "".join([itos[idx] for idx in x])

#-------------------
train_split = 90
block_size = 256  #T
batch_size = 32 #B
vocab_size = len(charset) #C
learning_rate = 3e-4
num_iterations = int(2e3)
print_interval = 100
n_embd = 384
dropout = 0.2
n_heads = 6
n_layers = 6
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#--------------

data = torch.tensor(encode(text), dtype = torch.long)
split_len = int(len(data) * train_split / 100)
train = data[:split_len]
test = data[split_len:]


class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        logits = self.token_embedding_table(idx) # B, T, C
        if targets is None:
            loss = None
        else:
            C = logits.shape[-1]
            loss = F.cross_entropy(logits.view(-1, C), targets.view(-1))

        return logits, loss

    def generate(self, idx, max_new_tokens):
        for i in range(max_new_tokens):
            logits,_ = self(idx)
            logits = logits[:, -1, :]
            prob = F.softmax(logits, dim=-1)
            next_idx = torch.multinomial(prob, num_samples=1)
            idx = torch.concat((idx, next_idx), dim=1)
        return decode(idx.squeeze().tolist())
                

def generate_random_sample(split):
    data = train if split == 'train' else test
    rand_idx = torch.randint(len(data) - block_size, size=(batch_size,))
    x = torch.stack([data[idx: idx + block_size] for idx in rand_idx], dim=0)
    y = torch.stack([data[idx + 1: idx + block_size + 1] for idx in rand_idx], dim=0)
    return x.to(device), y.to(device)
    

class BigramAverageLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.pos_embedding_table = nn.Embedding(block_size, n_embd)
        self.lm_head = nn.Sequential(
            nn.Linear(n_embd, vocab_size, bias=False),
            nn.ReLU())
        
    def forward(self, idx, targets=None):
        B,T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.pos_embedding_table(torch.arange(T))
        logits = tok_emb + pos_emb
        logits = self.lm_head(logits)
        weights = torch.ones((T, T))
        mask = torch.tril(weights) == 0
        weights = weights.masked_fill(mask, -torch.inf)
        weights = torch.softmax(weights, dim=1)
        logits = (logits.transpose(1,2) @ weights.T).transpose(1,2)
        if targets is None:
            loss = None
        else:
            C = logits.shape[-1]
            loss = F.cross_entropy(logits.reshape(-1, C), targets.view(-1))
        return logits, loss
    
    def generate(self, idx, max_gen_len):
        for _ in range(max_gen_len):
            logits, _ = self(idx[:, -block_size:])
            probs = torch.softmax(logits[:, -1, :], dim=-1)
            next_idx = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_idx), dim=-1)
        return decode(idx.tolist()[0])
    
class Head(nn.Module):
    def __init__(self, n_embd, head_size):
        super().__init__()
        self.head_size = head_size
        self.qkv = nn.Linear(n_embd, 3*head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size, device=device)))
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        T = x.shape[1]
        q,k,v = self.qkv(x).split(self.head_size, dim=-1)
        wei = q @ k.transpose(1,2) / torch.sqrt(torch.tensor(k.shape[-1], device=device))
        mask = self.tril[:T, :T] == 0
        wei = torch.softmax(wei.masked_fill(mask, -torch.inf), dim=-1)
        out = wei @ v
        return self.dropout(out)

class MultiHead(nn.Module):
    def __init__(self, n_heads,head_size):
        super().__init__()
        self.heads = nn.ModuleList(Head(n_embd, head_size) for _ in range(n_heads))
        self.linear = nn.Linear(n_heads * head_size, n_embd)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.linear(out)
        return self.dropout(out)
        
class FFN(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4*n_embd),
            nn.ReLU(),
            nn.Linear(4*n_embd, n_embd),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        return self.net(x)
    
class Block(nn.Module):
    def __init__(self, n_heads, n_embd):
        super().__init__()
        self.ma = MultiHead(n_heads, n_embd // n_heads)
        self.ffn = FFN(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        
    def forward(self, x):
        x = x + self.ma(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x
    
class GPTModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.pos_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_heads, n_embd) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)
        
    def forward(self, idx, targets=None):
        _,T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.pos_embedding_table(torch.arange(T, device=device))
        emb = tok_emb + pos_emb
        x = self.blocks(emb)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        if targets is None:
            loss = None
        else:
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))
        return logits, loss
    
    @torch.no_grad()
    def generate(self, idx, max_gen_len):
        self.eval()
        for _ in range(max_gen_len):
            logits, _ = self(idx[:, -block_size:])
            probs = torch.softmax(logits[:, -1, :], dim=-1)
            next_idx = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_idx), dim=-1)
        return decode(idx.detach().cpu().tolist()[0])
        

class LLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.pos_embedding_table = nn.Embedding(block_size, n_embd)
        self.head = Head(n_embd, n_embd)
        self.proj = nn.Linear(n_embd, vocab_size, bias=False)
        
    def forward(self, idx, targets=None):
        T = idx.shape[-1]
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.pos_embedding_table(torch.arange(T))
        emb = tok_emb + pos_emb
        out = self.head(emb)
        logits = self.proj(out)
        if targets is None:
            loss = None
        else:
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))
        return logits, loss
    
    def generate(self, idx, max_gen_len):
        for _ in range(max_gen_len):
            logits, _ = self(idx[:, -block_size:])
            probs = torch.softmax(logits[:, -1, :], dim=-1)
            next_idx = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_idx), dim=-1)
        return decode(idx.tolist()[0])

def train_model(model_type):
    if model_type == 'simple':
        model = BigramLanguageModel()
        model_path = 'simple_best_model.pt'
    elif model_type == 'average':
        model = BigramAverageLanguageModel()
        model_path = 'average_best_model.pt'
    elif model_type == 'llm':
        model = LLM()
        model_path = 'llm_head_best_model.pt'
    elif model_type == 'gpt':
        model = GPTModel().to(device)
        model_path = 'gpt_head_best_model.pt'
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    best_loss = None
    for i in range(num_iterations):
        x, y = generate_random_sample("train")
        model.train()
        _, loss = model(x, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        model.eval()
        x_val, y_val = generate_random_sample("test")
        _, val_loss = model(x_val, y_val)
        if (i+1) % print_interval == 0:
            print(f"iteraction: {i+1}, train_loss: {loss}, val_loss: {val_loss}")
        # if best_loss is None or loss.item() < best_loss:
        #     torch.save(model.state_dict(), model_path)
        #     best_loss = loss.item()

    # model.load_state_dict(torch.load(model_path))
    torch.save(model.state_dict(), model_path)
    print(model.generate(torch.zeros((1,1), dtype=torch.long, device=device),1000))


if __name__ == "__main__":
    x, y = generate_random_sample("train")
    # model = GPTModel()
    # logits, loss = model(x, y)
    # print(logits.shape)
    # print(loss.shape)
    # print(loss)
    # print(x.shape)
    # print(y.shape)
    # model = BigramAverageLanguageModel()
    
    # logits, loss = model(x, y)
    # print(logits.shape)
    # print(loss)
    # print(model.generate(x[0].view(1,-1), 100))
    # st = time.time()
    # train_model('gpt')
    # et = time.time()
    # print(et - st)    
    
    # model = GPTModel().to(device)
    # model.load_state_dict(torch.load("gpt_head_best_model.pt", map_location=device))
    # model.eval()
    # # model.eval()
    # logits, loss = model(x, y)
    # # print(logits.shape)
    # # print(loss.shape)
    # print(loss)
    # print(model.generate(torch.zeros((1,1), dtype=torch.long, device=device), 1000))