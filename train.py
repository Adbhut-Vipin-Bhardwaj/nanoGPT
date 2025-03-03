import os
import torch
import torch.nn as nn
import torch.nn.functional as F
torch.manual_seed(42)

from nano_gpt import nanoGPT


ctxt_len = 256
batch_size = 256
n_embed = 384
num_layers = 8
num_heads = 6
dropout = 0.2
device = 'cuda' if torch.cuda.is_available() else 'cpu'
max_iters = 5000
eval_interval = 100
eval_iters = 200 # how many batches to eval on in one evaluation
learning_rate = 5e-4
model_dir = "./models/"
model_filename = "nanoGPT"

with open("./datasets/tiny_shakespeare/input.txt", "r") as f:
    text = f.read()


chars = sorted(list(set(text)))
vocab_size = len(chars)
print(f"Vocab size: {vocab_size}")

ctoi = {c:i for i, c in enumerate(chars)}
itoc = {i:c for i, c in enumerate(chars)}

def encode(s):
    return [ctoi[c] for c in s]

def decode(token_seq):
    return "".join([itoc[i] for i in token_seq])

tokenized_text = encode(text)

train_data_size = int(0.9*len(tokenized_text))
val_data_size = len(tokenized_text) - train_data_size
print(f"Number of tokens in train data: {train_data_size}")
print(f"Number of tokens in val data: {val_data_size}")

train_data = torch.tensor(
    tokenized_text[:train_data_size],
    dtype=torch.long,
)
val_data = torch.tensor(
    tokenized_text[train_data_size:],
    dtype=torch.long,
)


def get_batch(split):
    data = train_data if split=='train' else val_data
    start_idxs = torch.randint(0, len(data)-ctxt_len, (batch_size, ))
    x = torch.stack([data[idx:idx+ctxt_len] for idx in start_idxs])
    y = torch.stack([data[idx+1:idx+1+ctxt_len] for idx in start_idxs])
    x, y = x.to(device), y.to(device)
    return x, y

llm = nanoGPT(vocab_size, ctxt_len, n_embed, num_heads, num_layers, dropout, device)
llm.to(device)

## Generated tokens before train
x = torch.tensor([0], device=device).reshape(1, 1)
generated_tokens = llm.generate(
    x, max_new_tokens=500
)
generated_text = decode(generated_tokens[0].tolist())
print(f"Text generated before training: {generated_text}")


@torch.no_grad()
def estimate_loss():
    losses = {}
    llm.eval()
    for split in ['train', 'val']:
        running_loss = 0;
        for i in range(eval_iters):
            x, y = get_batch(split)
            logits = llm(x)
            loss = llm.calc_loss(logits, y)
            running_loss += loss.item()
        losses[split] = running_loss / eval_iters
    llm.train()

    return losses

optimizer = torch.optim.AdamW(llm.parameters(), lr=learning_rate)

best_loss = torch.inf
for iter in range(max_iters):
    print(f"Train iter: {iter}")
    if iter%eval_interval == 0 or iter == max_iters-1:
        losses = estimate_loss()
        if losses['val'] < best_loss:
            os.makedirs(model_dir, exist_ok=True)
            torch.save(
                llm.state_dict(), os.path.join(model_dir, model_filename)
            )
        print(
            f"step {iter}: train loss {losses['train']:.4f}"
            + f", val loss {losses['val']:.4f}"
        )

    x, y = get_batch('train')

    logits = llm(x)
    loss = llm.calc_loss(logits, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

x = torch.tensor([0], device=device).reshape(1, 1)
generated_tokens = llm.generate(
    x, max_new_tokens=500
)
generated_text = decode(generated_tokens[0].tolist())
print(f"Text generated after training: {generated_text}")
