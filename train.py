# train.py
from triton_transformer import Transformer
from triton_transformer.autoregressive_wrapper import AutoregressiveWrapper

import random
import tqdm
import gzip
import os
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

# =====================
# constants
# =====================
NUM_BATCHES = int(1e5)
BATCH_SIZE = 4
GRADIENT_ACCUMULATE_EVERY = 4
LEARNING_RATE = 2e-4
VALIDATE_EVERY  = 100
GENERATE_EVERY  = 500
GENERATE_LENGTH = 512
SEQ_LEN = 512

SAVE_EVERY = 5000
SAVE_DIR = "./checkpoints"
os.makedirs(SAVE_DIR, exist_ok=True)

# =====================
# helpers
# =====================
def cycle(loader):
    while True:
        for data in loader:
            yield data

def decode_token(token):
    return str(chr(max(32, token)))

def decode_tokens(tokens):
    return ''.join(list(map(decode_token, tokens)))

# =====================
# model
# =====================
model = Transformer(
    num_tokens = 256,
    dim = 512,
    max_seq_len = SEQ_LEN,
    depth = 8,
    heads = 8,
    causal = True,
    use_triton = True,
    attn_dropout = 0.1,
    ff_dropout = 0.1,
)

model = AutoregressiveWrapper(model)
model.cuda()

# =====================
# dataset
# =====================
with gzip.open('./data/enwik8.gz') as file:
    buf = file.read(int(95e6))
    X = np.frombuffer(buf, dtype=np.uint8)
    trX, vaX = np.split(X, [int(90e6)])
    data_train, data_val = torch.from_numpy(trX.copy()), torch.from_numpy(vaX.copy())

class TextSamplerDataset(Dataset):
    def __init__(self, data, seq_len):
        super().__init__()
        self.data = data
        self.seq_len = seq_len

    def __getitem__(self, index):
        rand_start = torch.randint(0, self.data.size(0) - self.seq_len, (1,))
        full_seq = self.data[rand_start: rand_start + self.seq_len + 1].long()
        return full_seq.cuda()

    def __len__(self):
        return self.data.size(0) // self.seq_len

train_dataset = TextSamplerDataset(data_train, SEQ_LEN)
val_dataset   = TextSamplerDataset(data_val, SEQ_LEN)
train_loader  = cycle(DataLoader(train_dataset, batch_size = BATCH_SIZE))
val_loader    = cycle(DataLoader(val_dataset, batch_size = BATCH_SIZE))

# =====================
# optimizer
# =====================
optim = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# =====================
# checkpoint restore
# =====================
start_step = 0
latest_ckpt = None
if os.path.exists(SAVE_DIR):
    ckpts = [f for f in os.listdir(SAVE_DIR) if f.endswith(".pt")]
    if ckpts:
        latest_ckpt = os.path.join(SAVE_DIR, sorted(ckpts, key=lambda x: int(x.split("_")[2].split(".")[0]))[-1])

if latest_ckpt:
    print(f"🔄 Restoring from {latest_ckpt}")
    ckpt = torch.load(latest_ckpt)
    model.load_state_dict(ckpt['model_state_dict'])
    optim.load_state_dict(ckpt['optimizer_state_dict'])
    start_step = ckpt['step'] + 1
    print(f"Resumed from step {start_step}, last loss {ckpt['loss']:.4f}")

# =====================
# training loop
# =====================
for i in tqdm.tqdm(range(start_step, NUM_BATCHES), mininterval=10., desc='training'):
    model.train()

    for __ in range(GRADIENT_ACCUMULATE_EVERY):
        loss = model(next(train_loader))
        loss.backward()

    if i % 100 == 0:
        print(f'step {i} training loss: {loss.item():.4f}')

    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
    optim.step()
    optim.zero_grad()

    # validation
    if i % VALIDATE_EVERY == 0:
        model.eval()
        with torch.no_grad():
            val_loss = model(next(val_loader))
            print(f'✅ validation loss: {val_loss.item():.4f}')

    # text generation
    if i % GENERATE_EVERY == 0:
        model.eval()
        inp = random.choice(val_dataset)[:-1]
        prime = decode_tokens(inp)
        print(f'\n=== Prompt ===\n{prime}\n{"*"*50}')

        sample = model.generate(inp[None, ...], GENERATE_LENGTH)
        output_str = decode_tokens(sample[0])
        print(f'=== Generated ===\n{output_str}\n')

    # save checkpoint
    if i % SAVE_EVERY == 0 and i > 0:
        ckpt_path = os.path.join(SAVE_DIR, f"model_step_{i}.pt")
        torch.save({
            'step': i,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optim.state_dict(),
            'loss': loss.item(),
        }, ckpt_path)
        print(f"💾 Saved checkpoint: {ckpt_path}")