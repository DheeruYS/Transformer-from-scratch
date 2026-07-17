import torch
import torch.nn as nn
import numpy as np
import sentencepiece as spm
import tqdm
from torchtext.data import Field, BucketIterator, TabularDataset
from torchtext.data.metrics import bleu_score
from transformer import Transformer
from test import get_bleu
import os
import json
from test import get_bleu
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
print(torch.cuda.get_device_name(0))

with open("config.json", "r") as f:
    config = json.load(f)

pos_method = config["positional_embedding"]
decoding_strategy = config["decoding_strategy"]

train_tokenizer = False

if train_tokenizer :
    spm.SentencePieceTrainer.train(
        '--input=./EUbookshop/train.fi --model_prefix=finnish_bpe --vocab_size=32000 --model_type=bpe'
    )

    spm.SentencePieceTrainer.train(
        '--input=./EUbookshop/train.en --model_prefix=english_bpe --vocab_size=32000 --model_type=bpe'
    )

sp_fin = spm.SentencePieceProcessor()
sp_fin.load('finnish_bpe.model')

sp_eng = spm.SentencePieceProcessor()
sp_eng.load('english_bpe.model')

def tokenizer_bpe_fin(text):
    return sp_fin.encode_as_pieces(text)

def tokenizer_bpe_eng(text):
    return sp_eng.encode_as_pieces(text)

finnish_bpe = Field(
    tokenize=tokenizer_bpe_fin,
    lower=False,
    init_token='<sos>',
    eos_token='<eos>'
)

english_bpe = Field(
    tokenize=tokenizer_bpe_eng,
    lower=False,
    init_token='<sos>',
    eos_token='<eos>'
)

datafields = [("src", finnish_bpe), ("trg", english_bpe)]
train_data, valid_data, test_data = TabularDataset.splits(
    path="EUbookshop/",
    train="train.tsv",
    validation="valid.tsv",
    test="test.tsv",
    format="tsv",
    fields=datafields,
    skip_header=True
)

print(len(train_data), len(valid_data), len(test_data))

finnish_bpe.build_vocab(train_data)
english_bpe.build_vocab(train_data)

print(english_bpe.vocab.itos[:10])
print(english_bpe.vocab.stoi["man"])
print(vars(train_data.examples[0]))
print(len(finnish_bpe.vocab))

num_epochs = 20
batch_size = 32

src_vocab_size = len(finnish_bpe.vocab)
trg_vocab_size = len(english_bpe.vocab)
d_model = 512
heads = 8
num_layers = 3
d_ff = 2048
dropout = 0.15
max_seq_len = 256
src_pad_idx = english_bpe.vocab.stoi["<pad>"]

label_smoothing_epsilon = 0.1

train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
    (train_data,valid_data,test_data),
    batch_size=batch_size,
    device=device,
    sort_within_batch=True,
    sort_key = lambda x: len(x.src)
)

def get_lr(step, d_model, warmup_steps=4000):
    return d_model ** -0.5 * min(step ** -0.5, step * warmup_steps ** -1.5)

model = Transformer(src_vocab_size,trg_vocab_size,d_model,heads,d_ff,num_layers,num_layers,dropout,pos_method).to(device)

criterion = nn.CrossEntropyLoss(
    ignore_index=src_pad_idx,
    label_smoothing=label_smoothing_epsilon    
)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0,
    betas=(0.9, 0.98),
    eps=1e-9,
)

train_loss_curve = []
val_loss_curve = []

save_model = False
load_model = True

checkpoint_dir = "RoPE"

os.makedirs(checkpoint_dir, exist_ok=True)

if load_model:

    epoch_to_load = 2

    checkpoint_path = os.path.join(checkpoint_dir, f"epoch_{epoch_to_load}.pt")
    checkpoint = torch.load(checkpoint_path)

    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    train_loss_curve = checkpoint["train_loss_curve"]
    val_loss_curve = checkpoint["val_loss_curve"]

best_bleu_score = -float('inf')
step = 0

train = False
test = True
plot = False

if train :

    for epoch in tqdm.tqdm(range(num_epochs)):

        model.train()
        train_loss = 0

        for batch_idx,batch in enumerate(train_iterator):

            step += 1
            lr = get_lr(step, d_model, warmup_steps=4000)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

            input_data = batch.src.transpose(0,1).to(device) # [N, src_len]
            target = batch.trg.transpose(0,1).to(device) # [N, trg_len]

            output = model(input_data,target[:,:-1], src_pad_idx) # [N, trg_len, trg_vocab_size]

            output = output.reshape(-1,output.shape[2]) # Ignore the <sos> token, reshaped into [max seq_length * batch_size, target_vocab_size]
            target = target[:,1:].reshape(-1) # Ignore the <sos> token, reshaped into [max seq_length * batch_size] , the second dimension is removed, making it a list of indices / classes from the target vocabulary
            
            optimizer.zero_grad()
            loss = criterion(output,target) # mean of loss over all points
            
            loss.backward() # backprop

            torch.nn.utils.clip_grad_norm_(model.parameters(),max_norm=1)
            optimizer.step()

            train_loss += loss.item()

        model.eval()
        val_loss = 0

        with torch.no_grad():

            for batch_idx, batch in enumerate(valid_iterator):

                input_data =  batch.src.transpose(0,1).to(device)
                target = batch.trg.transpose(0,1).to(device)

                output = model(input_data,target[:,:-1],src_pad_idx)

                output = output.reshape(-1,output.shape[2])
                target = target[:,1:].reshape(-1)
                
                loss = criterion(output, target)
                val_loss += loss.item()

        cur_bleu = get_bleu(test_data[:100], model, finnish_bpe, english_bpe, tokenizer_bpe_fin, device)

        train_loss /= len(train_iterator)
        val_loss /= len(valid_iterator)

        train_loss_curve.append(train_loss)
        val_loss_curve.append(val_loss)

        print(f"Epoch: {epoch+1} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | BLEU: {cur_bleu * 100:.2f}")

        if cur_bleu > best_bleu_score :

            best_bleu_score = cur_bleu

            if save_model :

                checkpoint = {
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "train_loss_curve": train_loss_curve,
                    "val_loss_curve": val_loss_curve
                }

                checkpoint_path = os.path.join(checkpoint_dir, f"epoch_{epoch+1}.pt")
                torch.save(checkpoint, checkpoint_path)

if test:

    test_bleu_score = get_bleu(test_data[:10],model, finnish_bpe, english_bpe, tokenizer_bpe_fin, device)
    print(f"BLEU: {test_bleu_score * 100:.2f}")

if plot:
    plt.figure()
    plt.plot(train_loss_curve, label="Train Loss")
    plt.plot(val_loss_curve, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.savefig("loss_plot.png")
    plt.close()
    
