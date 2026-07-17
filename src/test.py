import torch
import random
from torchtext.data.metrics import bleu_score
import tqdm
import json

with open("config.json", "r") as f:
    config = json.load(f)

pos_method = config["positional_embedding"]
decoding_strategy = config["decoding_strategy"]

def greedy_decode(model, sentence_tensor, finnish, english, device, src_pad_idx, max_seq_len=256):
    output_tokens = [english.vocab.stoi["<sos>"]]
    for _ in range(max_seq_len):
        trg_tensor = torch.LongTensor(output_tokens).unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(sentence_tensor, trg_tensor, src_pad_idx)
        output = output.squeeze(0)
        next_token = output[-1].argmax(dim=-1).item()
        output_tokens.append(next_token)
        if next_token == english.vocab.stoi["<eos>"]:
            break
    translated_sentence = [english.vocab.itos[idx] for idx in output_tokens]
    return translated_sentence[1:]

def beam_search_decode(model, sentence_tensor, finnish, english, device,
                       src_pad_idx, max_seq_len=256, beam_width=5, alpha=0.6):
    sos_idx = english.vocab.stoi["<sos>"]
    eos_idx = english.vocab.stoi["<eos>"]

    beams = [([sos_idx], 0.0)]  # (tokens, log_prob)
    completed = []

    for _ in range(max_seq_len):
        new_beams = []
        for tokens, score in beams:
            if tokens[-1] == eos_idx:
                completed.append((tokens, score))
                continue

            trg_tensor = torch.LongTensor(tokens).unsqueeze(0).to(device)
            with torch.no_grad():
                output = model(sentence_tensor, trg_tensor, src_pad_idx)

            log_probs = torch.log_softmax(output.squeeze(0)[-1], dim=-1)
            topk_log_probs, topk_indices = torch.topk(log_probs, beam_width)

            for log_prob, idx in zip(topk_log_probs, topk_indices):
                new_tokens = tokens + [idx.item()]
                new_score = score + log_prob.item()
                new_beams.append((new_tokens, new_score))

        beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[:beam_width]
        if not beams:
            break

    completed += beams

    # length-normalize
    def length_norm(tokens, log_prob):
        length = len(tokens)
        return log_prob / ((5 + length) ** alpha / (5 + 1) ** alpha)

    best_tokens, best_score = max(completed, key=lambda x: length_norm(*x))
    translated_sentence = [english.vocab.itos[idx] for idx in best_tokens]
    return translated_sentence[1:]  # skip <sos>


def topk_sampling_decode(model, sentence_tensor, finnish, english, device, src_pad_idx, max_seq_len=256, k=5):
    sos_idx = english.vocab.stoi["<sos>"]
    eos_idx = english.vocab.stoi["<eos>"]
    output_tokens = [sos_idx]
    for _ in range(max_seq_len):
        trg_tensor = torch.LongTensor(output_tokens).unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(sentence_tensor, trg_tensor, src_pad_idx)
        output = output.squeeze(0)[-1]
        probs = torch.softmax(output, dim=-1)
        topk_probs, topk_indices = torch.topk(probs, k)
        topk_probs = topk_probs.cpu().numpy()
        topk_indices = topk_indices.cpu().numpy()
        next_token = random.choices(topk_indices, weights=topk_probs, k=1)[0]
        output_tokens.append(int(next_token))
        if next_token == eos_idx:
            break
    translated_sentence = [english.vocab.itos[idx] for idx in output_tokens]
    return translated_sentence[1:]

def translate_sentence(model, sentence, finnish, english, device, tokenizer_bpe_fin, max_seq_len=256, beam_width=5, k=5):
    
    if type(sentence) == str:
        tokens = tokenizer_bpe_fin(sentence)
    else:
        tokens = sentence

    src_pad_idx = english.vocab.stoi["<pad>"]

    tokens.insert(0, finnish.init_token)
    tokens.append(finnish.eos_token)

    text_to_indices = [finnish.vocab.stoi[token] for token in tokens]
    sentence_tensor = torch.LongTensor(text_to_indices).unsqueeze(0).to(device)

    model.eval()
    
    if decoding_strategy == 'greedy':
        return greedy_decode(model, sentence_tensor, finnish, english, device, src_pad_idx, max_seq_len)
    elif decoding_strategy == 'beam':
        return beam_search_decode(model, sentence_tensor, finnish, english, device, src_pad_idx, max_seq_len, beam_width)
    else:
        return topk_sampling_decode(model, sentence_tensor, finnish, english, device, src_pad_idx, max_seq_len, k)

def get_bleu(data, model, finnish, english, tokenizer_bpe_fin, device):

    targets = []
    outputs = []

    for example in tqdm.tqdm(data):
        src = vars(example)["src"]
        trg = vars(example)["trg"]

        prediction = translate_sentence(model, src, finnish, english, device, tokenizer_bpe_fin)
        prediction = prediction[:-1]  # remove <eos> token

        targets.append([trg])
        outputs.append(prediction)

    return bleu_score(outputs, targets)