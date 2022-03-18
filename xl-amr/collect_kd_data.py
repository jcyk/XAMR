from datasets import load_dataset
import sys
import torch
import tqdm

device = torch.device('cuda')
def get_mt_model(lang):
    from transformers import MarianMTModel, MarianTokenizer
    tokenizer = MarianTokenizer.from_pretrained(f"Helsinki-NLP/opus-mt-en-{lang}")
    model = MarianMTModel.from_pretrained(f"Helsinki-NLP/opus-mt-en-{lang}").to(device)
    return tokenizer, model
def translate(tokenizer, model, sents):
    tokenized = tokenizer(sents, return_tensors="pt", padding=True).to(device)
    print (tokenized['input_ids'].size())
    translated = model.generate(**tokenized)
    sents = [tokenizer.decode(t, skip_special_tokens=True) for t in translated]
    return sents

lang = sys.argv[1]
batch_size = int(sys.argv[2])
tokenizer, model = get_mt_model(lang)
idx = 0
print ("start!")

sentences = []
for line in open(f'{lang}.txt').readlines():
    if len(line.strip().split()) > 128:
        continue
    else:
        sentences.append(line.strip())

import random
random.seed(199401170033)
random.shuffle(sentences)

sentences = sentences[:320000]

with open(f'kd_{lang}_en.txt', "w") as fo:
    batch = []
    for x in tqdm.tqdm(sentences):
        batch.append(x)
        if len(batch) == batch_size:
            translations = translate(tokenizer, model, batch)
            for source, translation in zip(batch, translations):
                fo.write(f"# ::id {idx}\n")
                fo.write(f"# ::snt {source}\n")
                fo.write(f"# ::tok-{lang} {translation}\n")
                fo.write(f"(e / empty)\n\n")
                idx += 1
            batch = []
        else:
            continue
    if batch:
        translations = translate(tokenizer, model, batch)
        for source, translation in zip(batch, translations):
            fo.write(f"# ::id {idx}\n")
            fo.write(f"# ::snt {source}\n")
            fo.write(f"# ::tok-{lang} {translation}\n")
            fo.write(f"(e / empty)\n\n")
            idx += 1


