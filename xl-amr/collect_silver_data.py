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
    translated = model.generate(**tokenized)
    sents = [tokenizer.decode(t, skip_special_tokens=True) for t in translated]
    return sents

lang = sys.argv[1]
batch_size = int(sys.argv[2])
tokenizer, model = get_mt_model(lang)


def read_snts(fname):
    snts = []
    for line in open(fname).readlines():
        if line.startswith("# ::snt"):
            snts.append(line[len("# ::snt"):].strip())
    return snts

for split in ['train', 'dev']:
    all_translations = []
    batch = []
    for x in tqdm.tqdm(read_snts(f'data/AMR/amr_2.0/{split}.txt')):
        batch.append(x)
        if len(batch) == batch_size:
            translations = translate(tokenizer, model, batch)
            all_translations.extend(translations)
            batch = []
        else:
            continue
    if batch:
        translations = translate(tokenizer, model, batch)
        all_translations.extend(translations)
    
    idx = 0
    with open(f'{split}_{lang}.txt', "w") as fo:
        for line in open(f'data/AMR/amr_2.0/{split}.txt'):
            if line.startswith("# ::snt"):
                fo.write(f"# ::snt {all_translations[idx]}\n")
                idx += 1
            else:
                fo.write(line)
    assert idx == len(all_translations)



