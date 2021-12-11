from datasets import load_dataset
dataset = load_dataset("europarl_bilingual", lang1="en", lang2="es")
with open("europarl.enes.en.txt", 'w') as fo0, open("europarl.enes.es.txt", 'w') as fo1:
    for pair in dataset['train']:
        pair = pair['translation']
        fo0.write(pair['en'] + '\n')
        fo1.write(pair['es'] + '\n')
dataset = load_dataset("europarl_bilingual", lang1="de", lang2="en")
with open("europarl.deen.de.txt", 'w') as fo0, open("europarl.deen.en.txt", 'w') as fo1:
    for pair in dataset['train']:
        pair = pair['translation']
        fo0.write(pair['de'] + '\n')
        fo1.write(pair['en'] + '\n')
dataset = load_dataset("europarl_bilingual", lang1="en", lang2="it")
with open("europarl.enit.en.txt", 'w') as fo0, open("europarl.enit.it.txt", 'w') as fo1:
    for pair in dataset['train']:
        pair = pair['translation']
        fo0.write(pair['en'] + '\n')
        fo1.write(pair['it'] + '\n')
