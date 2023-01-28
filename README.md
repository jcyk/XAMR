# XAMR
Code for our EMNLP 2021 (findings) paper [Multilingual AMR Parsing with Noisy Knowledge Distillation](https://arxiv.org/pdf/2109.15196.pdf).

We develop **one** multilingual AMR parser to parse **five** different languages including German (DE), Spanish (ES), Italian (IS), Chinese (ZH), and English (EN).

The parsing performance of our best parser [(Google Drive)](https://drive.google.com/file/d/1trLOWMAjKe4VpOExfB6AJyh7-qhxItEs/view?usp=sharing) is shown below.

| Language   | DE   | ES   | IT   | ZH   | EN   |
| :--------- | ---- | ---- | ---- | ---- | ---- |
| Smatch (%) | 73.1 | 76.2 | 75.8 | 62.2 | 84.2 |

(some numbers are ***higher*** than those reported in our paper due to different runs)

## Requirements

The code has been tested on **Python 3.6**. All dependencies are listed in [requirements.txt](requirements.txt).

## Multilingual AMR Parsing with the Pretrained Model

The pretrained model can be downloaded from [Google Drive](https://drive.google.com/file/d/1trLOWMAjKe4VpOExfB6AJyh7-qhxItEs/view?usp=sharing).

See `scripts/work.sh` for evaluation.

To parse your own data, use the following command:
```bash
checkpoint=ckpt/best.pt # this points to the pretrained model you have downloaded
dataset=tmp.txt # this points to the data you want to parse (see detailed explanation below)

PYTHONPATH=. python3 bin/predict_amrs.py \
   --model facebook/mbart-large-50-many-to-many-mmt \
   --checkpoint ${checkpoint} \
   --dataset ${dataset} \
   --nproc-per-node 4 \
   --gold-path tmp-gold.txt \
   --pred-path tmp-pred.txt \
   --beam-size 4 \
   --batch-size 5000 \
   --penman-linearization \
   --use-pointer-tokens
```
the `tmp.txt` file looks like below:
```
# ::id 0
# ::snt Resolutely support the thread starter! I compose a poem in reply:
# ::snt_lang en
(z0 / and)

# ::id 1
# ::snt Ich unterstütze denjenigen, der diesen Thread gestartet hat, ganz deutlich! Ich habe ein Gedicht als Antwort verfasst:
# ::snt_lang de
(z0 / and)

# ::id 2
# ::snt ¡Respalde firmemente el inicio del hilo! Escribo un poema en respuesta:
# ::snt_lang es
(z0 / and)

# ::id 3
# ::snt Sostenete assolutamente chi ha avviato questo thread! Scrivo una poesia come risposta:
# ::snt_lang it
(z0 / and)

# ::id 4
# ::snt 坚决支持楼主！我赋诗一首，以表寸心：
# ::snt_lang zh
(z0 / and)
```
As seen, each block has four fields:
- id: the sentence id
- snt: the input sentence
- snt_lang: the langauge of the input sentence (choosing from en, de, es, it, zh, ...)
- (z0/ and): this is just a placeholder

## Train New Parsers

### Data Preparation
follow the instructions in the readme in the `xlamr` folder.

### Training


use `train.f3.sh` and `train.f4.sh` in the `scripts` folder.

(loading the KD data can be very time-consuming, use `prepare.sh` to cache the data)


## Acknowledgements

This project is based on [SPRING](https://github.com/SapienzaNLP/spring) and [xl-amr](https://github.com/SapienzaNLP/xl-amr).

## Contact

For any questions, please drop an email to [Deng Cai](https://jcyk.github.io/).
