# XAMR
Code for our EMNLP 2021 (findings) paper [Multilingual AMR Parsing with Noisy Knowledge Distillation](https://arxiv.org/pdf/2109.15196.pdf).

We develop **one** multilingual AMR parser to parse **five** different languages including German (DE), Spanish (ES), Italian (IS), Chinese (ZH), and English (EN).

The parsing performance of our best parser is shown below.

| Language   | DE   | ES   | IT   | ZH   | EN   |
| :--------- | ---- | ---- | ---- | ---- | ---- |
| Smatch (%) | 73.1 | 75.9 | 75.4 | 61.9 | 83.9 |

## Requirements

The code has been tested on **Python 3.6**. All dependencies are listed in [requirements.txt](requirements.txt).

## Multilingual AMR Parsing with the Pretrained Model

The pretrained model can be downloaded from [google drive]().

See `scripts/work.sh` for evaluation.

## Train New Parsers

coming soon

## Acknowledgements

This project is based on [SPRING](https://github.com/SapienzaNLP/spring).

## Contact

For any questions, please drop an email to [Deng Cai](https://jcyk.github.io/).
