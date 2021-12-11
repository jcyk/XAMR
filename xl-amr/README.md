## Acknowledgement

The preprocessing code is largely borrowed from [xl-amr](https://github.com/SapienzaNLP/xl-amr)

## 1. Install 

Create a conda environment with **Python 3.6** and [requirements.txt](requirements.txt).

## 2. Gold Dataset
1 - Download AMR 2.0 ([LDC2017T10](https://catalog.ldc.upenn.edu/LDC2017T10)) and AMR 2.0 - Four Translations ([LDC2020T07](https://catalog.ldc.upenn.edu/LDC2020T07)).

2 - Unzip the AMR 2.0 corpus to `data/LDC2017T10`. It should look like this:

    (xlamr)$ tree data/LDC2017T10 -L 2
    ├── data
    │   ├── alignments
    │   ├── amrs
    │   └── frames
    ├── docs
    │   ├── AMR-alignment-format.txt
    │   ├── amr-guidelines-v1.2.pdf
    │   ├── file.tbl
    │   ├── frameset.dtd
    │   ├── PropBank-unification-notes.txt
    │   └── README.txt
    └── index.html
    
Prepare training/dev/test data:

    ./scripts/prepare_data.sh -v 2 -p data/LDC2017T10
    
3 - Unzip the Translations corpus to copy ```*.txt``` files into ```data/AMR/amr_2.0/translations```.

Project English test AMR graphs across languages:

    ./scripts/project_test.sh 

    
## 3. Knowledge Distillation Data

1. download it from [Google Drive A](https://drive.google.com/file/d/13GzzXvisjTU09HqgrRY9ZMT1drMMVLY3/view?usp=sharing) and the translations from [Google Drive B](https://drive.google.com/file/d/1AWh5HcyhOgSV4a6fHNV6M06eLwSWOknl/view?usp=sharing)  

2. parse *.translated.txt using SPRING

 



## 4. Silver Data 
We machine translated sentences of AMR 2.0 using [OPUS-MT](https://huggingface.co/transformers/model_doc/marian.html) pretrained models and filtered less accurate translations. The translated sentences are found in the following folder: 

    cd data/AMR/amr_2.0_zh_de_es_it/translations/

To respect the LDC agreement for AMR 2.0, we release the translations without the gold graphs. Therefore to project the AMR graphs from AMR 2.0 run:

    ./scripts/project_train_dev.sh

