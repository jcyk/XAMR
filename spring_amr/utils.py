from glob import glob
from pathlib import Path

import os, torch
from transformers import AutoConfig, AutoModelForSeq2SeqLM
from spring_amr.modeling import MyMBartForConditionalGeneration
from spring_amr.dataset import AMRDataset, AMRDatasetTokenBatcherAndLoader
from spring_amr.tokenization_bart import AMRBartTokenizer, PENMANBartTokenizer
from spring_amr.tokenization_mbart50 import AMRMBart50Tokenizer, PENMANMBart50Tokenizer

def instantiate_model_and_tokenizer(
        name=None,
        checkpoint=None,
        additional_tokens_smart_init=True,
        dropout = 0.15,
        attention_dropout = 0.15,
        from_pretrained = True,
        collapse_name_ops = False,
        penman_linearization = False,
        use_pointer_tokens = False,
        raw_graph = False,
        my_model = False,
):
    if raw_graph:
        assert penman_linearization

    skip_relations = False

    if name is None:
        name = 'facebook/bart-large'

    if name.startswith('facebook/mbart'):
        AMRTokenizer, PENMANTokenizer = AMRMBart50Tokenizer, PENMANMBart50Tokenizer
    else:
        AMRTokenizer, PENMANTokenizer = AMRBartTokenizer, PENMANBartTokenizer
     
    if os.path.isdir('/apdcephfs/share_916081/jcykcai/' + name):
        name = '/apdcephfs/share_916081/jcykcai/' + name

    config = AutoConfig.from_pretrained(name)
    config.output_past = False
    config.no_repeat_ngram_size = 0
    config.prefix = " "
    config.output_attentions = True
    config.dropout = dropout
    config.attention_dropout = attention_dropout
    config.return_dict = False
    config.forced_bos_token_id = None

    if penman_linearization:
        tokenizer = PENMANTokenizer.from_pretrained(
            name,
            collapse_name_ops=collapse_name_ops,
            use_pointer_tokens=use_pointer_tokens,
            raw_graph=raw_graph,
        )
    else:
        tokenizer = AMRTokenizer.from_pretrained(
            name,
            collapse_name_ops=collapse_name_ops,
            use_pointer_tokens=use_pointer_tokens,
        )

    ModelFactory = MyMBartForConditionalGeneration if my_model else AutoModelForSeq2SeqLM
    if from_pretrained:
        model = ModelFactory.from_pretrained(name, config=config)
    else:
        model = ModelFactory.from_config(config)

    model.resize_token_embeddings(tokenizer.vocab_size)

    if additional_tokens_smart_init:
        modified = 0
        vocab = tokenizer.get_vocab()
        for tok, idx in vocab.items():
            tok = tok.lstrip(tokenizer.INIT)

            if idx < tokenizer.old_enc_size:
                continue

            elif tok.startswith('<pointer:') and tok.endswith('>'):
                tok_split = ['pointer', str(tok.split(':')[1].strip('>'))]

            elif tok.startswith('<'):
                continue

            elif tok.startswith(':'):

                if skip_relations:
                    continue

                elif tok.startswith(':op'):
                    tok_split = ['relation', 'operator', str(int(tok[3:]))]

                elif tok.startswith(':snt'):
                    tok_split = ['relation', 'sentence', str(int(tok[4:]))]

                elif tok.startswith(':ARG'):
                    tok_split = ['relation', 'argument', str(int(tok[4:]))]

                else:
                    tok_split = ['relation'] + tok.lstrip(':').split('-')

            else:
                tok_split = tok.split('-')

            tok_split_ = tok_split
            tok_split = []
            for s in tok_split_:
                s_ = s + tokenizer.INIT
                if s_ in vocab:
                    tok_split.append(s_)
                else:
                    tok_split.extend(tokenizer._tok_bpe(s))

            vecs = []
            for s in tok_split:
                idx_split = vocab.get(s, -1)
                if idx_split > -1:
                    vec_split = model.model.shared.weight.data[idx_split].clone()
                    vecs.append(vec_split)

            if vecs:
                vec = torch.stack(vecs, 0).mean(0)
                noise = torch.empty_like(vec)
                noise.uniform_(-0.1, +0.1)
                model.model.shared.weight.data[idx] = vec + noise
                modified += 1

    if checkpoint is not None:
        model.load_state_dict(torch.load(checkpoint, map_location='cpu')['model'])

    return model, tokenizer


def instantiate_loader(
        glob_pattn,
        tokenizer,
        batch_size=500,
        evaluation=True,
        out=None,
        use_recategorization=False,
        remove_longer_than=None,
        remove_wiki=False,
        dereify=True,
        rank=0,
        world_size=1,
        cached=False,
        max_cached_samples=None,
        noise=0.,
):
    paths = []
    if isinstance(glob_pattn, str) or isinstance(glob_pattn, Path):
        glob_pattn = [glob_pattn]
    for gpattn in glob_pattn:
        paths += [Path(p) for p in glob(str(gpattn))]

    paths.sort()
    if cached:
        return instantiate_loader_from_cached(paths, tokenizer, batch_size=batch_size, evaluation=evaluation, out=out, use_recategorization=use_recategorization, remove_longer_than=remove_longer_than, remove_wiki=remove_wiki, dereify=dereify, rank=rank, world_size=world_size, max_cached_samples=max_cached_samples, noise=noise) 
    if out is not None:
        Path(out).write_text(
            '\n\n'.join([p.read_text() for p in paths]))
    dataset = AMRDataset(
        paths,
        tokenizer,
        use_recategorization=use_recategorization,
        remove_longer_than=remove_longer_than,
        remove_wiki=remove_wiki,
        dereify=dereify,
        evaluation=evaluation,
        rank=rank,
        world_size=world_size,
        noise=noise,
    )
    loader = AMRDatasetTokenBatcherAndLoader(
        dataset,
        batch_size=batch_size,
        shuffle=not evaluation,
    )
    return loader

def instantiate_loader_from_cached(
        paths,
        tokenizer,
        batch_size=500,
        evaluation=True,
        out=None,
        use_recategorization=False,
        remove_longer_than=None,
        remove_wiki=False,
        dereify=True,
        rank=0,
        world_size=1,
        max_cached_samples=None,
        noise=0.,
):
    if out is not None:
        assert False, "cannot print text from cached"
        Path(out).write_text(
            '\n\n'.join([torch.load(p)['text'] for p in paths]))
    dataset = AMRDataset.from_cached(
        paths,
        tokenizer,
        use_recategorization=use_recategorization,
        remove_longer_than=remove_longer_than,
        remove_wiki=remove_wiki,
        dereify=dereify,
        evaluation=evaluation,
        rank=rank,
        world_size=world_size,
        max_cached_samples=max_cached_samples,
        noise=noise,
    )
    loader = AMRDatasetTokenBatcherAndLoader(
        dataset,
        batch_size=batch_size,
        shuffle=not evaluation,
    )
    return loader
