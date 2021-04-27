from pathlib import Path

import penman
import torch
import logging

import ignite.distributed as idist

from tqdm import tqdm
from spring_amr.penman import encode
from spring_amr.utils import instantiate_model_and_tokenizer
from spring_amr.evaluation import generate
from spring_amr.dataset import to_tensor
from spring_amr.tokenization_bart import AMRBartTokenizer, PENMANBartTokenizer
from spring_amr.tokenization_mbart50 import AMRMBart50Tokenizer, PENMANMBart50Tokenizer


logging.getLogger("penman").setLevel(logging.ERROR)

def read_file_in_batches(path, batch_size, max_length=100, rank=0, world_size=1):

    data = []
    line_idx = -1
    for line in Path(path).read_text().strip().splitlines():
        line_idx += 1
        if line_idx % world_size != rank % world_size:
            continue
        sent = line.strip()
        if not sent:
            continue
        length = len(sent.split())
        if length > max_length:
            continue
        data.append({
            'line_idx':line_idx,
            'sent': sent,
            'length':length}
        )

    def _iterator(data):
        ids = list(range(len(data)))
        ids.sort(key=lambda x: data[x]['length'], reverse=True)

        batch_longest = 0
        batch_nexamps = 0
        batch_ntokens = 0
        batch_data = []

        while ids:
            idx = ids.pop()
            size = data[idx]['length']
            cand_batch_ntokens = max(size, batch_longest) * (batch_nexamps + 1)
            if cand_batch_ntokens > batch_size and batch_data:
                yield batch_data
                batch_longest = 0
                batch_nexamps = 0
                batch_ntokens = 0
                batch_data = []
            batch_longest = max(batch_longest, size)
            batch_nexamps += 1
            batch_ntokens = batch_longest * batch_nexamps
            batch_data.append(data[idx])

            if len(batch_data) == 1 and batch_ntokens > batch_size:
                yield batch_data
                batch_data = []
                batch_longest = 0
                batch_nexamps = 0
                batch_ntokens = 0
                batch_data = []

        if batch_data:
            yield batch_data
            batch_longest = 0
            batch_nexamps = 0
            batch_ntokens = 0
            batch_data = []

    return _iterator(data), len(data)

def prepare_batch(batch, tokenizer, device):
    extra = {
        'ids': [ x['line_idx'] for x in batch],
        'sentences': [ x['sent'] for x in batch],
    }
    for x in batch:
        x['tokenized'] = tokenizer.encode(x['sent'], return_tensors='pt')[0]
    input_ids, attention_mask = to_tensor(batch, 'tokenized', tokenizer.pad_token_id, device)
    x = {'input_ids': input_ids, 'attention_mask': attention_mask}
    return x, extra

def run(local_rank, args):
    rank = idist.get_rank()
    world_size = idist.get_world_size()
    device = idist.device()
    model, tokenizer = instantiate_model_and_tokenizer(
        args.model,
        dropout=0.,
        attention_dropout=0,
        penman_linearization=args.penman_linearization,
        use_pointer_tokens=args.use_pointer_tokens,
        raw_graph=False,
    )
    tokenizer.src_lang = args.src_lang
    model.load_state_dict(torch.load(args.checkpoint, map_location='cpu')['model'])
    model.to(device)
    model.eval()

    iterator, nsent = read_file_in_batches(args.input_path, args.batch_size, rank=rank, world_size=world_size)

    is_bart = isinstance(tokenizer, AMRBartTokenizer) or isinstance(tokenizer, PENMANBartTokenizer)

    with tqdm(total=nsent) as bar, open(args.output_path + str(rank), 'w') as fo:
        for batch in iterator:
            x, extra = prepare_batch(batch, tokenizer, device)

            out = generate(is_bart, model, x, args.beam_size)

            graphs_same_batch = []
            for i1 in range(0, out.size(0), args.beam_size):
                tokens_same_source = [out[i2].tolist() for i2 in range(i1, i1+args.beam_size)]
                graphs_same_source = []
                for tokk in tokens_same_source:
                    graph, status, (lin, backr) = tokenizer.decode_amr(tokk, restore_name_ops=args.restore_name_ops)
                    graph.status = status
                    graph.metadata['status'] = str(status) 
                    graphs_same_source.append(graph)
                graphs_same_source = tuple(zip(*sorted(enumerate(graphs_same_source), key=lambda x: (x[1].status.value, x[0]))))[1]
                top_graph = graphs_same_source[0]
                graphs_same_batch.append(top_graph)
            for idx, sent, graph in zip(extra['ids'], extra['sentences'], graphs_same_batch):
                graph.metadata['idx'] = str(idx)
                graph.metadata['snt'] = sent
                fo.write(encode(graph)+'\n\n')
            bar.update(len(extra['ids']))

if __name__ == '__main__':

    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

    parser = ArgumentParser(
        description="Script to predict AMR graphs given sentences. LDC format as input.",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--input-path', type=str, required=True,
        help="Required. One file containing \\n-separated sentences.")
    parser.add_argument('--output-path', type=str)
    parser.add_argument('--src-lang', type=str, default="en_XX", help="en_XX, de_DE, zh_CN, es_XX, it_IT")
    parser.add_argument('--checkpoint', type=str, required=True,
        help="Required. Checkpoint to restore.")
    parser.add_argument('--model', type=str, default='facebook/bart-large',
        help="Model config to use to load the model class.")
    parser.add_argument('--beam-size', type=int, default=1,
        help="Beam size.")
    parser.add_argument('--batch-size', type=int, default=1000,
        help="Batch size (as number of linearized graph tokens per batch).")
    parser.add_argument('--penman-linearization', action='store_true',
        help="Predict using PENMAN linearization instead of ours.")
    parser.add_argument('--use-pointer-tokens', action='store_true')
    parser.add_argument('--restore-name-ops', action='store_true')
    parser.add_argument('--nproc-per-node', type=int, default=2)

    args = parser.parse_args()
    with idist.Parallel(backend="nccl", nproc_per_node=args.nproc_per_node, master_port=8888) as parallel:
        parallel.run(run, args)
