from pathlib import Path

import penman
import torch
import ignite.distributed as idist

from spring_amr import ROOT
from spring_amr.evaluation import predict_amrs, compute_smatch
from spring_amr.penman import encode
from spring_amr.utils import instantiate_loader, instantiate_model_and_tokenizer
from ignite.utils import setup_logger

def load_spring_ckpt(model, checkpoint):
    model_ckpt = torch.load(checkpoint, map_location='cpu')['model']
    for x in ["model.decoder.pointer_k.weight", "model.decoder.pointer_k.bias", "model.decoder.pointer_q.weight", "model.decoder.pointer_q.bias"]:
        model_ckpt.pop(x)
    model_ckpt["lm_head.weight"] = model_ckpt["model.shared.weight"]
    model.load_state_dict(model_ckpt)

def run(local_rank, args):
    rank = idist.get_rank()
    world_size = idist.get_world_size()
    device = idist.device()
    logger = setup_logger(name="Training", distributed_rank=local_rank)

    logger.info(f"\nrank:{rank}\tworld_size:{world_size}\tbackend: {idist.backend()}\n")

    model, tokenizer = instantiate_model_and_tokenizer(
        args.model,
        dropout=0.,
        attention_dropout=0.,
        penman_linearization=args.penman_linearization,
        use_pointer_tokens=args.use_pointer_tokens,
        raw_graph=args.raw_graph,
    )

    #load_spring_ckpt(model, args.checkpoint)
    model.load_state_dict(torch.load(args.checkpoint, map_location='cpu')['model'])
    model.to(device)

    gold_path = args.gold_path + rank
    pred_path = args.pred_path + rank
    loader = instantiate_loader(
        args.datasets,
        tokenizer,
        batch_size=args.batch_size,
        evaluation=True, out=gold_path,
        use_recategorization=args.use_recategorization,
        rank=rank,
        world_size=world_size
    )
    loader.device = device

    graphs = predict_amrs(
        loader,
        model,
        tokenizer,
        beam_size=args.beam_size,
        restore_name_ops=args.restore_name_ops,
        return_all=args.return_all,
    )
    if args.return_all:
        graphs = [g for gg in graphs for g in gg]

    pieces = [encode(g) for g in graphs]
    pred_path.write_text('\n\n'.join(pieces))

    idist.barrier()
    if rank != 0:
        return
    gold_pieces = []
    pred_pieces = []
    for rk in world_size:
        gold_path = args.gold_path + rk
        gold_pieces.append(gold_path.open().read())
        pred_path = args.pred_path + rk
        pred_pieces.append(pred_path.open().read())
    args.gold_path.write_text('\n\n'.join(gold_pieces))
    args.pred_path.write_text('\n\n'.join(pred_path))
    if not args.return_all:
        score = compute_smatch(args.gold_path, args.pred_path)
        print(f'Smatch: {score:.3f}')

if __name__ == '__main__':

    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

    parser = ArgumentParser(
        description="Script to predict AMR graphs given sentences. LDC format as input.",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--datasets', type=str, required=True, nargs='+',
        help="Required. One or more glob patterns to use to load amr files.")
    parser.add_argument('--checkpoint', type=str, required=True,
        help="Required. Checkpoint to restore.")
    parser.add_argument('--model', type=str, default='facebook/bart-large',
        help="Model config to use to load the model class.")
    parser.add_argument('--beam-size', type=int, default=1,
        help="Beam size.")
    parser.add_argument('--batch-size', type=int, default=1000,
        help="Batch size (as number of linearized graph tokens per batch).")
    parser.add_argument('--pred-path', type=Path, default=ROOT / 'data/tmp/inf-pred.txt',
        help="Where to write predictions.")
    parser.add_argument('--gold-path', type=Path, default=ROOT / 'data/tmp/inf-gold.txt',
        help="Where to write the gold file.")
    parser.add_argument('--use-recategorization', action='store_true',
        help="Predict using Zhang recategorization on top of our linearization (requires recategorized sentences in input).")
    parser.add_argument('--penman-linearization', action='store_true',
        help="Predict using PENMAN linearization instead of ours.")
    parser.add_argument('--use-pointer-tokens', action='store_true')
    parser.add_argument('--raw-graph', action='store_true')
    parser.add_argument('--restore-name-ops', action='store_true')
    parser.add_argument('--return-all', action='store_true')
    parser.add_argument('--nproc_per_node', type=int, default=2)

    with idist.Parallel(backend="nccl", nproc_per_node=args.nproc_per_node) as parallel:
        parallel.run(run, args)

