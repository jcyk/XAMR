from pathlib import Path

import torch
try:
    from torch.cuda.amp import autocast
    autocast_available = True
except ImportError:
    class autocast:
        def __init__(self, enabled=True): pass
        def __enter__(self): return self
        def __exit__(self, exc_type, exc_value, exc_traceback): pass
    autocast_available = False

from torch.cuda.amp.grad_scaler import GradScaler
import transformers


from spring_amr.dataset import reverse_direction
from spring_amr.optim import RAdam
from spring_amr.evaluation import write_predictions, compute_smatch, predict_amrs, predict_sentences, compute_bleu
from spring_amr.utils import instantiate_model_and_tokenizer, instantiate_loader
from spring_amr.penman import encode

from ignite.engine import Engine, Events
from ignite.metrics import RunningAverage
from ignite.handlers import ModelCheckpoint, global_step_from_engine

import ignite.distributed as idist
from ignite.utils import setup_logger, manual_seed

def do_train(local_rank, args, config, where_checkpoints):

    rank = idist.get_rank()
    manual_seed(config["seed"] + rank)
    world_size = idist.get_world_size()
    device = idist.device()

    fp16 = args.fp16
    
    logger = setup_logger(name="Training")
    logger.info(config)


    checkpoint = args.checkpoint
    model, tokenizer = instantiate_model_and_tokenizer(
        config['model'],
        checkpoint=checkpoint,
        additional_tokens_smart_init=config['smart_init'],
        dropout=config['dropout'],
        attention_dropout=config['attention_dropout'],
        from_pretrained=config['warm_start'],
        penman_linearization=config['penman_linearization'],
        collapse_name_ops=config['collapse_name_ops'],
        use_pointer_tokens=config['use_pointer_tokens'],
        raw_graph=config.get('raw_graph', False)
    )

    model = idist.auto_model(model)

    if checkpoint is not None:
        logger.info(f'Checkpoint restored ({checkpoint})!')

    
    optimizer = RAdam(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay'])

    if checkpoint is not None:
        optimizer.load_state_dict(torch.load(checkpoint)['optimizer'])

    optimizer = idist.auto_optim(optimizer)

    if config['scheduler'] == 'cosine':
        scheduler = transformers.get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=config['warmup_steps'],
            num_training_steps=config['training_steps'])
    elif config['scheduler'] == 'constant':
        scheduler = transformers.get_constant_schedule_with_warmup(
            optimizer,
            num_warmup_steps=config['warmup_steps'])
    else:
        raise ValueError

    scaler = GradScaler(enabled=fp16)

    train_loader = instantiate_loader(
        config['train'],
        tokenizer,
        batch_size=config['batch_size'],
        evaluation=False,
        use_recategorization=config['use_recategorization'],
        remove_longer_than=config['remove_longer_than'],
        remove_wiki=config['remove_wiki'],
        dereify=config['dereify'],
    )

    dev_gold_path = where_checkpoints / 'tmp-dev-gold.txt'
    dev_pred_path = where_checkpoints / 'tmp-dev-pred.txt'
    dev_loader = instantiate_loader(
        config['dev'],
        tokenizer,
        batch_size=config['batch_size'],
        evaluation=True, out=dev_gold_path if rank==0 else None,
        use_recategorization=config['use_recategorization'],
        remove_wiki=config['remove_wiki'],
        dereify=config['dereify'],
    )


    def train_step(engine, batch):
        model.train()
        x, y, extra = batch
        try:
            with autocast(enabled=fp16):
                loss, *_ = model(**x, **y)
            scaler.scale((loss / config['accum_steps'])).backward()
            loss = loss.item()
        except RuntimeError as e:
            if "out of memory" in str(e):
                print (engine.state.iteration, rank, "train OOM", x['input_ids'].size(), y['labels'].size())
            raise e

        return loss

    @torch.no_grad()
    def eval_step(engine, batch):
        model.eval()
        x, y, extra = batch
        try:
            loss, *_ = model(**x, **y)
            loss = loss.item()
        except RuntimeError as e:
            if "out of memory" in str(e):
                print (engine.state.iteration, rank, "eval OOM", x['input_ids'].size(), y['labels'].size())
            raise e
        return loss


    trainer = Engine(train_step)
    evaluator = Engine(eval_step)

    @trainer.on(Events.STARTED)
    def update(engine):
        logger.info('training started!')

    @trainer.on(Events.ITERATION_COMPLETED(every=config['accum_steps']))
    def update(engine):
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), config['grad_norm'])
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        scheduler.step()

    @trainer.on(Events.ITERATION_COMPLETED(every=config['accum_steps']*config['eval_every'])|Events.COMPLETED)
    def log_trn_loss(engine):
        log_msg = f"training epoch: {engine.state.epoch}, iteration {engine.state.iteration}"
        log_msg += f" | loss_amr: {engine.state.metrics['trn_amr_loss']:.3f}"
        logger.info(log_msg)
        dev_loader.batch_size = config['batch_size']
        dev_loader.device = device
        if rank == 0:
            m = model.module if hasattr(model, "module") else model
            o = optimizer.module if hasattr(optimizer, "module") else optimizer
            to_save = {'model': m.state_dict(), 'optimizer': o.state_dict()}
            torch.save(to_save, where_checkpoints / 'last_ckpt')
        evaluator.run(dev_loader)
        torch.cuda.empty_cache()

    @evaluator.on(Events.STARTED)
    def evaluate(engine):
        logger.info('evaluating started!')

    if not config['best_loss']:
        @evaluator.on(Events.COMPLETED)
        def smatch_eval(engine):
            loader = instantiate_loader(
                config['dev'],
                tokenizer,
                batch_size=config['batch_size']//2,
                evaluation=True,
                use_recategorization=config['use_recategorization'],
                rank=rank,
                world_size=world_size
            )
            loader.device = device 
            graphs = predict_amrs(
                loader,
                model,
                tokenizer,
                beam_size=config['beam_size'],
                restore_name_ops=config['collapse_name_ops']
            )
            
            pieces = [encode(g) for g in graphs]
            pred_path = Path(str(dev_pred_path) + str(rank))
            pred_path.write_text('\n\n'.join(pieces))

            idist.barrier()
            if rank == 0:
                pred_pieces = []
                tot = 0
                for rk in range(world_size):
                    pred_path = Path(str(dev_pred_path) + str(rk))
                    pred_pieces.append(pred_path.open().read().split('\n\n'))
                    tot += len(pred_pieces[-1])
                    pred_path.unlink()
                pieces = [ pred_pieces[i%world_size][i//world_size] for i in range(tot) ]
                dev_pred_path.write_text('\n\n'.join(pieces))
                #write_predictions(dev_pred_path, tokenizer, graphs)
            idist.barrier()
            try:
                smatch = compute_smatch(dev_gold_path, dev_pred_path)
            except:
                smatch = 0.
            engine.state.metrics['dev_smatch'] = smatch

    @evaluator.on(Events.COMPLETED)
    def log_dev_loss(engine):
        log_msg = f"dev epoch: {trainer.state.epoch} iteration {trainer.state.iteration}"
        log_msg += f" | loss_amr: {engine.state.metrics['dev_amr_loss']:.3f}"
        if not config['best_loss']:
            log_msg += f" | smatch: {engine.state.metrics['dev_smatch']:.3f}"
        
        logger.info(log_msg)

    RunningAverage(output_transform=lambda out: out).attach(trainer, 'trn_amr_loss')
    RunningAverage(output_transform=lambda out: out).attach(evaluator, 'dev_amr_loss')
    
    if config['save_checkpoints']:

        if config['best_loss']:
            prefix = 'best-loss-amr'
            score_function = lambda x: 1 / evaluator.state.metrics['dev_amr_loss']
        else:
            prefix = 'best-smatch'
            score_function = lambda x: evaluator.state.metrics['dev_smatch']

        to_save = {'model': model, 'optimizer': optimizer}

        handler = ModelCheckpoint(
            str(where_checkpoints),
            prefix,
            n_saved=1,
            create_dir=True,
            score_function=score_function,
            global_step_transform=global_step_from_engine(trainer),
        )
        evaluator.add_event_handler(Events.COMPLETED, handler, to_save)

    train_loader.device = device
    trainer.run(train_loader, max_epochs=config['max_epochs'])

def check_data(args, config):
    checkpoint = args.checkpoint
    model, tokenizer = instantiate_model_and_tokenizer(
        config['model'],
        checkpoint=checkpoint,
        additional_tokens_smart_init=config['smart_init'],
        dropout=config['dropout'],
        attention_dropout=config['attention_dropout'],
        from_pretrained=config['warm_start'],
        penman_linearization=config['penman_linearization'],
        collapse_name_ops=config['collapse_name_ops'],
        use_pointer_tokens=config['use_pointer_tokens'],
        raw_graph=config.get('raw_graph', False)
    )

    train_loader = instantiate_loader(
        config['test'],
        tokenizer,
        batch_size=config['batch_size'],
        evaluation=False,
        use_recategorization=config['use_recategorization'],
        remove_longer_than=config['remove_longer_than'],
        remove_wiki=config['remove_wiki'],
        dereify=config['dereify'],
    )

    cnt = 0
    mx_io = 0
    mx_oi = 0
    for x, y, extra in train_loader:
        #print (tokenizer.convert_ids_to_tokens(x["input_ids"][0]))
        #print (x["attention_mask"])
        #print (tokenizer.convert_ids_to_tokens(y["labels"][0]))
        #print (tokenizer.convert_ids_to_tokens(y["decoder_input_ids"][0]))
        il = x["input_ids"].size(1)
        ol = y["labels"].size(1)
        mx_oi = max(ol/il, mx_oi)
        mx_io = max(il/ol, mx_io)
        cnt += 1
    print (cnt, mx_oi, mx_io)
    assert True == False

if __name__ == '__main__':

    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
    import yaml

    parser = ArgumentParser(
        description="Trainer script",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--config', type=Path, default='configs/sweeped.yaml',
        help='Use the following config for hparams.')
    parser.add_argument('--checkpoint', type=str, default=None,
        help='Warm-start from a previous fine-tuned checkpoint.')
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('--ROOT', type=Path)

    args, unknown = parser.parse_known_args()

    if args.fp16 and autocast_available:
        raise ValueError('You\'ll need a newer PyTorch version to enable fp16 training.')

    with args.config.open() as y:
        config = yaml.load(y, Loader=yaml.FullLoader)

    #check_data(args, config)

    root = args.ROOT/'runs'
    root.mkdir(parents=True, exist_ok=True)
    where_checkpoints = root/str(len(list(root.iterdir())))
    where_checkpoints.mkdir()
    
    with idist.Parallel(backend="nccl", nproc_per_node=config['nproc_per_node']) as parallel:
        parallel.run(do_train, args, config, where_checkpoints)


