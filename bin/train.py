from pathlib import Path
from glob import glob
from distutils.util import strtobool
import math
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
from spring_amr.modeling import get_teacher_logits

from ignite.contrib.engines import common
from ignite.engine import Engine, Events
from ignite.metrics import RunningAverage
from ignite.handlers import ModelCheckpoint, global_step_from_engine
import ignite.distributed as idist
from ignite.utils import setup_logger, manual_seed

import logging
import random
logging.getLogger("penman").setLevel(logging.ERROR)

def do_train(local_rank, args, config, where_checkpoints):

    rank = idist.get_rank()
    manual_seed(config["seed"] + rank)
    world_size = idist.get_world_size()
    device = idist.device()

    fp16 = args.fp16
    
    logger = setup_logger(name="Training", filepath=where_checkpoints / 'log')
    logger.info(args)
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
        raw_graph=config.get('raw_graph', False),
        my_model=config['my_model']
    )

    if checkpoint is not None:
        logger.info(f'Checkpoint restored ({checkpoint})!')

    if args.fix_encoder:
        for x, y in model.named_parameters():
            if 'model.encoder' in x:
                y.requires_grad = False
 
    model = idist.auto_model(model)
    if args.kd:
        teacher, _ = instantiate_model_and_tokenizer(
            config['model'],
            checkpoint=args.teacher_checkpoint,
            additional_tokens_smart_init=config['smart_init'],
            dropout=config['dropout'],
            attention_dropout=config['attention_dropout'],
            from_pretrained=config['warm_start'],
            penman_linearization=config['penman_linearization'],
            collapse_name_ops=config['collapse_name_ops'],
            use_pointer_tokens=config['use_pointer_tokens'],
            raw_graph=config.get('raw_graph', False),
        )
        teacher = idist.auto_model(teacher)
    

    optimizer = RAdam(
        [y for x, y in model.named_parameters() if 'model.encoder' not in x],
        lr=config['learning_rate'],
        weight_decay=config['weight_decay'])

    if checkpoint is not None:
        optimizer.load_state_dict(torch.load(checkpoint, map_location='cpu')['optimizer'])

    optimizer = idist.auto_optim(optimizer)
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
        cached=args.cache,
        max_cached_samples=args.max_cached_samples,
        noise=args.noise,
    )
    print ("load train")
    dev_gold_path = where_checkpoints / 'tmp-dev-gold.txt'
    dev_pred_path = where_checkpoints / 'tmp-dev-pred.txt'
    dev_loader = instantiate_loader(
        config['dev'],
        tokenizer,
        batch_size=config['batch_size'],
        evaluation=True,
        use_recategorization=config['use_recategorization'],
        remove_wiki=config['remove_wiki'],
        dereify=config['dereify'],
    )

    dev_gen_loader = instantiate_loader(
        config['dev'],
        tokenizer,
        batch_size=4*config['batch_size'],
        evaluation=True, out=dev_gold_path if rank==0 else None,
        use_recategorization=config['use_recategorization'],
        rank=rank,
        world_size=world_size,
    )
    dev_gen_loader.device = device

    epoch_length = len(train_loader) / config['accum_steps']
    epoch_length = math.ceil(idist.all_reduce(epoch_length)/world_size)
    if config['training_steps']:
        config['max_epochs'] = config['training_steps'] / epoch_length * world_size
    else:
        config['training_steps'] = config['max_epochs'] / world_size * epoch_length

    if config['scheduler'] == 'constant':
        scheduler = transformers.get_constant_schedule_with_warmup(
            optimizer,
            num_warmup_steps=config['warmup_steps'])
    elif config['scheduler'] == 'linear':
        scheduler = transformers.get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=config['warmup_steps'],
            num_training_steps=config['training_steps'])
    else:
        raise ValueError
    
    def train_step(engine, batch):
        model.train()
        x, y, extra = batch
        #### configure the traininig for my_model
        input_ids_en, attention_mask_en = None, None
        teacher_lm_logits = None
        m = model.module if hasattr(model, "module") else model
        if config['my_model']:
            m.degenerate()
        if args.kd and engine.state.iteration > args.kd_start_after*config['accum_steps']:
            m.kd = True
            m.kd_alpha = args.kd_alpha
            m.kd_temperature = args.kd_temperature
            teacher_lm_logits = get_teacher_logits(teacher,
                input_ids=extra['input_ids_en'],
                attention_mask=extra['attention_mask_en'],
                **y)
        if args.es and engine.state.iteration < args.es_stop_after*config['accum_steps']:
            m.es = True
            # move from 0.8 to 0. linearly 
            m.es_rate = 0.8 * (1.0 - engine.state.iteration/(args.es_stop_after*config['accum_steps']))
            input_ids_en = extra['input_ids_en']
            attention_mask_en = extra['attention_mask_en']
        ####
        try:
            with autocast(enabled=fp16):
                if config['my_model']:
                    loss, *_ = model(**x, **y,
                        input_ids_en=input_ids_en,
                        attention_mask_en=attention_mask_en,
                        teacher_lm_logits=teacher_lm_logits)
                else:
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
    def start(engine):
        logger.info(f"training started! total epochs: {config['max_epochs']} steps: {config['training_steps']}")

    @trainer.on(Events.ITERATION_COMPLETED(every=config['accum_steps']))
    def update(engine):
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), config['grad_norm'])
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        scheduler.step()

    @trainer.on(Events.ITERATION_COMPLETED(every=config['accum_steps']*config['eval_every']))
    def log_trn_loss(engine):
        log_msg = f"training epoch: {engine.state.epoch}, iteration {engine.state.iteration}"
        log_msg += f" | loss_amr: {engine.state.metrics['trn_amr_loss']:.3f}"
        logger.info(log_msg)
        dev_loader.device = device
        if rank == 0:
            m = model.module if hasattr(model, "module") else model
            o = optimizer.module if hasattr(optimizer, "module") else optimizer
            to_save = {'model': m.state_dict(), 'optimizer': o.state_dict()}
            torch.save(to_save, where_checkpoints / 'last_ckpt')
        # back to a normal seq2seq model for testing
        if config['my_model']:
            (model.module if hasattr(model, "module") else model).degenerate()
        evaluator.run(dev_loader)
        torch.cuda.empty_cache()

    @trainer.on(Events.ITERATION_COMPLETED(once=config['accum_steps']*config['training_steps']))
    def stop_training():
        trainer.terminate()

    def smatch_eval(gen_loader): 
        graphs = predict_amrs(
            gen_loader,
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
        return smatch

    @trainer.on(Events.COMPLETED)
    def do_test(engine):
        # search & load the best ckpt
        # also return smatch score for each test file and log
        logger.info('testing started!')
        paths = []
        glob_pattn = config['test']
        if isinstance(glob_pattn, str) or isinstance(glob_pattn, Path):
            glob_pattn = [glob_pattn]
        for gpattn in glob_pattn:
            paths += [Path(p) for p in glob(gpattn)]

        for x in where_checkpoints.iterdir():
            if str(x).endswith('.pt'):
                (model.module if hasattr(model, "module") else model).load_state_dict(torch.load(x, map_location='cpu')['model'])
                for path in paths:
                    test_gen_loader = instantiate_loader(
                        path, 
                        tokenizer,
                        batch_size=4*config['batch_size'],
                        evaluation=True, out=dev_gold_path if rank==0 else None,
                        use_recategorization=config['use_recategorization'],
                        rank=rank,
                        world_size=world_size,
                    )
                    test_gen_loader.device = device
                    smatch = smatch_eval(test_gen_loader)
                    smatch = 100 * smatch
                    log_msg = f"test {x} {path} {smatch:.1f}"
                    logger.info(log_msg)

    @evaluator.on(Events.STARTED)
    def evaluate(engine):
        logger.info('evaluating started!')

    if not config['best_loss']:
        @evaluator.on(Events.COMPLETED)
        def do_eval(engine):
            smatch = smatch_eval(dev_gen_loader)
            engine.state.metrics['dev_smatch'] = smatch

    @evaluator.on(Events.COMPLETED)
    def log_dev_loss(engine):
        log_msg = f"evaluating finished epoch: {trainer.state.epoch} iteration {trainer.state.iteration}\n"
        log_msg += f"loss_amr: {engine.state.metrics['dev_amr_loss']:.3f}"
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
 
    common.add_early_stopping_by_val_score(5, evaluator, trainer, "dev_smatch") 
    trainer.run(train_loader, max_epochs=math.ceil(config['max_epochs']/world_size))

def cache_check_data(args, config):
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
        raw_graph=config.get('raw_graph', False),
        my_model=config['my_model']
    )

    print ("model and tokenizer ready")
    train_loader = instantiate_loader(
        config['train'],
        tokenizer,
        batch_size=config['batch_size'],
        evaluation=False,
        use_recategorization=config['use_recategorization'],
        remove_longer_than=config['remove_longer_than'],
        remove_wiki=config['remove_wiki'],
        dereify=config['dereify'],
        cached=args.cache,
        noise=args.noise,
    )

    train_loader.dataset.save_cached(args.make_cache)


    cnt = 0
    mx_io = 0
    mx_oi = 0
    for x, y, extra in train_loader:
        #print (tokenizer.convert_ids_to_tokens(x["input_ids"][-1]))
        #print (tokenizer.convert_ids_to_tokens(x["input_ids_en"][-1]))
        #print (extra['sentences'][-1])
        #print (x["attention_mask"])
        #print (tokenizer.convert_ids_to_tokens(y["labels"][0]))
        #print (tokenizer.convert_ids_to_tokens(y["decoder_input_ids"][0]))
        il = x["input_ids"].size(1)
        ol = y["labels"].size(1)
        mx_oi = max(ol/il, mx_oi)
        mx_io = max(il/ol, mx_io)
        cnt += 1
        #print("-"*55)
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

    # Our faster data loading by caching
    parser.add_argument('--make_cache', type=str, default=None)
    parser.add_argument('--cache', action='store_true')
    parser.add_argument('--max_cached_samples', type=int, default=None)
    
    # our innovations
    parser.add_argument('--fix_encoder', action='store_true')
    parser.add_argument('--noise', type=float, default=0.)
    parser.add_argument('--kd', action='store_true')
    parser.add_argument('--kd_start_after', type=int, default=0)
    parser.add_argument('--kd_alpha', type=float, default=0.5)
    parser.add_argument('--kd_temperature', type=float, default=1.)
    parser.add_argument('--teacher_checkpoint', type=str, default=None)

    parser.add_argument('--es', action='store_true')
    parser.add_argument('--es_stop_after', type=int, default=0)
    
    args, extra_args = parser.parse_known_args()

    if args.fp16 and autocast_available:
        raise ValueError('You\'ll need a newer PyTorch version to enable fp16 training.')

    with args.config.open() as y:
        config = yaml.load(y, Loader=yaml.FullLoader)

    # update config from cmd line
    for k, v in zip(extra_args[0::2], extra_args[1::2]):
        k = k[2:]
        assert k in config, "unknown argument: {}; acceptable arguments: {}".format(k, config.keys())
        if k in config:
            config[k] = type(config[k])(v) if not isinstance(config[k], bool) else bool(strtobool(v))
    
    # self-distillation if teacher not specified
    if args.teacher_checkpoint is None:
        args.teacher_checkpoint = args.checkpoint
    
    # only my_model support es and kd
    if args.es or args.kd:
        assert config['my_model']
    else:
        print ("no es or kd, my_model is turned off.")
        config['my_model'] = False
    
    if args.make_cache is not None:
        cache_check_data(args, config)

    root = args.ROOT
    root.mkdir(parents=True, exist_ok=True)
    where_checkpoints = root/str(len(list(root.iterdir())))
    where_checkpoints.mkdir()

    with idist.Parallel(backend="nccl", nproc_per_node=config['nproc_per_node'], master_port=random.randint(55555, 88888)) as parallel:
        parallel.run(do_train, args, config, where_checkpoints)
