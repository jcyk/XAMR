import random
import torch
from cached_property import cached_property
from torch.utils.data import Dataset
from spring_amr.IO import read_raw_amr_data
from ignite.utils import setup_logger

def reverse_direction(x, y, pad_token_id=1):
    input_ids = torch.cat([y['decoder_input_ids'], y['labels'][:, -1:]], 1)
    attention_mask = torch.ones_like(input_ids)
    attention_mask[input_ids == pad_token_id] = 0
    decoder_input_ids = x['input_ids'][:,:-1]
    labels = x['input_ids'][:,1:]
    x = {'input_ids': input_ids, 'attention_mask': attention_mask}
    y = {'decoder_input_ids': decoder_input_ids, 'labels': labels}
    return x, y

class AMRDataset(Dataset):
    """ Note we use *{en,zh,es,it}.txt to indicate the right tokenization method (src_lang).
    """
    def __init__(
        self,
        paths,
        tokenizer,
        device=torch.device('cpu'),
        use_recategorization=False,
        remove_longer_than=None,
        remove_wiki=False,
        dereify=True,
        evaluation=True,
        rank=0,
        world_size=1
    ):
        logger = setup_logger(name="Data Loading")
        self.rank = rank
        self.paths = paths
        self.tokenizer = tokenizer
        self.device = device
        graphs = read_raw_amr_data(paths, use_recategorization, remove_wiki=remove_wiki, dereify=dereify)
        graphs = graphs[rank::world_size]
        self.graphs = []
        self.sentences = []
        self.tokenized = []
        self.sentences_en = []
        self.tokenized_en = []
        self.linearized = []
        self.linearized_extra = []
        self.remove_longer_than = remove_longer_than
        discarded = 0
        is_train = not evaluation
        for g in graphs:
            l, e = self.tokenizer.linearize(g)
            

            self.tokenizer.src_lang = g.metadata['snt_lang']
            x = self.tokenizer.encode(g.metadata['snt'], return_tensors='pt')[0]
            if is_train and remove_longer_than and len(l) > remove_longer_than:
                discarded += 1
                continue
 
            if is_train and x.size(0) / len(l) > 5.:
                logger.warning('bad training instance len(in):{}/len(out):{}'.format(x.size(0), len(l)))
                discarded += 1
                continue

            token_en = g.metadata.get('tok-en', None)
            if token_en:
                self.tokenizer.src_lang = "en_XX"
                x_en = self.tokenizer.encode(token_en, return_tensors='pt')[0]
            else:
                token_en = g.metadata['snt']
                x_en = x
            self.sentences_en.append(token_en)
            self.tokenized_en.append(x_en)
            
            self.sentences.append(g.metadata['snt'])
            self.tokenized.append(x)
            self.graphs.append(g)
            self.linearized.append(l)
            self.linearized_extra.append(e)

        logger.info('the number of instances {}, discarded {}'.format(len(self.sentences), discarded))
    
    def __len__(self):
        return len(self.sentences)
    
    def __getitem__(self, idx):
        sample = {}
        sample['id'] = idx
        sample['sentence'] = self.sentences[idx]
        sample["tokenized_ids"] = self.tokenized[idx]
        if self.linearized:
            sample['linearized_graphs_ids'] = self.linearized[idx]
            sample.update(self.linearized_extra[idx])
        if self.tokenized_en:
            sample['tokenized_ids_en'] = self.tokenized_en[idx]
            sample['sentence_en'] = self.sentences_en[idx]
        return sample
    
    def collate_fn(self, samples, device=torch.device('cpu')):
        x = [s['tokenized_ids'] for s in samples]
        x, extra = self.tokenizer.batch_encode_sentences_from_tokenized(x, samples, device=device)
        if 'linearized_graphs_ids' in samples[0]:
            y = [s['linearized_graphs_ids'] for s in samples]
            y, extra_y = self.tokenizer.batch_encode_graphs_from_linearized(y, samples, device=device)
            extra.update(extra_y)
        else:
            y = None
        extra['ids'] = [s['id'] for s in samples]
        for name in y:
            y[name] = y[name].contiguous()
        return x, y, extra
    
class AMRDatasetTokenBatcherAndLoader:
    
    def __init__(self, dataset, batch_size=800 ,device=torch.device('cpu'), shuffle=False, sort=True):
        self.batch_size = batch_size
        self.tokenizer = dataset.tokenizer
        self.dataset = dataset
        self.device = device
        self.shuffle = shuffle
        self.sort = sort

    @property
    def rank(self):
        return self.dataset.rank

    def __iter__(self):
        it = self.sampler()
        it = [[self.dataset[s] for s in b] for b in it]
        if self.shuffle:
            random.shuffle(it)
        it = (self.dataset.collate_fn(b, device=self.device) for b in it)
        return it

    def sampler(self):
        ids = list(range(len(self.dataset)))
        if self.shuffle:
            random.shuffle(ids)
        if self.sort:
            if self.shuffle:
                lengths = [len(x) for x in self.dataset.linearized]
            else:
                lengths = [s.size(0) for s in self.dataset.tokenized]
            ids.sort(key=lambda x: -lengths[x])

        batch_longest = 0
        batch_nexamps = 0
        batch_ntokens = 0
        batch_ids = []

        def discharge():
            nonlocal batch_longest
            nonlocal batch_nexamps
            nonlocal batch_ntokens
            ret = batch_ids.copy()
            batch_longest *= 0
            batch_nexamps *= 0
            batch_ntokens *= 0
            batch_ids[:] = []
            return ret

        while ids:
            idx = ids.pop()
            size = len(self.dataset.linearized[idx])
            cand_batch_ntokens = max(size, batch_longest) * (batch_nexamps + 1)
            if cand_batch_ntokens > self.batch_size and batch_ids:
                yield discharge()
            batch_longest = max(batch_longest, size)
            batch_nexamps += 1
            batch_ntokens = batch_longest * batch_nexamps
            batch_ids.append(idx)

            if len(batch_ids) == 1 and batch_ntokens > self.batch_size:
                yield discharge()

        if batch_ids:
            yield discharge()
