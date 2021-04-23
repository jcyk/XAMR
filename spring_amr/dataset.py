import random
import torch
import tqdm
from cached_property import cached_property
from torch.utils.data import Dataset
from spring_amr.IO import read_raw_amr_data
from ignite.utils import setup_logger
from torch.nn.utils.rnn import pad_sequence

def to_tensor(samples, key_value, padding_value, device, return_mask=True):
    tokenized = [s[key_value] for s in samples]
    input_ids = pad_sequence(tokenized, batch_first=True, padding_value=padding_value)
    if not return_mask:
        return input_ids.to(device)
    attention_mask = torch.ne(input_ids, padding_value).to(torch.int64)
    return input_ids.to(device), attention_mask.to(device)

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
        teacher_tokenizer=None,
        rank=0,
        world_size=1
    ):
        logger = setup_logger(name="Data Loading")
        self.rank = rank
        self.paths = paths
        self.tokenizer = tokenizer
        self.teacher_tokenizer = teacher_tokenizer
        self.device = device
        graphs = read_raw_amr_data(paths, use_recategorization, remove_wiki=remove_wiki, dereify=dereify)
        graphs = graphs[rank::world_size]
        self.graphs = []
        self.sentences = []
        self.tokenized = []
        self.sentences_en = []
        self.tokenized_en = []
        self.sentences_teacher = []
        self.tokenized_teacher = []
        self.linearized = []
        self.linearized_extra = []
        self.remove_longer_than = remove_longer_than
        discarded = 0
        is_train = not evaluation
        for g in tqdm.tqdm(graphs): 
            l, e = self.tokenizer.linearize(g)
            l = torch.LongTensor(l)

            self.tokenizer.src_lang = g.metadata['snt_lang']
            x = self.tokenizer.encode(g.metadata['snt'], return_tensors='pt')[0]
            if is_train and remove_longer_than and len(l) > remove_longer_than:
                discarded += 1
                continue
 
            if is_train and x.size(0) / len(l) > 5.:
                logger.warning('bad training instance len(in):{}/len(out):{}'.format(x.size(0), len(l)))
                discarded += 1
                continue
            
            #####
            token_en = g.metadata.get('tok-en', None)
            if token_en and g.metadata['snt_lang'] != "en_XX":
                self.tokenizer.src_lang = "en_XX"
                x_en = self.tokenizer.encode(token_en, return_tensors='pt')[0]
            else:
                token_en = g.metadata['snt']
                x_en = x
            self.sentences_en.append(token_en)
            self.tokenized_en.append(x_en)

            self.sentences_teacher.append(token_en)
            if self.teacher_tokenizer is not None:
                self.tokenized_teacher.append(self.teacher_tokenizer.encode(token_en, return_tensors='pt')[0])
            else:
                self.tokenized_teacher.append(x_en)
            ####

            self.sentences.append(g.metadata['snt'])
            self.tokenized.append(x)
            self.graphs.append(g)
            self.linearized.append(l)
            self.linearized_extra.append(e)

        ### teacher_tokenizer is tokenizer if not given
        if self.teacher_tokenizer is None:
            self.teacher_tokenizer = self.tokenizer

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
        if self.tokenized_teacher:
            sample['tokenized_ids_teacher'] = self.tokenized_teacher[idx]
            sample['sentence_teacher'] = self.sentences_teacher[idx]

        return sample

    def collate_fn(self, samples, device=torch.device('cpu')):
        input_ids, attention_mask = to_tensor(samples, 'tokenized_ids', self.tokenizer.pad_token_id, device)
        x = {'input_ids':input_ids, 'attention_mask':attention_mask}
        extra = {'sentences': [x['sentence'] for x in samples]}
        if 'linearized_graphs_ids' in samples[0]:
            extra['graphs'] = [x['graphs'] for x in samples]
            extra['linearized_graphs'] = [x['linearized_graphs'] for x in samples]
            batch = to_tensor(samples, 'linearized_graphs_ids', self.tokenizer.pad_token_id, device, return_mask=False)
            y = {'decoder_input_ids': batch[:, :-1].contiguous(), 'labels': batch[:, 1:].contiguous()}
        else:
            y = None

        extra['ids'] = [s['id'] for s in samples]
        extra['input_ids_en'], extra['attention_mask_en'] = to_tensor(samples, 'tokenized_ids_en', self.tokenizer.pad_token_id, device)
        extra['input_ids_teacher'], extra['attention_mask_teacher'] = to_tensor(samples, 'tokenized_ids_teacher', self.teacher_tokenizer.pad_token_id, device)
        
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
