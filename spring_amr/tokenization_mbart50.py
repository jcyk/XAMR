import copy
import sys
from pathlib import Path

import penman
import regex as re
import torch
from transformers import MBart50Tokenizer

from spring_amr import ROOT, postprocessing
from spring_amr.linearization import AMRTokens, AMRLinearizer
from spring_amr.penman import encode


class AMRMBart50Tokenizer(MBart50Tokenizer):

    INIT = 'Ġ'

    ADDITIONAL = [
        AMRTokens.PNTR_N,
        AMRTokens.STOP_N,
        AMRTokens.LIT_START,
        AMRTokens.LIT_END,
        AMRTokens.BACKR_SRC_N,
        AMRTokens.BACKR_TRG_N,]

    def __init__(self, *args, use_pointer_tokens=False, collapse_name_ops=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.patterns = re.compile(
            r""" ?<[a-z]+:?\d*>| ?:[^\s]+|'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
        self.linearizer = AMRLinearizer(use_pointer_tokens=use_pointer_tokens, collapse_name_ops=collapse_name_ops)
        self.use_pointer_tokens = use_pointer_tokens
        self.collapse_name_ops = collapse_name_ops
        self.recategorizations = set()
        self.modified = 0

    @classmethod
    def from_pretrained(cls, pretrained_model_path, pred_min=5, *args, **kwargs):
        inst = super().from_pretrained(pretrained_model_path, *args, **kwargs)
        inst.init_amr_vocabulary(pred_min=pred_min)
        return inst

    def init_amr_vocabulary(self, pred_min=5):
        for tok in [self.bos_token, self.eos_token, self.pad_token, '<mask>', '<unk>']:
            ntok = self.INIT + tok
            i = self.encoder[tok]
            self.decoder[i] = ntok
            del self.encoder[tok]
            self.encoder[ntok] = i

        tokens = []
        for line in Path(ROOT/'data/vocab/predicates.txt').read_text().strip().splitlines():
            tok, count = line.split()
            if int(count) >= pred_min:
                tokens.append(tok)
                
        for tok in Path(ROOT/'data/vocab/additions.txt').read_text().strip().splitlines():
            tokens.append(tok)

        for tok in Path(ROOT/'data/vocab/recategorizations.txt').read_text().strip().splitlines():
            if not tok.startswith('_'):
                self.recategorizations.add(tok)
            tokens.append(tok)

        if self.use_pointer_tokens:
            for cnt in range(512):
                tokens.append(f"<pointer:{cnt}>")

        tokens += self.ADDITIONAL
        tokens = [self.INIT + t if t[0] not in ('_', '-') else t for t in tokens]
        tokens = [t for t in tokens if t not in self.encoder]
        self.old_enc_size = old_enc_size = len(self.encoder)
        for i, t in enumerate(tokens, start= old_enc_size):
            self.encoder[t] = i

        self.encoder = {k: i for i, (k,v) in enumerate(sorted(self.encoder.items(), key=lambda x: x[1]))}
        self.decoder = {v: k for k, v in sorted(self.encoder.items(), key=lambda x: x[1])}
        self.modified = len(tokens)
        
        self.bos_token = self.INIT + '<s>'
        self.pad_token = self.INIT + '<pad>'
        self.eos_token = self.INIT + '</s>'
        self.unk_token = self.INIT + '<unk>'

    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
        output = [self.bos_token_id] + token_ids_0 + [self.eos_token_id]
        if token_ids_1 is None:
            return output
        return output + [self.eos_token_id] + token_ids_1 + [self.eos_token_id]

    def _tokenize(self, text):
        """ Tokenize a string. Modified in order to handle sentences with recategorization pointers"""
        bpe_tokens = []
        for tok_span in text.lstrip().split(' '):
            tok_span = tok_span.strip()
            recats = tok_span.rsplit('_', 1)
            if len(recats) == 2 and recats[0] in self.recategorizations and ('_' + recats[1]) in self.encoder:
                bpe_tokens.extend([self.INIT + recats[0], '_' + recats[1]])
            else:
                for token in re.findall(self.pat, ' ' + tok_span):
                    token = "".join(
                        self.byte_encoder[b] for b in token.encode("utf-8")
                    )  # Maps all our bytes to unicode strings, avoiding controle tokens of the BPE (spaces in our case)
                    bpe_tokens.extend(bpe_token for bpe_token in self.bpe(token).split(" "))

        return bpe_tokens

    def _tok_bpe(self, token, add_space=True):
        # if add_space:
        #     token = ' ' + token.lstrip()
        tokk = []
        tok = token.strip()
        recats = tok.rsplit('_', 1)
        if len(recats) == 2 and recats[0] in self.recategorizations and ('_' + recats[1]) in self.encoder:
            tokk.extend([self.INIT + recats[0], '_' + recats[1]])
        else:
            for tok in self.patterns.findall(' ' + token):
                tok = "".join(
                    self.byte_encoder[b] for b in tok.encode("utf-8"))
                toks = self.bpe(tok).split(' ')
                tokk.extend(toks)
        return tokk

    def _get_nodes_and_backreferences(self, graph):
        lin = self.linearizer.linearize(graph)
        linearized_nodes, backreferences = lin.nodes, lin.backreferences
        return linearized_nodes, backreferences

    def tokenize_amr(self, graph):
        linearized_nodes, backreferences = self._get_nodes_and_backreferences(graph)

        bpe_tokens = []
        bpe_backreferences = []
        counter = 0
        
        for i, (backr, tokk) in enumerate(zip(backreferences, linearized_nodes)):
            is_in_enc = self.INIT + tokk in self.encoder
            is_rel = tokk.startswith(':') and len(tokk) > 1
            is_spc = tokk.startswith('<') and tokk.endswith('>')
            is_of  = tokk.startswith(':') and tokk.endswith('-of')
            is_frame = re.match(r'.+-\d\d', tokk) is not None

            if tokk.startswith('"') and tokk.endswith('"'):
                tokk = tokk[1:-1].replace('_', ' ')
                bpe_toks = [self.INIT + AMRTokens.LIT_START]
                bpe_toks += self._tok_bpe(tokk, add_space=True)
                bpe_toks.append(self.INIT + AMRTokens.LIT_END)

            elif (is_rel or is_spc or is_frame or is_of):
                if is_in_enc:
                    bpe_toks = [self.INIT + tokk]
                elif is_frame:
                    bpe_toks = self._tok_bpe(tokk[:-3], add_space=True) + [tokk[-3:]]
                elif is_of:
                    rel = tokk[:-3]
                    if self.INIT + rel in self.encoder:
                        bpe_toks = [self.INIT + rel, '-of']
                    else:
                        bpe_toks = [self.INIT + ':'] + self._tok_bpe(rel[1:], add_space=True) + ['-of']
                elif is_rel:
                    bpe_toks = [self.INIT + ':'] + self._tok_bpe(tokk[1:], add_space=True)
                else:
                    raise

            else:
                if is_in_enc:
                    bpe_toks = [self.INIT + tokk]
                else:
                    bpe_toks = self._tok_bpe(tokk, add_space=True)

            bpe_tokens.append(bpe_toks)

            if i == backr:
                bpe_backr = list(range(counter, counter + len(bpe_toks)))
                counter += len(bpe_toks)
                bpe_backreferences.append(bpe_backr)
            else:
                bpe_backreferences.append(bpe_backreferences[backr][0:1])
                counter += 1               
        bpe_tokens = [b for bb in bpe_tokens for b in bb]
        bpe_token_ids = [self.encoder.get(b, self.unk_token_id) for b in bpe_tokens]
        bpe_backreferences = [b for bb in bpe_backreferences for b in bb]
        return bpe_tokens, bpe_token_ids, bpe_backreferences

    def batch_encode_sentences(self, sentences, device=torch.device('cpu')):
        sentences = [s for s in sentences]
        extra = {'sentences': sentences}
        batch = super().batch_encode_plus(sentences, return_tensors='pt', padding="longest") 
        batch = {k: v.to(device) for k, v in batch.items()}
        return batch, extra
    
    def linearize(self, graph):
        shift = len(self.encoder)
        tokens, token_ids, backreferences = self.tokenize_amr(graph)
        extra = {'linearized_graphs': tokens, 'graphs': graph}
        token_uni_ids = \
            [idx if i == b else b + shift for i, (idx, b) in enumerate(zip(token_ids, backreferences))]
        if token_uni_ids[-1] != (self.INIT + AMRTokens.EOS_N):
            tokens.append(self.INIT + AMRTokens.EOS_N)
            token_ids.append(self.eos_token_id)
            token_uni_ids.append(self.eos_token_id)
            backreferences.append(len(backreferences))
        return token_uni_ids, extra
        
    def batch_encode_graphs(self, graphs, device=torch.device('cpu')):
        linearized, extras = zip(*[self.linearize(g) for g in graphs])
        return self.batch_encode_graphs_from_linearized(linearized, extras, device=device)
    
    def batch_encode_graphs_from_linearized(self, linearized, extras=None, device=torch.device('cpu')):
        if extras is not None:
            batch_extra = {'linearized_graphs': [], 'graphs': []}
            for extra in extras:
                batch_extra['graphs'].append(extra['graphs'])
                batch_extra['linearized_graphs'].append(extra['linearized_graphs'])
        else:
            batch_extra = {}
        maxlen = 0
        batch = []
        for token_uni_ids in linearized:
            maxlen = max(len(token_uni_ids), maxlen)
            batch.append(token_uni_ids)
        batch = [x + [self.pad_token_id] * (maxlen - len(x)) for x in batch]
        batch = torch.tensor(batch).to(device)
        batch = {'decoder_input_ids': batch[:, :-1], 'labels': batch[:, 1:]}
        return batch, batch_extra

    def decode_amr(self, tokens, restore_name_ops=False):
        try:
            nodes, backreferences = postprocessing.decode_into_node_and_backreferences(tokens, self)
        except Exception as e:
            print('Decoding failure:', file=sys.stderr)
            print(e, file=sys.stderr)
            return postprocessing.BACKOFF, postprocessing.ParsedStatus.BACKOFF, (None, None)
        if self.use_pointer_tokens:
            nodes, backreferences = postprocessing.restore_backreferences_from_pointers(nodes)
        try:
            graph_ = graph = postprocessing.build_graph(nodes, backreferences, restore_name_ops=restore_name_ops)
        except Exception as e:
            print('Building failure:', file=sys.stderr)
            print(nodes, file=sys.stderr)
            print(backreferences, file=sys.stderr)
            print(e, file=sys.stderr)
            return postprocessing.BACKOFF, postprocessing.ParsedStatus.BACKOFF, (None, None)
        try:
            graph, status = postprocessing.connect_graph_if_not_connected(graph)
            if status == postprocessing.ParsedStatus.BACKOFF:
                print('Reconnection 1 failure:')
                print(nodes, file=sys.stderr)
                print(backreferences, file=sys.stderr)
                print(graph_, file=sys.stderr)
            return graph, status, (nodes, backreferences)
        except Exception as e:
            print('Reconnction 2 failure:', file=sys.stderr)
            print(e, file=sys.stderr)
            print(nodes, file=sys.stderr)
            print(backreferences, file=sys.stderr)
            print(graph_, file=sys.stderr)
            return postprocessing.BACKOFF, postprocessing.ParsedStatus.BACKOFF, (nodes, backreferences)
