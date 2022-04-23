# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import psutil
import logging
import sys
from typing import *
from dataclasses import dataclass
from itertools import islice
import multiprocessing
from more_itertools import ichunked
import tqdm

from transformers import AutoConfig, AutoModelForSeq2SeqLM, AutoTokenizer, BartTokenizer, BartForConditionalGeneration

from seal.index import FMIndex
from seal.beam_search import fm_index_generate
from seal import keys as rk
from seal.utils import \
    load_state_dict_from_lightning_checkpoint, \
    load_state_dict_from_fairseq_checkpoint

word_tokenizer = None

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

DEBUG = False

def _init_word_tokenizer():
    global word_tokenizer
    if word_tokenizer is None:
        from spacy.lang.en import English
        word_tokenizer = English().tokenizer

def _get_process_memory():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss

def batch_generate_keys(searcher, queries, constrained_generation=True):

    if searcher.add_query_to_keys:
        _init_word_tokenizer()

    def process_batch(inputs):

        inputs = [(" " + q.strip()) if searcher.prepend_space else q.strip() for q in inputs]
        input_tokens = searcher.bart_tokenizer(inputs, padding=False)['input_ids']

        if searcher.decode_body:

            batch_str = inputs
            if searcher.use_markers:
                batch_str = [i + " || body" for i in batch_str]
            if searcher.value_conditioning:
                batch_str = [i + " || +" for i in batch_str]
                
            batch = searcher.bart_tokenizer(batch_str, return_tensors='pt', padding=True, truncation=True)
            batch = {k: v.to(searcher.device) for k, v in batch.items()}

            found_keys = fm_index_generate(
                searcher.bart_model, searcher.fm_index,
                **batch, 
                min_length=searcher.length,
                max_length=searcher.length,
                length_penalty=searcher.length_penalty,
                num_beams=searcher.beam,
                disable_fm_index=not constrained_generation,
                diverse_bs_groups=searcher.diverse_bs_groups,
                diverse_bs_penalty=searcher.diverse_bs_penalty,
                stop_at_count=searcher.stop_at_count,
                keep_history=True,
                topk=searcher.topk,
            )

            for fk in found_keys:
                fk[:] = [(s, k[1:] if k[0] in searcher.strip_token_ids else k) for s, k in fk if k]
                fk[:] = [(s, k[1:] if k[0] in searcher.strip_token_ids else k) for s, k in fk if k]
                fk[:] = [(s, k[:-1] if k[-1] in searcher.strip_token_ids else k) for s, k in fk if k]
                if searcher.min_length > 0:
                    fk[:] = [(s, k) for s, k in fk if len(k) == searcher.min_length]
                fk[:] = [(s, k)  for s, k in fk if k and searcher.fm_index.get_count(k) > 0]

            if searcher.rescore and searcher.use_markers:

                input_tokens = searcher.bart_tokenizer(inputs, padding=False)['input_ids']
                
                found_keys = rk.rescore_keys(
                    searcher.bart_model,
                    input_tokens,
                    found_keys,
                    batch_size=100,
                    length_penalty=0.0,
                    strip_from_bos=[
                        searcher.title_bos_token_id,
                        searcher.code_bos_token_id,
                        searcher.bart_model.config.decoder_start_token_id],
                    strip_from_eos=[
                        searcher.title_eos_token_id,
                        searcher.code_eos_token_id,
                        searcher.bart_model.config.eos_token_id])
            
        else:
            found_keys = [[] for _ in inputs]

        if searcher.add_query_to_keys:
            
            found_keys_input_no_score = []
            for inp in inputs:
                new_fk = searcher.bart_tokenizer(
                    rk.decompose_query_into_keys(inp, word_tokenizer, 3),
                    padding=False,
                    add_special_tokens=False)['input_ids']
                with searcher.bart_tokenizer.as_target_tokenizer():
                    new_fk = searcher.bart_tokenizer(searcher.bart_tokenizer.batch_decode(new_fk), padding=False)['input_ids']
                new_fk = [k[:-1] if k and k[-1] in searcher.strip_token_ids else k for k in new_fk if k]
                new_fk = [k[1:] if k and k[0] in searcher.strip_token_ids else k for k in new_fk if k]
                new_fk = [k[1:] if k and k[0] in searcher.strip_token_ids else k for k in new_fk if k]
                if searcher.min_length > 0:
                    new_fk = [k for k in new_fk if len(k) == searcher.min_length]
                new_fk = [k  for k in new_fk if k and searcher.fm_index.get_count(k) > 0]
                found_keys_input_no_score.append(new_fk)

            batch_str = inputs
            if searcher.use_markers:
                batch_str = [i + " || body" for i in batch_str]
            if searcher.value_conditioning:
                batch_str = [i + " || +" for i in batch_str]

            input_tokens = searcher.bart_tokenizer(batch_str, padding=False)['input_ids']
            
            found_keys_input = rk.rescore_keys(
                searcher.bart_model,
                input_tokens,
                found_keys_input_no_score,
                batch_size=100,
                length_penalty=0.0)
            
            for fk, nfk in zip(found_keys, found_keys_input):
                fk += nfk

        if searcher.decode_titles:
            
            batch_str = inputs
            if searcher.use_markers:
                batch_str = [i + " || title" for i in batch_str]
            if searcher.value_conditioning:
                batch_str = [i + " || +" for i in batch_str]
            
            batch = searcher.bart_tokenizer(batch_str, return_tensors='pt', padding=True, truncation=True)
            batch = {k: v.to(searcher.device) for k, v in batch.items()}
    
            decoded_title = fm_index_generate(
                searcher.bart_title_model, searcher.fm_index,
                **batch, 
                min_length=1, 
                max_length=15,
                num_beams=searcher.beam,
                length_penalty=searcher.length_penalty,
                force_decoding_from=[searcher.title_bos_token_id],
                eos_token_id=searcher.title_eos_token_id,
                diverse_bs_groups=searcher.diverse_bs_groups,
                diverse_bs_penalty=searcher.diverse_bs_penalty,
                keep_history=True,
                disable_fm_index=not constrained_generation,
                topk=searcher.topk,
            )

            found_keys_title = [[(sco, hyp) for sco, hyp in dec] for dec in decoded_title]
 
            for new_fk, fk in zip(found_keys_title, found_keys):
                
                if searcher.force_decoding_second_token >= 0:
                    new_fk[:] = [(s, k[:1] + k[2:]) for s, k in new_fk if len(k) >= 3]

                new_fk[:] = [(s, k[:-1] if k[-1] in searcher.strip_token_ids else k) for s, k in new_fk]
                if not searcher.partial_titles:
                    new_fk[:] = [(s, k) for s, k in new_fk if k[-1] == searcher.title_eos_token_id]
                    if searcher.min_length > 0:
                        new_fk[:] = [(s, k) for s, k in new_fk if len(k) == (searcher.min_length+1)]
                new_fk[:] = [(s, [searcher.title_bos_token_id] + k if k[0] != searcher.title_bos_token_id else k) for s, k in new_fk]
                new_fk[:] = [(s, k)  for s, k in new_fk if k and searcher.fm_index.get_count(k) > 0]
            
            if searcher.rescore and searcher.use_markers:

                input_tokens = searcher.bart_tokenizer(batch_str, padding=False)['input_ids']
                
                found_keys_title = rk.rescore_keys(
                    searcher.bart_title_model,
                    input_tokens,
                    found_keys_title,
                    batch_size=100,
                    length_penalty=0.0,
                    strip_from_bos=[
                        searcher.title_bos_token_id,
                        searcher.code_bos_token_id,
                        searcher.bart_model.config.decoder_start_token_id],
                    strip_from_eos=[searcher.bart_model.config.eos_token_id])
                
            for new_fk, fk in zip(found_keys_title, found_keys):
                fk += new_fk

        if searcher.decode_code:
            
            batch_str = inputs
            if searcher.use_markers:
                batch_str = [i + " || code" for i in batch_str]
            if searcher.value_conditioning:
                batch_str = [i + " || +" for i in batch_str]
            
            batch = searcher.bart_tokenizer(batch_str, return_tensors='pt', padding=True, truncation=True)
            batch = {k: v.to(searcher.device) for k, v in batch.items()}
    
            decoded_code = fm_index_generate(
                searcher.bart_code_model, searcher.fm_index,
                **batch, 
                min_length=1, 
                max_length=15,
                num_beams=searcher.beam,
                length_penalty=searcher.length_penalty,
                eos_token_id=searcher.code_eos_token_id,
                diverse_bs_groups=searcher.diverse_bs_groups,
                diverse_bs_penalty=searcher.diverse_bs_penalty,
                keep_history=True,
                force_decoding_from=[searcher.code_bos_token_id],
                disable_fm_index=not constrained_generation,
            )

            found_keys_code = [[(sco, hyp) for sco, hyp in dec] for dec in decoded_code]
 
            for new_fk, fk in zip(found_keys_code, found_keys):
                if searcher.force_decoding_second_token >= 0:
                    new_fk[:] = [(s, k[:1] + k[2:]) for s, k in new_fk if len(k) >= 2]
                new_fk[:] = [(s, k[1:-1] if k[-1] in searcher.strip_token_ids else k[1:]) for s, k in new_fk if k]
                if not searcher.partial_code:
                    new_fk[:] = [(s, k) for s, k in new_fk if k and (k[-1] == searcher.code_eos_token_id)]
                new_fk[:] = [(s, [searcher.code_bos_token_id] + k if k[0] != searcher.code_bos_token_id else k) for s, k in new_fk if k]
                new_fk[:] = [(s, k)  for s, k in new_fk if k and searcher.fm_index.get_count(k) > 0]

            if searcher.rescore and searcher.use_markers:

                input_tokens = searcher.bart_tokenizer(batch_str, padding=False)['input_ids']
                
                found_keys_code = rk.rescore_keys(
                    searcher.bart_code_model,
                    input_tokens,
                    found_keys_code,
                    batch_size=100,
                    length_penalty=0.0,
                    strip_from_bos=[
                        searcher.title_bos_token_id,
                        searcher.code_bos_token_id,
                        searcher.bart_model.config.decoder_start_token_id],
                    strip_from_eos=[searcher.bart_model.config.eos_token_id])

            for new_fk, fk in zip(found_keys_code, found_keys):
                fk += new_fk

        if searcher.rescore and not searcher.use_markers:
            found_keys = rk.rescore_keys(
                searcher.bart_scorer_model,
                input_tokens,
                found_keys,
                batch_size=100,
                length_penalty=0.0,
                strip_from_bos=[
                    searcher.title_bos_token_id,
                    searcher.code_bos_token_id,
                    searcher.bart_model.config.decoder_start_token_id],
                strip_from_eos=[searcher.bart_model.config.eos_token_id])

        for fk in found_keys:
            fk[:] = rk.deduplicate(fk)

        found_keys = [[(n, s) for s, n in xx] for xx in found_keys]

        if searcher.unigram_scores:

            batch_str = inputs
            if searcher.use_markers:
                batch_str = [i + " || body" for i in batch_str]
            if searcher.value_conditioning:
                batch_str = [i + " || +" for i in batch_str]

            input_tokens = searcher.bart_tokenizer(batch_str, padding=False)['input_ids']

            unigram_scores = rk.compute_unigram_scores(
                searcher.bart_scorer_model,
                input_tokens, 
                searcher.fm_index,
                prefix=[searcher.force_decoding_second_token] if searcher.force_decoding_second_token >= 0 else [],
            )
            found_keys = list(zip(found_keys, unigram_scores))
            return found_keys
        else:
            return found_keys

    with tqdm.tqdm(total=len(queries), desc="Generating keys", disable=not searcher.progress) as bar:
        batches = ichunked(queries, searcher.batch_size)
        for batch in batches:
            for instance in process_batch(batch):
                bar.update()
                yield instance 


class SEALDocument:

    def __init__(
            self, idx: int, 
            score: float, 
            fm_index: FMIndex,
            bart_tokenizer: BartTokenizer, 
            delim1: int = 49314,
            delim2: int = None,
            keys=None, 
            query=None):

        self.idx = idx
        self.score = score
        self.fm_index = fm_index
        self.bart_tokenizer = bart_tokenizer
        self.delim1 = delim1
        self.delim2 = delim2
        self.keys = keys
        self.query = query
        self._raw_tokens = None
        self._body = None
        self._title = None

    @property
    def docid(self):
        return self.fm_index.labels[self.idx]

    def id(self):
        return self.idx

    def raw_tokens(self):
        if self._raw_tokens is None:
            self._raw_tokens = self.fm_index.get_doc(self.idx)
        return self._raw_tokens

    def raw_text(self):
        tokens = self.raw_tokens()
        return self.bart_tokenizer.decode(tokens, clean_up_tokenization_spaces=False)

    def text(self):
        if self._body is None or self._title is None:
            tokens = self.raw_tokens()
            title_tokens, body_tokens = self.split_tokens(tokens)
            if title_tokens:
                title = self.bart_tokenizer.decode(title_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            else:
                title = ""
            body = self.bart_tokenizer.decode(body_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            self._title = title
            self._body = body
        return self._title, self._body

    def split_tokens(self, tokens):
        
        if self.delim1 is None:
            title_tokens = []
            body_tokens = []
        else:
            try:
                i = tokens.index(self.delim1)
                title_tokens = tokens[:i]
                body_tokens = tokens[i+1:] 
            except IndexError:
                title_tokens = []
                body_tokens = tokens
            except ValueError:
                title_tokens = []
                body_tokens = tokens

        if self.delim2 is not None:
            try:
                i = body_tokens.index(self.delim2) + 1
            except IndexError: 
                i = 0
            except ValueError:
                i = 0

        body_tokens = body_tokens[i:]
        return title_tokens, body_tokens

    def __repr__(self):
        return f'<GRDocument: {self.idx}, "{self.raw_text()[:30]}[...]">'

class SEALSearcher:

    DEFAULTS = \
    {
        "backbone": 'facebook/bart-large',
        "fairseq_checkpoint": True,
        "length": 10,
        "min_length": 0,
        "length_penalty": 0.0,
        "scoring_length_penalty": 0.0,
        "repetition_penalty": 0.8,
        "score_exponent": 2.0,
        "beam": 15,
        "max_hits": 1500,
        "fully_score": 1500,
        "skip_frequent_keys": 10_000_000,
        "add_query_to_keys": True,
        "batch_size": 20,
        "jobs": 1,
        "progress": False,
        "free_generation": False,
        "use_fm_index_frequency": True,
        "unigram_scores": True,
        "add_best_unigrams_to_ngrams": True,
        "use_top_k_ngrams": 5000,
        "sort_by_length": False,
        "sort_by_freq": False,
        "print_n_doc": False,
        "allow_overlaps": False,
        "diverse_bs_groups": 1,
        "diverse_bs_penalty": 0.0,
        "rescore": True,
        "detokenize": True,
        "include_keys": False,
        "single_key": 0.0,
        "unigrams_ignore_free_places": False,
        "use_markers": True,
        "value_conditioning": True,
        "decode_body": True,
        "decode_titles": True,
        "decode_code": False,
        "partial_code": False, 
        "partial_titles": False,
        "smoothing": 5.0,
        "stop_at_count": 0,
        "topk": 0,
        "force_decoding_second_token": -1,    
    }

    def __init__(
        self, 
        fm_index: FMIndex,
        bart_tokenizer: BartTokenizer,
        bart_model: BartForConditionalGeneration,
        bart_scorer_model: Optional[BartForConditionalGeneration] = None,
        bart_title_model: Optional[BartForConditionalGeneration] = None,
        bart_code_model: Optional[BartForConditionalGeneration] = None,
        **params):

        self.fm_index = fm_index
        self.docid2idx = {k: i for i, k in enumerate(self.fm_index.labels)}
        self.bart_tokenizer = bart_tokenizer
        self.bart_model = bart_model
        
        if bart_scorer_model is None:
            self.bart_scorer_model = self.bart_model
        else:
            self.bart_scorer_model = bart_scorer_model

        if bart_title_model is None:
            self.bart_title_model = self.bart_model
        else:
            self.bart_title_model = bart_title_model

        if bart_code_model is None:
            self.bart_code_model = self.bart_model
        else:
            self.bart_code_model= bart_code_model
        
        self.num_docs = fm_index.n_docs
        self.docids = fm_index.labels
        self.set_params(params)

        if 'bart' in self.backbone:
            self.title_bos_token = '</s>'
            self.title_bos_token_id = 2
            self.title_eos_token = '@@'
            self.title_eos_token_id = 49314
            self.code_bos_token = '@@'
            self.code_bos_token_id = 49314
            self.code_eos_token = '||'
            self.code_eos_token_id = 45056
            self.prepend_space = True
            self.strip_token_ids = (0, 2)

        elif 't5' in self.backbone:
            self.title_bos_token = '</s>'
            self.title_bos_token_id = 1
            self.title_eos_token = '<extra_id_99>'
            self.title_eos_token_id = 32000
            self.code_bos_token = '<extra_id_99>'
            self.code_bos_token_id = 32000
            self.code_eos_token = '<extra_id_98>'
            self.code_eos_token_id = 32001
            self.prepend_space = False
            self.strip_token_ids = (0, 1)

        else:
            raise NotImplementedError

    @property
    def device(self):
        return next(self.bart_model.parameters()).device

    @device.setter
    def device(self, device: str):
        self.bart_model.to(device)

    def set_params(self, params):
        for key, val in self.DEFAULTS.items():
            setattr(self, key, params.get(key, val))

    @classmethod
    def add_args(cls, parser):
        parser.add_argument('--fm_index', required=True, type=str)
        parser.add_argument('--checkpoint', required=False, type=str)
        parser.add_argument('--checkpoint_scorer', required=False, type=str, default=None)
        parser.add_argument('--checkpoint_title', required=False, type=str, default=None)
        parser.add_argument('--checkpoint_code', required=False, type=str, default=None)
        parser.add_argument('--device', default="cpu", type=str)
        for name, value in cls.DEFAULTS.items():
            if value is True:
                parser.add_argument(f'--dont_{name}', action="store_false", dest=name)
            elif value is False:
                parser.add_argument(f'--{name}', action="store_true")
            else:
                parser.add_argument(f'--{name}', required=False, type=type(value), default=value)
        
    @classmethod
    def from_args(cls, args):
        params = {}
        for name, value in cls.DEFAULTS.items():
            params[name] = getattr(args, name)
        return cls.load(
            args.fm_index, 
            args.checkpoint, 
            bart_scorer_model_path=args.checkpoint_scorer,
            bart_title_model_path=args.checkpoint_title,
            bart_code_model_path=args.checkpoint_code,
            device=args.device, 
            **params
        )

    @staticmethod
    def load_fm_index(fm_index_path: str):
        mem_before = _get_process_memory()
        logger.log(logging.WARN, f"initializing FM-index from {fm_index_path}")
        index = FMIndex.load(fm_index_path)
        mem_after = _get_process_memory()
        logger.log(logging.WARN, f"FM-index initialized ({(mem_after - mem_before) // 1024 ** 2} MBs)")
        return index

    @staticmethod
    def load_bart(bart_model_path: str, device: str = "cpu", backbone="facebook/bart-large", fairseq_checkpoint=True):

        logger.log(logging.WARN, f"initializing BART large")
        config = AutoConfig.from_pretrained(backbone)
        config.forced_bos_token_id = None
        tokenizer = AutoTokenizer.from_pretrained(backbone)
        if bart_model_path:
            model = AutoModelForSeq2SeqLM.from_config(config)
            model.resize_token_embeddings(len(tokenizer))
            logger.log(logging.WARN, f"loading weights from checkpoint: {bart_model_path}")
            if fairseq_checkpoint:
                load_state_dict_from_fairseq_checkpoint(model, bart_model_path)
            else:
                load_state_dict_from_lightning_checkpoint(model, bart_model_path)
        else:
            model = AutoModelForSeq2SeqLM.from_pretrained(backbone)
            model.resize_token_embeddings(len(tokenizer))

        model.config.forced_bos_token_id = None
        model.eval()

        # for some trained models, the mask logit is set to 0 for some reason. This ugly hack fixes it
        if hasattr(model, 'final_logits_bias'):
            model.config.add_bias_logits = True
            model.final_logits_bias[0, tokenizer.pad_token_id] = float('-inf')
            model.final_logits_bias[0, tokenizer.bos_token_id] = float('-inf')
            model.final_logits_bias[0, tokenizer.mask_token_id] = float('-inf')

        model.to(device)
        logger.log(logging.WARN, f"model successfully loaded")
        return tokenizer, model

    @classmethod
    def load(cls, fm_index_path, bart_model_path, device="cpu", **params):

        fm_index = cls.load_fm_index(fm_index_path)
        bart_tokenizer, bart_model = cls.load_bart(
            bart_model_path, 
            device, 
            backbone=params.get('backbone', "facebook/bart-large"), 
            fairseq_checkpoint=params.get('fairseq_checkpoint', True)
        )
        if params.get('bart_scorer_model_path') is None:
            bart_scorer_model = None
        else:
            _, bart_scorer_model = cls.load_bart(
            params.get('bart_scorer_model_path'),             
            device, 
            backbone=params.get('backbone', "facebook/bart-large"), 
            fairseq_checkpoint=params.get('fairseq_checkpoint', True)        
        )

        if params.get('bart_title_model_path') is None:
            bart_title_model = None
        else:
            _, bart_title_model = cls.load_bart(
            params.get('bart_title_model_path'),             
            device, 
            backbone=params.get('backbone', "facebook/bart-large"), 
            fairseq_checkpoint=params.get('fairseq_checkpoint', True)        
        )
        if params.get('bart_code_model_path') is None:
            bart_code_model = None
        else:
            _, bart_code_model = cls.load_bart(
            params.get('bart_code_model_path'),             
            device, 
            backbone=params.get('backbone', "facebook/bart-large"), 
            fairseq_checkpoint=params.get('fairseq_checkpoint', True)        
        )

        searcher = cls(
            fm_index, 
            bart_tokenizer, 
            bart_model, 
            bart_scorer_model=bart_scorer_model,
            bart_title_model=bart_title_model,
            bart_code_model=bart_code_model, 
            **params
        )
        return searcher

    def search(self, query: str, k: int = 10, added_documents=None, detokenize=True) -> List[SEALDocument]:
        if added_documents is not None:
            added_documents = [added_documents]
        return self.batch_search([query], k=k, added_documents=added_documents, detokenize=True)[0]

    def batch_search(self, queries, k: int = 10, added_documents=None, detokenize=None) -> List[List[SEALDocument]]:
        if detokenize is None:
            detokenize = self.detokenize
        retrieved = []
        keys = self.batch_generate_keys(queries)
        if added_documents is not None:
            if self.unigram_scores:
                keys = ((kk, us, added_documents[i]) for i, (kk, us) in enumerate(keys))
            else:
                keys = ((kk, None, added_documents[i]) for i, kk in enumerate(keys))
        
        results, keys = zip(*self.batch_retrieve_from_keys(keys))

        keys = list({k for kk in keys for k in kk})
        vals = self.bart_tokenizer.batch_decode([list(k) for k in keys], clean_up_tokenization_spaces=False)
        keys = {k: (v, self.fm_index.get_count(list(k))) for k, v in zip(keys, vals)}

        for query, res in zip(queries, results):
            docs = []
            for idx, (score, kk, _, full, _) in islice(res.items(), k):
                doc = SEALDocument(
                    idx, 
                    score, 
                    self.fm_index, 
                    self.bart_tokenizer, 
                    delim1=self.title_eos_token_id,
                    delim2=self.code_eos_token_id, 
                    keys=None, 
                    query=query
                )
                if self.include_keys:
                    for k, _ in kk:
                        if k not in keys:
                            keys[k] = (self.bart_tokenizer.decode(list(k), clean_up_tokenization_spaces=False), self.fm_index.get_count(list(k)))
                    kk = [(*keys[k], s) for k, s in kk]
                    doc.keys = kk
                doc._raw_tokens = full
                docs.append(doc)
            retrieved.append(docs)
        if detokenize:
            return self.detokenize_retrieved(retrieved)
        else:
            return retrieved

    def detokenize_retrieved(self, retrieved):
        flat = [d for dd in retrieved for d in dd]
        batch_tokens = []
        for d in flat:
            if d._raw_tokens is not None:
                title, body = d.split_tokens(d._raw_tokens)
            else:
                title, body = d.split_tokens(d.raw_tokens())
            batch_tokens.append(title)
            batch_tokens.append(body)
        if self.jobs > 2:
            batch_tokens = list(self._mp_batch_detokenize(batch_tokens))
        else:
            batch_tokens = self._batch_detokenize(batch_tokens)
        
        for i in range(len(flat)):
            j = i * 2
            flat[i]._title = batch_tokens[j]
            flat[i]._body = batch_tokens[j+1]
        return retrieved

    def generate_keys(self, query):
        return next(self.batch_generate_keys([query]))

    def batch_generate_keys(self, queries):
        return batch_generate_keys(self, queries, constrained_generation=not self.free_generation)

    def retrieve_from_keys(self, keys):
        
        unigram_scores = None
        added_documents = None
        if isinstance(keys, tuple) and len(keys) == 1:
            keys = keys[0]
        elif isinstance(keys, tuple) and len(keys) == 2:
            keys, unigram_scores = keys
        elif isinstance(keys, tuple) and len(keys) == 3:
            keys, unigram_scores, added_documents = keys

        # if self.single_term_based:
        results, ngrams = rk.aggregate_evidence(
            ngrams_and_scores=keys,
            unigram_scores=unigram_scores,
            index=self.fm_index,
            max_occurrences_1=self.max_hits,
            n_docs_complete_score=self.fully_score,
            alpha=self.score_exponent,
            beta=self.repetition_penalty,
            length_penalty=self.scoring_length_penalty,
            use_fm_index_frequency=self.use_fm_index_frequency,
            add_best_unigrams_to_ngrams=self.add_best_unigrams_to_ngrams,
            use_top_k_unigrams=self.use_top_k_ngrams,
            sort_by_length=self.sort_by_length,
            sort_by_freq=self.sort_by_freq,
            smoothing=self.smoothing,
            allow_overlaps=self.allow_overlaps,
            single_key=self.single_key,
            unigrams_ignore_free_places=self.unigrams_ignore_free_places)

        if DEBUG:
            for n, s in ngrams.items():
                print(s, self.bart_tokenizer.decode(n))
        return results, ngrams

    def batch_retrieve_from_keys(self, keys):
        if self.jobs >= 2:
            yield from self._mp_batch_retrieve_from_keys(keys)
        else:
            yield from self._batch_retrieve_from_keys(keys)

    def _mp_batch_retrieve_from_keys(self, keys):
        assert self.jobs >= 2
        idx = id(self)
        setattr(sys.modules['__main__'], f'_searcher_global_{idx}', self)
        with multiprocessing.Pool(self.jobs) as pool:
            for i, (res, ngrams) in enumerate(pool.imap(_retrieve_from_keys_mp_aux, tqdm.tqdm(
                [(idx, kk) for kk in keys],
                desc="Retrieving from keys",
                disable=not self.progress
            ))):
                if self.print_n_doc:
                    print(i)
                yield res, ngrams
        delattr(sys.modules['__main__'], f'_searcher_global_{idx}')

    def _batch_detokenize(self, seqs):
        return [self.bart_tokenizer.decode(seq, skip_special_tokens=True, clean_up_tokenization_spaces=False).strip() if seq else "" for seq in seqs]

    def _mp_batch_detokenize(self, seqs):
        assert self.jobs >= 2
        idx = id(self)
        setattr(sys.modules['__main__'], f'_searcher_global_{idx}', self)
        with multiprocessing.Pool(self.jobs) as pool:
            for i, out in enumerate(pool.imap(_detokenize_mp_aux, tqdm.tqdm(
                [(idx, seq) for seq in seqs],
                desc="Detokenizing",
                disable=not self.progress
            ))):
                if self.print_n_doc:
                    print(i)
                yield out
        delattr(sys.modules['__main__'], f'_searcher_global_{idx}')

    def _batch_retrieve_from_keys(self, keys):
        keys = tqdm.tqdm(
                keys,
                desc="Retrieving from keys",
                disable=not self.progress
            )
        for i, kk in enumerate(keys):
            if self.print_n_doc:
                print(i)
            yield self.retrieve_from_keys(kk)

    def doc(self, docid: Union[str, int]) -> Optional[SEALDocument]:
        if isinstance(docid, str):
            idx = self.docid2idx[docid]
        else:
            idx = docid
        return SEALDocument(idx, None, self.fm_index, self.bart_tokenizer, delim1=self.title_eos_token_id, delim2=self.code_eos_token_id)


def _retrieve_from_keys_mp_aux(args):
    idx, keys = args
    return getattr(sys.modules['__main__'], f'_searcher_global_{idx}').retrieve_from_keys(keys)


def _detokenize_mp_aux(args):
    idx, seq = args
    if not seq:
        return ""
    return getattr(sys.modules['__main__'], f'_searcher_global_{idx}').bart_tokenizer.decode(seq, skip_special_tokens=True, clean_up_tokenization_spaces=False).strip()
