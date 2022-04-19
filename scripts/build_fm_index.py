import argparse
import logging
import multiprocessing
from collections import Counter
import math
from string import punctuation
import ftfy
import re
    
import torch
import tqdm

import csv

import logging

from generative_retrieval.index import FMIndex

logger = logging.getLogger()
logger.setLevel(logging.ERROR)

def process(line):
    tokens = tokenize(line)
    return tokens

def read_id2code(id2code_path):

    id2code = {}

    with open(id2code_path) as fin:

        for line in tqdm.tqdm(fin):

            line = line.strip()
            if not line:
                continue

            idx, code = line.split("\t")

            id2code[idx] = code

    return id2code

def preprocess_file(input_path, labels, format="kilt", lowercase=False, tokenize=False, idx2code=None):

    with open(input_path, 'r', 2**16) as f:

        if format == "dpr":

            next(f)
            pieces_it = csv.reader(f, delimiter='\t', quotechar='"')
            pieces_it = ((pp[0], pp[2], pp[1]) for pp in pieces_it if len(pp) == 3)

        elif format == "kilt":

            pieces_it = (line.strip() for line in f)
            pieces_it = (line.split('\t', 2) for line in pieces_it)
            pieces_it = ((pp[0], pp[1], pp[2]) for pp in pieces_it if len(pp) == 3)

        pieces_it = tqdm.tqdm(pieces_it)           

        for idx, title, text in pieces_it:

            idx = idx.strip()
            title = title.strip()

            text = re.sub(r'\s+', ' ', text)
            text = ftfy.fix_text(text)
            text = text.replace('BULLET::::', '')
            text = text.replace('SECTION::::', '')
            text = text.strip()

            if not text:
                continue

            if tokenize:
                title = " ".join(word_tokenize(title))
                text  = " ".join(word_tokenize(text))

            if idx2code:
                code = idx2code.get(idx)
                if code and title:
                    title = f'{title} {args.delim1} {code} {args.delim2}'
                else:
                    title = f'{title} {args.delim1}'
            else:
                title = f'{title} {args.delim1}'

            if args.include_title and title:
                text = f'{title} {text}'

            if lowercase:
                text = text.lower()

            labels.append(idx)

            yield text

def build_index(input_path, idx2code_path=None):
    
    labels = []
    index = FMIndex()

    idx2code = None if idx2code_path is None else read_id2code(idx2code_path)
    
    lines = preprocess_file(
        input_path, 
        labels, 
        args.format, 
        lowercase=args.lowercase, 
        tokenize=args.tokenize,
        idx2code=idx2code,
    )
    
    with multiprocessing.Pool(args.jobs) as p:
        sequences = p.imap(process_rotate if args.rotate else process, lines)
        #sequences = map(process_rotate if args.rotate else process, lines)
        index.initialize(sequences)
    
    index.labels = labels
    
    return index


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('input')
    parser.add_argument('output')
    parser.add_argument('--jobs', type=int, default=1)
    parser.add_argument('--rotate', action='store_true')
    parser.add_argument('--include_title', action='store_true')
    parser.add_argument('--delim1', default="@@")
    parser.add_argument('--delim2', default="||")
    parser.add_argument('--format', choices=['kilt', 'dpr'], default='kilt')
    parser.add_argument('--hf_model', default=None, type=str)
    parser.add_argument('--lowercase', action="store_true")
    parser.add_argument('--tokenize', action="store_true")
    parser.add_argument('--code')
    return parser.parse_args()

if __name__ == '__main__':

    args = parse_args()

    print(args)

    if args.tokenize:
        from spacy.lang.en import English
        nlp = English()
        _tokenizer = nlp.tokenizer
        def word_tokenize(text):
            return [t.text.strip() for t in _tokenizer(text)]

    if args.hf_model is not None:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.hf_model, use_fast=False)
        is_bart = 'bart' in args.hf_model
        def tokenize(text):
            text = text.strip()
            if is_bart:
                text = " " + text
            with tokenizer.as_target_tokenizer():
                return tokenizer(text, add_special_tokens=False)['input_ids'] + [tokenizer.eos_token_id]
    else:
        bart = torch.hub.load('pytorch/fairseq', 'bart.large').eval()
        def tokenize(text):
            return bart.encode(' ' + text.strip()).tolist()[1:]

    delim1 = tokenize(args.delim1)[:-1]
    delim2 = tokenize(args.delim2)[:-1]

    def process_rotate(line):
        tokens = tokenize(line)
        tokens = tokens[:-1] + delim2 + tokens
        return tokens

    index = build_index(args.input, idx2code_path=args.code)
    index.save(args.output)
