# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import csv
import logging
import multiprocessing
import re

import ftfy
import torch
import tqdm

from seal.index import FMIndex

logger = logging.getLogger()
logger.setLevel(logging.ERROR)


def process(line):
    tokens = tokenize(line)
    return tokens


def preprocess_file(input_path, labels, format="kilt", lowercase=False, tokenize=False):
    with open(input_path, "r", 2**16) as f:

        if format == "dpr":

            next(f)
            pieces_it = csv.reader(f, delimiter="\t", quotechar='"')
            pieces_it = ((pp[0], pp[2], pp[1]) for pp in pieces_it if len(pp) == 3)

        elif format == "kilt":

            pieces_it = (line.strip() for line in f)
            pieces_it = (line.split("\t", 2) for line in pieces_it)
            pieces_it = ((pp[0], pp[1], pp[2]) for pp in pieces_it if len(pp) == 3)

        pieces_it = tqdm.tqdm(pieces_it)

        for idx, title, text in pieces_it:

            idx = idx.strip()
            title = title.strip()

            text = re.sub(r"\s+", " ", text)
            text = ftfy.fix_text(text)
            text = text.replace("BULLET::::", "")
            text = text.replace("SECTION::::", "")
            text = text.strip()

            if not text:
                continue

            if tokenize:
                title = " ".join(word_tokenize(title))
                text = " ".join(word_tokenize(text))

            title = f"{title} {args.delim}"

            if args.include_title and title:
                text = f"{title} {text}"

            if lowercase:
                text = text.lower()

            labels.append(idx)

            yield text


def build_index(input_path):

    labels = []
    index = FMIndex()

    lines = preprocess_file(input_path, labels, args.format, lowercase=args.lowercase, tokenize=args.tokenize)

    with multiprocessing.Pool(args.jobs) as p:
        sequences = p.imap(process, lines)
        index.initialize(sequences)

    index.labels = labels

    return index


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("output")
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--include_title", action="store_true")
    parser.add_argument("--delim", default="@@")
    parser.add_argument("--format", choices=["kilt", "dpr"], default="kilt")
    parser.add_argument("--hf_model", default=None, type=str)
    parser.add_argument("--lowercase", action="store_true")
    parser.add_argument("--tokenize", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":

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
        is_bart = "bart" in args.hf_model

        def tokenize(text):
            text = text.strip()
            if is_bart:
                text = " " + text
            with tokenizer.as_target_tokenizer():
                return tokenizer(text, add_special_tokens=False)["input_ids"] + [tokenizer.eos_token_id]

    else:
        bart = torch.hub.load("pytorch/fairseq", "bart.large").eval()

        def tokenize(text):
            return bart.encode(" " + text.strip()).tolist()[1:]

    delim = tokenize(args.delim)[:-1]
    index = build_index(args.input)

    index.save(args.output)
