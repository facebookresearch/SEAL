# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from argparse import ArgumentParser
from collections import defaultdict
import json
import multiprocessing
from pathlib import Path
import random
import re
import tqdm
import math
import pickle
import jsonlines

import ftfy
from fuzzywuzzy import fuzz
from nltk.corpus import stopwords

banned = set(stopwords.words('english'))

def parse_args():

    parser = ArgumentParser()
    parser.add_argument('input')
    parser.add_argument('output')
    parser.add_argument(
        '--target',
        default = "span",
        choices = [
            "chunk",
            "span",
            "title",
            "code",
        ])
    parser.add_argument('--min_length', default=10, type=int)
    parser.add_argument('--max_length', default=10, type=int)
    parser.add_argument('--temperature', default=1.0, type=float)
    parser.add_argument('--jobs', default=1, type=int)
    parser.add_argument('--mark_target', action="store_true")
    parser.add_argument('--mark_silver', action="store_true")
    parser.add_argument('--n_samples', default=1, type=int)
    parser.add_argument('--id2code', default=None, type=str)
    parser.add_argument('--kb', required=True, type=str)
    parser.add_argument('--limit', default=300_000, type=int)
    parser.add_argument('--template', action="store_true")

    parser.add_argument('--mode', choices=["w", "a"], default="w")


    return parser.parse_args()

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

def span_iterator(tokens, ngrams=3, banned=banned):
    for i in range(len(tokens)):
        if tokens[i] not in banned:
            yield (i, i+ngrams)
        # for j in range(i+1, min(i+1+ngrams, len(tokens) + 1)):
        #     tok_orig = tokens[i:j]
        #     tok = [t for t in tok_orig if t not in banned]
        #     if not tok:
        #         break
        #     yield (i, j)

def extract_spans(text, source, n_samples, min_length, max_length, temperature=1.0):
    source = source.split("||", 1)[0]
    query_tokens = source.split()
    query_tokens_lower = [t.lower() for t in query_tokens]

    passage_tokens = text.split()
    passage_tokens_lower = [t.lower() for t in passage_tokens]

    matches = defaultdict(int)

    for i1, _ in enumerate(query_tokens_lower):
        j1 = i1+3
        str_1 = " ".join(query_tokens_lower[i1:j1])

        for (i2, j2) in span_iterator(passage_tokens_lower, 3):
            str_2 = " ".join(passage_tokens_lower[i2:j2])
            ratio = fuzz.ratio(str_1, str_2) / 100.0
            matches[i2] += ratio

    if not matches:
        indices = [0]

    else:
        indices, weights = zip(*sorted(matches.items(), key=lambda x: -(x[1])))
        weights = list(weights)
        sum_weights = float(sum([0] + weights))
        if sum_weights == 0.0 or not weights:
            indices = [0]
            weights = [1.0]
        else:
            weights = [math.exp(float(w) / temperature) for w in weights]
            Z = sum(weights)
            weights = [w / Z for w in weights]

        indices = random.choices(indices, weights=weights, k=n_samples)

    for i in indices:
        subspan_size = random.randint(min_length, max_length)
        span = " ".join(passage_tokens[i:i+subspan_size])
        yield span

def extract_spans_wrapper(args):
    return args[1], list(extract_spans(*args))

def clean(text):
    text = re.sub(r'\s+', ' ', text)
    text = ftfy.fix_text(text)
    text = text.replace('BULLET::::', '')
    text = text.replace('SECTION::::', '')
    text = text.strip()
    return text

def _iterator_span_get_arguments(data, mark_target, mark_silver, limit=float('inf'), template=False):
    for sample in tqdm.tqdm(data):
        
        if template:
            source = sample['meta']['template_questions'][0]
        else:
            source = sample['input']

        source = preprocess_question(source)
        
        if mark_target:
            source += " || body"   
                    
        i = 0
        for out in sample['output']:

            if i >= limit:
                break

            if "provenance" not in out:
                continue


            for ctx in out['provenance']:

                if i >= limit:
                    break

                idx = ctx['wikipedia_id'] + '-' + str(ctx['start_paragraph_id'])
                try:
                    title, text = kb[idx]
                except KeyError:
                    continue
                
                if mark_silver:
                    yield text, source + " || +"
                else:
                    yield text, source

                i += 1
            
def iterator_span(args):

    with jsonlines.open(args.input) as data:

        arg_it = _iterator_span_get_arguments(data, args.mark_target, args.mark_silver, args.limit, args.template)
        arg_it = ((text, source, args.n_samples, args.min_length, args.max_length, args.temperature) for text, source in arg_it)

        if args.jobs > 1:
            with multiprocessing.Pool(args.jobs) as pool:
                for source, spans in pool.imap(extract_spans_wrapper, arg_it):
                    for target in spans:
                        yield source, target
        else:
            for source, spans in map(extract_spans_wrapper, arg_it):
                for target in spans:
                    yield source, target

def iterator(args):

    if args.target == "code" and args.id2code:
        id2code = read_id2code(args.id2code)

    with jsonlines.open(args.input) as data:

        for sample in tqdm.tqdm(data):

            if args.template:
                source = sample['meta']['template_questions'][0]
            else:
                source = sample['input']

            source = preprocess_question(source)

            if args.target == "chunk" and args.mark_target:
                source += " || body"    
                
            elif args.target == "title" and args.mark_target:
                source += " || title"    
                
            elif args.target == "code" and args.mark_target:
                source += " || code"    

            else:
                raise ValueError("Wrong target")

            i = 0
            for out in sample['output']:
                    
                    if i >= args.limit:
                        break

                    if "provenance" not in out:
                        continue

                    for ctx in out['provenance']:

                        if i >= args.limit:
                            break

                        idx = ctx['wikipedia_id'] + '-' + str(ctx['start_paragraph_id'])

                        try:
                            title, text = kb[idx]
                        except KeyError:
                            continue

                        i += 1

                        if args.target == "chunk":
                            target = text
                            for _ in range(args.n_samples):
                                if args.mark_silver:
                                    yield source + " || +", target
                                else:
                                    yield source, target
                        
                        elif args.target == "title":
                            target = title + " @@"
                            for _ in range(args.n_samples):
                                if args.mark_silver:
                                    yield source + " || +", target
                                else:
                                    yield source, target

                        elif args.target == "code":
                            code = id2code.get(idx)
                            if not code: continue
                            target = code.strip() + " ||"
                            for _ in range(args.n_samples):
                                if args.mark_silver:
                                    yield source + " || +", target
                                else:
                                    yield source, target

                        else:
                            raise ValueError("Wrong target")

def preprocess(line):
    line = line.strip()
    if not line:
        return None
    try:
        idx, title, text = line.split('\t', 2)
    except:
        return None
    idx, title = idx.strip(), title.strip()
    text = text.replace('BULLET::::', '').strip()
    text = text.replace('Section::::', '').strip()
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    if not (idx and title and text):
        return None
    return idx, title, text

def preprocess_question(question):
    question = question.strip()
    question = question.replace('\n', ' / ')
    question = re.sub(r'\s+', ' ', question)
    return question

def main():

    global kb

    args = parse_args()

    if Path(args.kb + '.cached').exists():
        with open(args.kb + '.cached', 'rb') as fin:
            kb = pickle.load(fin)
    else:

        kb = {}
        with open(args.kb) as fin, multiprocessing.Pool(15) as pool:
            
            pipe = tqdm.tqdm(fin)
            pipe = map(preprocess, pipe)
            pipe = (x for x in pipe if x is not None)

            for idx, title, text in pipe:
                kb[idx] = (title, text)

        with open(args.kb + '.cached', 'wb') as fout:
            pickle.dump(kb, fout)

    with open(args.output + '.source', mode=args.mode) as src, open(args.output + '.target', mode=args.mode) as tgt:

        for source, target in iterator_span(args) if args.target == "span" else iterator(args):

            source = " " + source.strip()
            target = " " + target.strip()

            src.write(source + "\n")
            tgt.write(target + "\n")

if __name__ == '__main__':

    main()
