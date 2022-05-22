# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import csv
import random
import tqdm

from nltk.corpus import stopwords

banned = {
    "the", "The",
    "to", 
    "a", "A", "an", "An", 
    "he", "He", "his", "His", "him", "He's",  
    "she", "She", "her", "Her", "she's", "She's", 
    "it", "It", "its", "Its",  "it's", "It's",
    "and", "And",
    "or", "Or",
    "this", "This",
    "that", "That",
    "those", "Those",
    "these", "These",
    '"', '""', "'", "''",
}

def is_good(token):
    if token in banned:
        return False
    elif token[-1] in '?.!':
        return False
    elif token[0] in '([':
        return False
    return True

def preprocess_file(
    input_path,
    num_samples=1,
    num_title_samples=1,
    format="dpr", 
    delimiter='@@', 
    min_length_input=1,
    max_length_input=15,
    min_length_output=10, 
    max_length_output=10,
    full_doc_n=0,
    mark_pretraining=False,  
    ):
    
    if format == 'kilt':
        raise NotImplementedError
    elif format == 'dpr':
        with open(input_path, 'r', 2 ** 20) as f:
            next(f)
            
            f = csv.reader(f, delimiter='\t', quotechar='"')
            f = (l for l in f if len(l) == 3)
            
            for _, text, title in tqdm.tqdm(f):
                
                text = text
                title = title

                if text == title:
                    continue

                tokens = text.split()

                for _ in range(full_doc_n):
                    a = text.strip() + " || title"
                    if mark_pretraining:
                        a += " || p"
                    b = title.strip() + " " + delimiter
                    yield a, b

                sampled = 0
                failures = 0
                while sampled < num_title_samples and failures < 10:

                    if random.random() > 0.5:
                        len_a = random.randint(min_length_input, max_length_input)
                        idx_a = random.randint(0, max(0, len(tokens)-len_a))
                        a = ' '.join(tokens[idx_a:idx_a+len_a]).strip() + " || title"
                        if mark_pretraining:
                            a += " || p"
                        b = title.strip() + " " + delimiter
                    
                    else:

                        len_b = random.randint(min_length_output, max_length_output)
                        idx_b = random.randint(0, max(0, len(tokens)-len_b))

                        if not is_good(tokens[idx_b]):
                            failures += 1
                            continue

                        b = ' '.join(tokens[idx_b:idx_b+len_b]).strip()
                        a = title.strip() + ' || body'
                        if mark_pretraining:
                            a += " || p"
                    
                    yield a, b
                    sampled += 1

                sampled = 0
                failures = 0
                while sampled < num_samples and failures < 10:
                    len_a = random.randint(min_length_input, max_length_input)
                    len_b = random.randint(min_length_output, max_length_output)
                    idx_a = random.randint(0, max(0, len(tokens)-len_a))
                    idx_b = random.randint(0, max(0, len(tokens)-len_b))

                    if idx_a == idx_b or (not is_good(tokens[idx_b])):
                        failures += 1
                        continue

                    a = ' '.join(tokens[idx_a:idx_a+len_a]).strip() + ' || body'
                    if mark_pretraining:
                        a += " || p"
                    b = ' '.join(tokens[idx_b:idx_b+len_b]).strip()
                    yield a, b
                    sampled += 1
    else:
        raise ValueError

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('input')
    parser.add_argument('source')
    parser.add_argument('target')
    parser.add_argument('--delim', default="@@")
    parser.add_argument('--format', choices=['kilt', 'dpr'], default='dpr')
    parser.add_argument('--min_length_input', type=int, default=10)
    parser.add_argument('--max_length_input', type=int, default=10)
    parser.add_argument('--min_length_output', type=int, default=10)
    parser.add_argument('--max_length_output', type=int, default=10)
    parser.add_argument('--num_samples', type=int, default=10)
    parser.add_argument('--num_title_samples', type=int, default=3)
    parser.add_argument('--full_doc_n', type=int, default=1)
    parser.add_argument('--mark_pretraining', action="store_true")
    return parser.parse_args()

def main():
    args = parse_args()

    with open(args.source, 'w', 2 ** 20) as src, open(args.target, 'w', 2 ** 20) as tgt:
    
        for i, (s, t) in enumerate(preprocess_file(
            args.input,
            format=args.format,
            num_samples=args.num_samples,
            num_title_samples=args.num_title_samples,
            full_doc_n=args.full_doc_n,
            delimiter=args.delim,
            min_length_input=args.min_length_input,
            max_length_input=args.max_length_input,
            min_length_output=args.min_length_output,
            max_length_output=args.max_length_output,
            mark_pretraining=args.mark_pretraining,         
        )):

            if random.random() < 0.1:
                s = s.lower()

            s = " " + s
            t = " " + t

            src.write(s + '\n')
            tgt.write(t + '\n')

if __name__ == '__main__':

    main()
