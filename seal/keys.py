# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math
from collections import defaultdict
from heapq import heappop, heappush
from itertools import chain, islice, product
from typing import *

import torch
from more_itertools import chunked
from tqdm import tqdm

from seal import FMIndex

def deduplicate(list_of_lists):
    present = set()
    result = []
    for el in list_of_lists:
        x = el
        if isinstance(el[0], float):
            el = el[1]
        if isinstance(el, torch.Tensor):
            t_el = tuple(el.tolist())
        else:
            t_el = tuple(el)
        if t_el in present:
            continue
        else:
            present.add(t_el)
            result.append(x)
    return result


def decompose_query_into_keys(query, word_tokenizer, length=3):
    strings = set()
    query = query.strip()
    tokens = word_tokenizer(query)
    tokens = [t.text for t in tokens]
    for i in range(len(tokens)):
        for j in range(i + 1, min(1 + len(tokens), i + length + 1)):
            span = tokens[i:j]
            for upper in product(*([[True, False]] * (j - i))):
                ss = [s[0].upper() + s[1:] if u else s for u, s in zip(upper, span)]
                ss = " " + " ".join(ss)
                strings.add(ss)
    strings = list(strings)
    return strings


def strip(seq, symbols_start, symbols_end):
    i = 0
    while i < len(seq) and seq[i] in symbols_start:
        i += 1
    j = len(seq)
    while j > i and seq[j - 1] in symbols_end:
        j -= 1
    return seq[i:j]


@torch.inference_mode()
def rescore_keys(model, inputs, list_of_decoded, batch_size=100, length_penalty=0.0, progress_bar=False, prefix=[],
                 strip_from_bos=[], strip_from_eos=[]):
    device = next(model.parameters()).device

    if inputs is None:
        batch_in = [[model.config.bos_token_id, model.config.eos_token_id]] * len(list_of_decoded)
    else:
        batch_in = list(inputs)

    list_of_decoded = [[x[1] if isinstance(x[0], float) else x for x in xx] for xx in list_of_decoded]

    maxlen = max([len(i) for i in batch_in])

    input_ids = [i + ([model.config.pad_token_id] * (maxlen - len(i))) for i in batch_in]
    input_ids = [torch.LongTensor(i).to(device) for i in input_ids]
    input_ids = torch.stack(input_ids, 0)
    attention_mask = input_ids != model.config.pad_token_id
    attention_mask = attention_mask.byte()

    encoder_outputs = model._prepare_encoder_decoder_kwargs_for_generation(
        input_ids, {'attention_mask': attention_mask})['encoder_outputs'].last_hidden_state

    decoder_inputs = enumerate(list_of_decoded)
    decoder_inputs = [(idx, di) for idx, ddi in decoder_inputs for di in ddi]

    all_out = {i: [] for i, _ in enumerate(list_of_decoded)}

    for batch in chunked(tqdm(decoder_inputs) if progress_bar else decoder_inputs, batch_size):

        idxs = []
        batch_in_decoder_orig = []
        batch_in_decoder = []
        for i, di in batch:
            stripped = [model.config.decoder_start_token_id] + prefix + strip(di, strip_from_bos, strip_from_eos)
            if stripped:
                idxs.append(i)
                batch_in_decoder_orig.append(di)
                batch_in_decoder.append(stripped)

        batch_in_decoder = [torch.LongTensor(di) for di in batch_in_decoder]
        batch_in_decoder = [
            torch.cat(
                [torch.LongTensor([model.config.decoder_start_token_id]), di]
            ) if di[0] != model.config.decoder_start_token_id else di for di in batch_in_decoder]
        maxlen = max([len(di) for di in batch_in_decoder])

        batch_decoder_input_ids = [
            torch.cat(
                [di, torch.LongTensor([model.config.pad_token_id] * (maxlen - len(di)))])
            for di in batch_in_decoder]
        batch_decoder_input_ids = [di for di in batch_decoder_input_ids]
        batch_decoder_input_ids = torch.stack(batch_decoder_input_ids, 0).to(device)

        batch_input_ids = torch.stack([input_ids[idx] for idx in idxs], 0)
        batch_attention_mask = torch.stack([attention_mask[idx] for idx in idxs], 0)
        batch_encoder_outputs = torch.stack([encoder_outputs[idx] for idx in idxs], 0)

        logits = model(
            input_ids=batch_input_ids,
            attention_mask=batch_attention_mask,
            encoder_outputs=(batch_encoder_outputs, None, None),
            decoder_input_ids=batch_decoder_input_ids[:, :-1],
        ).logits

        logprobs = logits.log_softmax(-1)

        logprobs = torch.gather(logprobs, -1, batch_decoder_input_ids[:, 1:].unsqueeze(-1))
        logprobs[batch_decoder_input_ids[:, 1:] < 2] = 0.0
        logprobs = logprobs[:, len(prefix):]
        logprobs = logprobs.squeeze(-1).sum(-1)
        logprobs = logprobs.tolist()

        for i, di, bdi, ll in zip(idxs, batch_in_decoder_orig, batch_decoder_input_ids, logprobs):
            sco = ll / (len(di) ** length_penalty)
            all_out[i].append((sco, di))

    return [v for k, v in sorted(all_out.items())]


# @torch.inference_mode()
@torch.no_grad()
def compute_unigram_scores(model, inputs, index: FMIndex, tokenizer=None, tolist=True, temperature=1.0, prefix=[]):
    device = next(model.parameters()).device

    if isinstance(inputs[0], str):
        batch = tokenizer(inputs, padding=True, return_tensors='pt')
    else:
        batch_in = list(inputs)
        maxlen = max([len(i) for i in batch_in])
        input_ids = [i + ([model.config.pad_token_id] * (maxlen - len(i))) for i in batch_in]
        input_ids = [torch.LongTensor(i).to(device) for i in input_ids]
        input_ids = torch.stack(input_ids, 0)
        attention_mask = input_ids != model.config.pad_token_id
        attention_mask = attention_mask.byte()
        batch = dict(input_ids=input_ids, attention_mask=attention_mask)

    batch = {k: v.to(device) for k, v in batch.items()}

    decoder_input_ids = torch.full_like(batch['input_ids'][:, :1 + len(prefix)], model.config.decoder_start_token_id)
    for i, idx in enumerate(prefix, start=1):
        decoder_input_ids[:, i] = idx

    logits = model(**batch, decoder_input_ids=decoder_input_ids).logits[:, 0 + len(prefix)]

    if temperature != 1.0:
        logits /= temperature
    logprobs = logits.log_softmax(-1)

    if tolist:
        return logprobs.tolist()
    else:
        return logprobs

def aggregate_evidence(ngrams_and_scores: List[Tuple[List[int], float]], unigram_scores: Optional[List[float]] = None,
                       index: Optional[FMIndex] = None, max_occurrences_1: int = 1500,
                       max_occurrences_2: int = 10_000_000, n_docs_complete_score: int = 500, alpha: float = 2.0,
                       beta: float = 0.8, length_penalty: float = 0.0, use_fm_index_frequency: bool = True,
                       add_best_unigrams_to_ngrams: bool = False, use_top_k_unigrams=1000, sort_by_length=False,
                       sort_by_freq=False, smoothing=5.0, allow_overlaps=False, single_key=0.0,
                       single_key_add_unigrams=False, unigrams_ignore_free_places=False) -> Tuple[List[int], List[float]]:

    def repetition(ngram, score, coverage):
        if not coverage:
            return score
        ngram = set(ngram)
        coeff = 1.0 - beta + (beta * len(ngram.difference(coverage)) / len(ngram))
        return coeff * score

    ntokens = float(index.beginnings[-1])

    ngrams_and_scores = [(ngram.tolist() if isinstance(ngram, torch.Tensor) else ngram, sr) for ngram, sr in ngrams_and_scores]
    counts = {tuple(): len(index)}

    if not use_fm_index_frequency:
        try:
            cutoff = sorted(ngrams_and_scores, key=lambda x: x[1])[0][1] - 0.1
        except IndexError as e:
            print(ngrams_and_scores)
            raise e
    else:
        cutoff = None

    unigrams = {0, 1, 2}
    for i in range(len(ngrams_and_scores)):
        ngram, sr = ngrams_and_scores[i]
        if len(ngram) == 1:
            unigrams.add(ngram[0])
        count = index.get_count(ngram)

        counts[tuple(ngram)] = count

        if count == 0:
            sco = 0.0
        elif use_fm_index_frequency:
            sr -= 1e-10
            sr *= (1.0 - length_penalty) ** (len(ngram) - 1.0)
            snr = math.log((count + smoothing) / (ntokens + smoothing))
            sco = \
                (sr + math.log(1 - math.exp(snr))) - \
                (snr + math.log(1 - math.exp(sr)))

            sco = max(sco, 0.0)
            sco **= alpha
        else:
            sco = sr - cutoff
            sco = max(sco, 0.0)
            sco *= (1.0 - length_penalty) ** (len(ngram) - 1.0)
            sco **= alpha

        ngrams_and_scores[i] = (ngram, sco)

    if unigram_scores is not None:

        unigram_scores = unigram_scores[:]
        best = sorted(range(len(unigram_scores)), reverse=True, key=lambda i: unigram_scores[i])
        best = best[:use_top_k_unigrams]
        best = set(best)
        unigram_scores = [s if i in best else float('-inf') for i, s in enumerate(unigram_scores)]
        for i in range(len(unigram_scores)):

            if i in unigrams:
                unigram_scores[i] = 0.0
                continue

            sr = unigram_scores[i]
            ngram = [i]

            count = index.get_count(ngram)
            if count == 0:
                sco = 0.0
            elif use_fm_index_frequency:
                snr = math.log((count + smoothing) / (ntokens + smoothing))
                sco = \
                    (sr + math.log(1 - math.exp(snr))) - \
                    (snr + math.log(1 - math.exp(sr)))

                sco = max(sco, 0.0)

            else:
                sco = sr - cutoff
                sco = max(sco, 0.0)
                sco **= alpha

            if sco == 0.0:
                unigram_scores[i] = 0.0
                continue

            unigram_scores[i] = sco

        if add_best_unigrams_to_ngrams:
            best_unigrams = sorted(list(range(len(unigram_scores))), key=lambda x: -unigram_scores[x])[:len(ngrams_and_scores)]
            for i in best_unigrams:
                counts[tuple([i])] = index.get_count([i])
                ngrams_and_scores.append(([i], unigram_scores[i]))
 
    # rare ngrams (occurring less than max_hits) --> used for the first stage and full scoring
    rare_ngrams = defaultdict(float)
    # frequent ngrams --> used just for full scoring
    freq_ngrams = defaultdict(float)
    # computing scores for all ngrams
    for ngram, sco in ngrams_and_scores:

        count = index.get_count(ngram)
        if count > max_occurrences_2:
            continue
        elif sco == 0.0:
            continue
        elif count > max_occurrences_1 or sco < 0.0:
            ngrams = freq_ngrams
            # ngrams = rare_ngrams
        else:
            ngrams = rare_ngrams

        ngram = tuple(ngram)
        ngrams[ngram] = sco

    # else:
    rare_ngrams = {k: v for k, v in sorted(rare_ngrams.items(), key=lambda x: x[1], reverse=True)}
    # rare_ngrams = remove_redundant_ngrams(rare_ngrams)
    freq_ngrams = {k: v for k, v in sorted(freq_ngrams.items(), key=lambda x: x[1], reverse=True)}
    # freq_ngrams = remove_redundant_ngrams(freq_ngrams)
    all_ngrams = {k: v for k, v in \
                  sorted(
                      chain(rare_ngrams.items(), freq_ngrams.items()),
                      key=lambda x: x[1], reverse=True)}

    covered_points = set()
    first_stage = defaultdict(lambda: [0.0, [], [[], 0.0]])

    for ngram, sco in rare_ngrams.items():
        # idfs[ngram] = idf(ngram, index)

        # each ngram only considered once for doc
        doc_done = defaultdict(set)

        for row in islice(range(*index.get_range(list(ngram))), max_occurrences_1):

            tok_end = index.locate(row)
            tok_start = tok_end - len(ngram)
            doc = index.get_doc_index(tok_end)
            new = all([i not in covered_points for i in range(tok_start, tok_end)])

            if sort_by_length:
                order = (len(ngram), sco)
                max_order = (len(first_stage[doc][2][0]), first_stage[doc][2][1])
            elif sort_by_freq:
                order = (-counts[tuple(ngram)], sco)
                max_order = (-counts[tuple(first_stage[doc][2][0])], first_stage[doc][2][1])
            else:
                order = sco
                max_order = first_stage[doc][2][1]

            if order > max_order:
                first_stage[doc][2] = [ngram, sco]

            if new:

                for tok in range(tok_start, tok_end):
                    covered_points.add(tok)

            if new or allow_overlaps:

                if ngram not in doc_done[doc]:
                    doc_done[doc].add(ngram)
                    first_stage[doc][0] += sco
                    first_stage[doc][1].append((ngram, sco))

    for doc, doc_info in first_stage.items():

        current_coverage = set()
        current_score = 0.0
        for i in range(len(doc_info[1])):
            tt, sco = doc_info[1][i]
            tts = set(tt)

            new_sco = repetition(tts, sco, current_coverage)
            current_score += new_sco
            doc_info[1][i] = [tt, new_sco]
            current_coverage |= tts
        doc_info[0] = current_score

    to_fully_score = sorted(first_stage.items(),
                            key=lambda x: (1.0 - single_key) * (-x[1][0]) + single_key * (-x[1][2][1]))[:n_docs_complete_score]
    results = defaultdict(lambda:
                          [
                              0.0,  # score
                              [],  # ngrams found
                              None,  # places filled
                              None,  # full doc tokens
                              [[], 0.0]  # max ngram
                          ])

    trie = {}
    for ngram, score in all_ngrams.items():
        if len(ngram) < 1 or score <= 0.0:
            continue
        current = trie
        for t in ngram:
            current = current.setdefault(t, {})
        current[-1] = score

    for doc, _ in to_fully_score:

        doc_tokens = [2] + index.get_doc(doc)[:-1]
        results[doc][3] = doc_tokens

        if unigram_scores is not None:
            type_scores = {t: unigram_scores[t] for t in doc_tokens}
        else:
            type_scores = {t: 0.0 for t in doc_tokens}

        matches = {}
        open_matches = []
        for i in range(len(doc_tokens)):
            open_matches = [(m.get(doc_tokens[i]), l + 1, n) for (m, l, n) in open_matches] + [
                (trie.get(doc_tokens[i]), 1, [])]
            for _, _, n in open_matches:
                n.append(doc_tokens[i])
            new_open_matches = []
            while open_matches:
                m, l, n = open_matches.pop()
                if m is None:
                    continue
                new_open_matches.append((m, l, n))
                if -1 in m:
                    start = i - l + 1
                    end = i + 1
                    matches.setdefault(tuple(n), [m[-1], []])[1].append((start, end))
            open_matches = new_open_matches

        greedy_matches = []
        for n, (s, d) in matches.items():

            if sort_by_length:
                order = (-len(n), -s)
                max_order = (-len(results[doc][4][0]), -results[doc][4][1])

            elif sort_by_freq:
                order = (counts[tuple(n)], -s)
                max_order = (counts[tuple(results[doc][4][0])], -results[doc][4][1])
            else:
                order = -s
                max_order = -results[doc][4][1]

            for (i, j) in d:
                heappush(greedy_matches, (-s, n, s, i, j))

            if order < max_order:
                results[doc][4] = [n, s]

        current_coverage = set()
        ngrams = []
        prev = None
        f = 0
        free = [True] * len(doc_tokens)

        while greedy_matches:

            order, n, s, i, j = heappop(greedy_matches)

            n_set = set(n)

            if prev == n:
                new_s = ngrams[-1][1]
            elif not n_set:
                new_s = 0.0
            else:
                new_s = repetition(n_set, s, current_coverage)

            if new_s <= 0.0:
                continue

            if allow_overlaps or all(free[i:j]):
                pass
            else:
                continue

            if prev == n:
                f += 1
                ngrams[-1] = (n, new_s)
            else:
                f = 1
                prev = n
                current_coverage |= n_set
                ngrams.append((n, new_s))

            free[i:j] = [False] * (j - i)

        if unigrams_ignore_free_places:
            free = [True for _ in free]

        single_key_score = results[doc][4][1]
        multi_key_score = sum([s for n, s in ngrams])
        unigram_score = 0.0

        for t, f in Counter([t for t, b in zip(doc_tokens, free) if b]).items():
            s = type_scores[t]
            if s > 0.0:
                n = (t,)
                s = repetition(n, s, current_coverage)
                if s != 0.0:
                    unigram_score += s
                    ngrams.append((n, s))

        if single_key_add_unigrams:
            single_key_score += unigram_score
        multi_key_score += unigram_score

        results[doc][0] = (1.0 - single_key) * multi_key_score + single_key * single_key_score
        results[doc][1] = ngrams

    results = {k: v for k, v in sorted(results.items(), key=lambda x: -x[1][0])}
    return results, all_ngrams
