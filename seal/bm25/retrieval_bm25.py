import math
from re import search
from typing import Optional, List, Dict, Union
import more_itertools
from collections import Counter

import tqdm
from more_itertools import chunked

from pyserini.search import SimpleSearcher, JSimpleSearcherResult
from pyserini.analysis import Analyzer, get_lucene_analyzer
from pyserini.search import querybuilder

import torch


from generative_retrieval import keys as rk
from generative_retrieval.retrieval import batch_generate_keys, GenerativeRetrievalSearcher, GenerativeRetrievalSearchResult, GenerativeRetrievalDocument

class GenerativeRetrievalBM25Searcher(SimpleSearcher):

    def __init__(
        self, 
        index_dir: str, 
        generative_searcher: Optional[GenerativeRetrievalSearcher] = None, 
        topk=50,
        expand_query=False,
        weights=0.0,
        ):
        super().__init__(index_dir=index_dir)
        self.generative_searcher = generative_searcher
        self.analyzer = Analyzer(get_lucene_analyzer())
        self.topk = topk
        self.expand_query = expand_query
        self.weights = weights
        self.progress = False
        counts = [self.generative_searcher.fm_index.get_count([i]) for i in range(self.generative_searcher.bart_tokenizer.vocab_size)]
        counts = torch.tensor(counts)
        tot = counts.sum()
        lprobs = (counts.float() + 0.5).log() - (tot.float() + 0.5).log()
        lprobs[counts == 0] = float('-inf')
        self.counts = counts.to(self.generative_searcher.device)
        self.lprobs = lprobs.to(self.generative_searcher.device)

    @staticmethod
    def visit(trie, tokens=[]):
        for k, new_trie in trie.items():
            tokens 

    def expand_queries(self, queries):
        expanded_queries = []

        for query, (keys, unigram_scores) in zip(queries, batch_generate_keys(
            self.generative_searcher, 
            queries, 
            constrained_generation=not self.generative_searcher.free_generation)):

            new_query = []
            
            # removing keys who are contained in one another
            trie = {}

            keys = [k for k, s in keys]
            keys = self.generative_searcher.bart_tokenizer.batch_decode(
                keys, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            keys = [k.strip() for k in keys]
            keys = [[ t for t in self.analyzer.analyze(k) if t != '@@'] for k in keys]
            keys = {tuple(k) for k in keys}
            keys = sorted(keys, key=lambda k: len(k))

            tokens = []

            for k in keys:
                if k:
                    tokens.append(k[-1])
                
            # tokens = [t for t, f in Counter(tokens).most_common(self.topk)]
            tokens = [' '.join([t] * int((float(f) ** self.weights) // 1)) for t, f in Counter(tokens).most_common(self.topk)]
            
            expansion = ' '.join(tokens)
            new_query = query + ' ' + expansion
            expanded_queries.append(new_query)
        
        return expanded_queries

    @classmethod
    def from_prebuilt_index(cls, prebuilt_index_name: str, generative_searcher: Optional[GenerativeRetrievalSearcher] = None):
        index_dir = SimpleSearcher.from_prebuilt_index(prebuilt_index_name).index_dir
        return cls(index_dir, generative_searcher)

    def search(self, query: str, k: int = 10, fields = dict()):
        return self.batch_search([query], k=k, fields=fields)[0]

    def _batch_search(self, queries: List[str], k: int = 10, fields = dict()) -> List[List[GenerativeRetrievalDocument]]:

        qids = list(map(str, range(len(queries))))
        all_results = []
        if self.generative_searcher is not None:
            threads = self.generative_searcher.jobs
            batch_size = self.generative_searcher.jobs
            queries = self.expand_queries(queries)
        else:
            batch_size = 10
            threads = 10

        with tqdm.tqdm(desc="Retrieving with BM25", total=len(qids), disable=not self.progress) as bar:
            for ii in chunked(range(len(qids)), batch_size):
                ii = list(ii)
                batch_queries = [queries[i] for i in ii]
                batch_qids = [qids[i] for i in ii]

                results = super().batch_search(batch_queries, batch_qids, k=k, threads=threads, fields=fields)
                results = [results[qid] for qid in batch_qids]
                if self.generative_searcher is not None:
                    results = [[GenerativeRetrievalDocument(
                        self.generative_searcher.docid2idx[r.docid], 
                        r.score, 
                        self.generative_searcher.fm_index, 
                        self.generative_searcher.bart_tokenizer, 
                        self.generative_searcher.wiki_token_id, 
                        keys=None) for r in rr] for rr in results]
                all_results.extend(results)
                bar.update(len(results))

        return all_results

    def search(self, query, k=10, fields=dict(), detokenize=True):
        return self.batch_search([query], k, fields, detokenize=detokenize)[0]

    def batch_search(self, queries: List[str], k: int = 10, fields = dict(), detokenize=None) -> List[List[GenerativeRetrievalDocument]]:
        if detokenize is None:
            detokenize = self.generative_searcher.detokenize

        if self.hybrid != 'none':
            retrieved = self.hybrid_batch_search(queries, k, fields)
        else:
            retrieved = self.vanilla_batch_search(queries, k, fields)
        if detokenize:
            retrieved = self.generative_searcher.detokenize_retrieved(retrieved)
        return retrieved


    def vanilla_batch_search(self, queries: List[str], k: int = 10, fields = dict()) -> List[List[GenerativeRetrievalDocument]]:

        qids = list(map(str, range(len(queries))))

        batch_size = self.generative_searcher.batch_size
        threads = self.generative_searcher.jobs

        if self.expand_query:

            new_queries = []

            with tqdm.tqdm(desc="Expanding queries", total=len(qids), disable=not self.progress) as bar:

                for batch in more_itertools.chunked(queries, batch_size):
                    batch = [q.strip() for q in batch]
                    new_queries += self.expand_queries(batch)
                    bar.update(len(batch))

            queries = new_queries

        retrieved = []

        with tqdm.tqdm(desc="Retrieving with BM25", total=len(qids), disable=not self.progress) as bar:
            
            for batch in chunked(queries, batch_size):
    
                ii = range(len(batch))
                ii = [str(i) for i in ii]
                results = super().batch_search(batch, ii, k=k, threads=threads, fields=fields)
                results = [results[i] for i in ii]    

                if self.generative_searcher is not None:
        
                    results = [[GenerativeRetrievalDocument(
                        self.generative_searcher.docid2idx[r.docid], 
                        r.score, 
                        self.generative_searcher.fm_index, 
                        self.generative_searcher.bart_tokenizer, 
                        delim1=self.generative_searcher.title_eos_token_id,
                        delim2=self.generative_searcher.code_eos_token_id, 
                        keys=None) for r in rr] for rr in results]
                
                retrieved += results
                bar.update(len(results))
        
        return retrieved

    def hybrid_batch_search(self, queries: List[str], k: int = 10, fields = dict()) -> List[List[GenerativeRetrievalDocument]]:
        
        if self.hybrid == 'ensemble':
            retrieved = self.vanilla_batch_search(queries, k=k)
            gs_progress = self.generative_searcher.progress
            self.generative_searcher.progress = self.progress
            retrieved_gs = self.generative_searcher.batch_search(queries, k=k, detokenize=False)
            self.generative_searcher.progress = gs_progress
        elif self.hybrid == 'recall' or self.hybrid == 'recall-ensemble':
            retrieved = self.vanilla_batch_search(queries, k=self.generative_searcher.fully_score)
            added_documents = [[d.idx for d in dd] for dd in retrieved]
            gs_progress = self.generative_searcher.progress
            self.generative_searcher.progress = self.progress
            retrieved_gs = self.generative_searcher.batch_search(queries, k=k, detokenize=False, added_documents=added_documents)
            self.generative_searcher.progress = gs_progress
            if self.hybrid == 'recall':
                return retrieved_gs

        hybrid_retrieved = []
        for rr, rrgs in zip(retrieved, retrieved_gs):
            rh = []
            done = set()
            hybrid_retrieved.append(rh)
            for i, (rgs, r) in enumerate(zip(rrgs, rr)):
                if rgs.docid not in done:
                    rgs.score = float(k - i)
                    done.add(rgs.docid)
                    rh.append(rgs)
                if r.docid not in done:
                    r.score = float(k - i - 0.5)
                    done.add(r.docid)
                    rh.append(r)
                if len(rh) >= k:
                    rh[:] = rh[:k]
                    break
        
        return hybrid_retrieved


    def doc(self, docid: Union[str, int]) -> Optional[GenerativeRetrievalDocument]:
        return self.generative_searcher.doc(docid)




                    # if self.topk > 0:
                    #     batch = [' ' + q for q in batch]
                    #     batch = self.generative_searcher.bart_tokenizer(batch, add_special_tokens=True, padding=False, truncation=True)['input_ids']
                    #     unigram_scores = rk.compute_unigram_scores(
                    #         self.generative_searcher.bart_scorer_model, 
                    #         batch, 
                    #         self.generative_searcher.fm_index, tolist=False)
                        
                    #     # topk mask
                    #     new_unigram_scores = torch.full_like(unigram_scores, float('-inf'))
                    #     top_values, top_indices = unigram_scores.topk(k=200, dim=-1)
                    #     new_unigram_scores.scatter_(-1, top_indices, top_values)
                    #     unigram_scores = new_unigram_scores

                    #     # scoring
                    #     unigram_scores = (unigram_scores + (1.0 - self.lprobs.exp()).log()) - (self.lprobs + (1.0 - unigram_scores.exp()).log())
                    #     unigram_scores[:, self.counts == 0] = 0.0

                    #     # topk
                    #     values, indices = torch.topk(unigram_scores, k=self.topk, dim=-1)
                    #     values = values.tolist()
                    #     indices = indices.tolist()
                        
                    #     for vv, ii, usb in zip(values, indices, unigram_scores_batch):
                    #         tt = self.generative_searcher.bart_tokenizer.convert_ids_to_tokens(ii)
                    #         tt = [t.lstrip('Ġ') for t in tt]

                    #         for v, t in zip(vv, tt):
                    #             t = self.analyzer.analyze(t)
                    #             if len(t) != 1:
                    #                 continue
                    #             t = t[0]
                    #             usb[t] = max(usb.get(t, 1.0), v)
                    
                    # for usb in unigram_scores_batch:
                    #     boolean_query_builder = querybuilder.get_boolean_query_builder()
                    #     for t, v in usb.items():
                    #         print(t, v)
                    #         term = querybuilder.get_term_query(t)
                    #         boost = querybuilder.get_boost_query(term, v)
                    #         should = querybuilder.JBooleanClauseOccur['should'].value
                    #         boolean_query_builder.add(boost, should)
                    #     query = boolean_query_builder.build()
                    #     new_queries.append(query)
                    # else:
                    # for usb in unigram_scores_batch:
                    #     q = []
                    #     for t, v in usb.items():
                    #         if self.deduplicate_expansion:
                    #             v = 1
                    #         else:
                    #             v = math.log(v)
                    #             v = math.floor(v)
                    #             v = 1 + int(v)
                    #         for _ in range(v):
                    #             q.append(t)
                    #     new_queries.append(' '.join(q))
                
                    # bar.update(len(batch))

