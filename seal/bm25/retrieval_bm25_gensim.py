from typing import Optional, List, Dict, Union
import csv
import copy
import re
import multiprocessing
from nltk.tokenize import word_tokenize
from gensim.summarization.bm25 import BM25
from tqdm import tqdm

class GensimDocument:

    def __init__(
            self,
            idx,
            docid,
            text,
            title,
        ):
        self.idx = idx
        self.docid = docid
        self._text = text
        self._title = title
        self.score = None

    def id(self):
        return self.idx

    def text(self):
        return self._title, self._text

    def __repr__(self):
        return f'<GRDocument: {self.idx}, "{self.text()[:30]}[...]">'

def preprocess(string):
    string = string.lower().strip()
    string = re.sub(r'\s+', ' ', string)
    return word_tokenize(string)

def preprocess(string):
    string = string.lower().strip()
    # string = re.sub(r'\s+', ' ', string)
    # return word_tokenize(string)
    return string.split()

class GensimBM25Searcher:

    def __init__(self, path):

        self.docs = []
        self.docid2idx = {}
        corpus = []
        with open(path) as fin:
            next(fin)
            fin = tqdm(fin)
            fin = csv.reader(fin, delimiter="\t", quotechar='"')
            for idx, (docid, text, title) in enumerate(fin):
                doc = GensimDocument(idx, docid, text, title)
                self.docid2idx[docid] = idx
                self.docs.append(doc)

        texts = (d._text for d in self.docs)
        texts = tqdm(texts)
        with multiprocessing.Pool(20) as pool:
            for tokens in pool.imap(preprocess, texts):
                corpus.append(tokens)

        self.bm25 = BM25(corpus)

    def search(self, query: str, k: int = 10):
        query = preprocess(query)
        scores = self.bm25.get_scores(query)
        indices = sorted(range(len(scores)), key=lambda x: -scores[x])[:k]

        to_return = []
        for idx in indices:
            doc = copy.deepcopy(self.docs[idx])
            doc.score = scores[idx]
            to_return.append(doc)

        return to_return

    def batch_search(self, queries: List[str], k: int = 10) -> List[List[GensimDocument]]:
        return [self.search(query, k) for query in tqdm(queries)]
    
    def doc(self, docid: Union[str, int]) -> Optional[GensimDocument]:
        if isinstance(docid, str):
            idx = self.docid2idx[docid]
        else:
            idx = docid
        return copy.deepcopy(self.docs[idx])

if __name__ == '__main__':

    from more_itertools import chunked
    from generative_retrieval.data import TopicsFormat, OutputFormat, get_query_iterator, get_output_writer
    import argparse
    import multiprocessing

    parser = argparse.ArgumentParser()

    parser.add_argument('--kb_path', type=str, metavar='kb_path', required=True, help="Name of topics.")
    parser.add_argument('--topics', type=str, metavar='topic_name', required=True,
                        help="Name of topics.")
    parser.add_argument('--hits', type=int, metavar='num',
                        required=False, default=100, help="Number of hits.")
    parser.add_argument('--topics_format', type=str, metavar='format', default=TopicsFormat.DEFAULT.value,
                        help=f"Format of topics. Available: {[x.value for x in list(TopicsFormat)]}")
    parser.add_argument('--output_format', type=str, metavar='format', default=OutputFormat.TREC.value,
                        help=f"Format of output. Available: {[x.value for x in list(OutputFormat)]}")
    parser.add_argument('--output', type=str, metavar='path',
                        help="Path to output file.")
    parser.add_argument('--max_passage',  action='store_true',
                        default=False, help="Select only max passage from document.")
    parser.add_argument('--max_passage_hits', type=int, metavar='num', required=False, default=100,
                        help="Final number of hits when selecting only max passage.")
    parser.add_argument('--max_passage_delimiter', type=str, metavar='str', required=False, default='#',
                        help="Delimiter between docid and passage id.")
    parser.add_argument('--remove_duplicates', action='store_true', default=False, help="Remove duplicate docs.")
    args = parser.parse_args()

    print(args)

    query_iterator = get_query_iterator(args.topics, TopicsFormat(args.topics_format))

    output_writer = get_output_writer(args.output, OutputFormat(args.output_format), 'w',
                                    max_hits=args.hits, tag="BM25-gensim", topics=query_iterator.topics,
                                    use_max_passage=args.max_passage,
                                    max_passage_delimiter=args.max_passage_delimiter,
                                    max_passage_hits=args.max_passage_hits)

    searcher = GensimBM25Searcher(args.kb_path)

    def predict(text):
        return searcher.search(text, k=args.hits)

    with output_writer, multiprocessing.Pool(60) as pool:
        topic_ids, texts = zip(*query_iterator)
        # results = map(predict, texts)
        results = pool.imap(predict, texts)
        results = tqdm(results, total=len(texts))
        for topic_id, hits in zip(topic_ids, results):
            output_writer.write(topic_id, hits)
        