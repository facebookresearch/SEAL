import random
import os

from more_itertools import chunked

from generative_retrieval.retrieval import GenerativeRetrievalSearcher
from generative_retrieval.bm25.retrieval_bm25 import GenerativeRetrievalBM25Searcher
from generative_retrieval.data import TopicsFormat, OutputFormat, get_query_iterator, get_output_writer

if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--bm25_index', type=str, default=None)
    parser.add_argument('--bm25_expand_query', action='store_true')
    parser.add_argument('--bm25_weights', type=float, default=0.0)
    parser.add_argument('--bm25_topk', default=25, type=int)
    parser.add_argument('--hybrid', default='none', choices=['none', 'ensemble', 'recall', 'recall-ensemble'])
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
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--keep_samples', type=int, default=None)
    parser.add_argument('--chunked', type=int, default=0)
    GenerativeRetrievalSearcher.add_args(parser)
    args = parser.parse_args()

    print(args)

    query_iterator = get_query_iterator(args.topics, TopicsFormat(args.topics_format))

    output_writer = get_output_writer(args.output, OutputFormat(args.output_format), 'w',
                                    max_hits=args.hits, tag="GR", topics=query_iterator.topics,
                                    use_max_passage=args.max_passage,
                                    max_passage_delimiter=args.max_passage_delimiter,
                                    max_passage_hits=args.max_passage_hits)

    if args.debug:
        query_iterator.order = query_iterator.order[:500]
        query_iterator.topics = {topic: query_iterator.topics[topic] for topic in query_iterator.order}

    if args.keep_samples is not None and args.keep_samples < len(query_iterator.order):
        random.seed(42)
        random.shuffle(query_iterator.order)
        query_iterator.order = query_iterator.order[:args.keep_samples]
        query_iterator.topics = {topic: query_iterator.topics[topic] for topic in query_iterator.order}

    searcher = GenerativeRetrievalSearcher.from_args(args)
    if args.bm25_index:
        if os.path.exists(args.bm25_index):
            searcher = GenerativeRetrievalBM25Searcher(args.bm25_index, searcher)
        else:
            searcher = GenerativeRetrievalBM25Searcher.from_prebuilt_index(args.bm25_index, searcher)
        searcher.topk = args.bm25_topk
        searcher.expand_query = args.bm25_expand_query
        searcher.weights = args.bm25_weights
        searcher.progress = args.progress
        searcher.generative_searcher.progress = False
        searcher.hybrid = args.hybrid

    with output_writer:
        if args.chunked <= 0:
            topic_ids, texts = zip(*query_iterator)
            for topic_id, hits in zip(topic_ids, searcher.batch_search(texts, k=args.hits)):
                output_writer.write(topic_id, hits)
        else:
            for batch_query_iterator in chunked(query_iterator, args.chunked):
                topic_ids, texts = zip(*batch_query_iterator)
                for topic_id, hits in zip(topic_ids, searcher.batch_search(texts, k=args.hits)):
                    output_writer.write(topic_id, hits)
