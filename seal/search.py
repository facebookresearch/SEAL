# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import random

from more_itertools import chunked

from seal.retrieval import SEALSearcher
from seal.data import TopicsFormat, OutputFormat, get_query_iterator, get_output_writer

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--hybrid", default="none", choices=["none", "ensemble", "recall", "recall-ensemble"])
    parser.add_argument("--topics", type=str, metavar="topic_name", required=True, help="Name of topics.")
    parser.add_argument("--hits", type=int, metavar="num", required=False, default=100, help="Number of hits.")
    parser.add_argument(
        "--topics_format",
        type=str,
        metavar="format",
        default=TopicsFormat.DEFAULT.value,
        help=f"Format of topics. Available: {[x.value for x in list(TopicsFormat)]}",
    )
    parser.add_argument(
        "--output_format",
        type=str,
        metavar="format",
        default=OutputFormat.TREC.value,
        help=f"Format of output. Available: {[x.value for x in list(OutputFormat)]}",
    )
    parser.add_argument("--output", type=str, metavar="path", help="Path to output file.")
    parser.add_argument(
        "--max_passage", action="store_true", default=False, help="Select only max passage from document."
    )
    parser.add_argument(
        "--max_passage_hits",
        type=int,
        metavar="num",
        required=False,
        default=100,
        help="Final number of hits when selecting only max passage.",
    )
    parser.add_argument(
        "--max_passage_delimiter",
        type=str,
        metavar="str",
        required=False,
        default="#",
        help="Delimiter between docid and passage id.",
    )
    parser.add_argument("--remove_duplicates", action="store_true", default=False, help="Remove duplicate docs.")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--keep_samples", type=int, default=None)
    parser.add_argument("--chunked", type=int, default=0)
    SEALSearcher.add_args(parser)
    args = parser.parse_args()

    print(args)

    query_iterator = get_query_iterator(args.topics, TopicsFormat(args.topics_format))

    output_writer = get_output_writer(
        args.output,
        OutputFormat(args.output_format),
        "w",
        max_hits=args.hits,
        tag="SEAL",
        topics=query_iterator.topics,
        use_max_passage=args.max_passage,
        max_passage_delimiter=args.max_passage_delimiter,
        max_passage_hits=args.max_passage_hits,
    )

    if args.debug:
        query_iterator.order = query_iterator.order[:500]
        query_iterator.topics = {topic: query_iterator.topics[topic] for topic in query_iterator.order}

    if args.keep_samples is not None and args.keep_samples < len(query_iterator.order):
        random.seed(42)
        random.shuffle(query_iterator.order)
        query_iterator.order = query_iterator.order[: args.keep_samples]
        query_iterator.topics = {topic: query_iterator.topics[topic] for topic in query_iterator.order}

    searcher = SEALSearcher.from_args(args)

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
