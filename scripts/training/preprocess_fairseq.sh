#!/usr/bin/env bash

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

DIR="$(cd "$(dirname "$0")" && pwd)"

DATASET=$1
# $DATASET must contain 4 plain text files, in which every line is a different training_fairseq/validation example:
# - $DATASET/train.source
# - $DATASET/train.target (same number of lines as train.source)
# - $DATASET/dev.source
# - $DATASET/dev.target (same number of lines as dev.source)

BART_FILES=$2
# $BART_FILES must contain the following files:
# - $BART_FILES/encoder.json - https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/encoder.json
# - $BART_FILES/vocab.bpe - https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/vocab.bpe
# - $BART_FILES/bart.large/dict.txt - https://dl.fbaipublicfiles.com/fairseq/models/bart.large.tar.gz (decompress)

echo "Processing $1"

# BPE training.
for SPLIT in train dev; do
    for LANG in "source" "target"; do
        python $DIR/multiprocessing_bpe_encoder.py \
            --encoder-json $BART_FILES/encoder.json\
            --vocab-bpe $BART_FILES/vocab.bpe \
            --inputs "$DATASET/$SPLIT.$LANG" \
            --outputs "$DATASET/$SPLIT.bpe.$LANG" \
            --workers 60 \
            --keep-empty;
    done
done

# Binarize the dataset.
fairseq-preprocess --source-lang "source" --target-lang "target" \
    --trainpref "$DATASET/train.bpe" \
    --validpref "$DATASET/dev.bpe" \
    --destdir "$DATASET/bin" \
    --workers 60 \
    --srcdict $BART_FILES/bart.large/dict.txt \
    --tgtdict $BART_FILES/bart.large/dict.txt;

cp "${BART_FILES}/bart.large/dict.txt" "${DATASET}/dict.source.txt"
cp "${BART_FILES}/bart.large/dict.txt" "${DATASET}/dict.target.txt"
