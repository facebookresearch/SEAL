# SEAL training

## Preprocessing

We assume you have downloaded in the `$DATASET` folder the following DPR files:
* `$DATASET/train.json` [[link](https://dl.fbaipublicfiles.com/dpr/data/retriever/biencoder-nq-train.json.gz)]
* `$DATASET/dev.json` [[link](https://dl.fbaipublicfiles.com/dpr/data/retriever/biencoder-nq-dev.json.gz)]

You will now need to run `scripts/training/make_supervised_dpr_dataset.py` to create training and validation examples:

```bash
for FILE in train dev ; do

    python scripts/training/make_supervised_dpr_dataset.py \
        $DATASET/nq-$FILE.json $DATASET/$FILE \
        --target title \
        --mark_target \
        --mark_silver \
        --n_samples 3 \
        --mode a
    
    python scripts/training/make_supervised_dpr_dataset.py \
        $DATASET/nq-$FILE.json $DATASET/$FILE \
        --target span \
        --mark_target \
        --mark_silver \
        --n_samples 10 \
        --mode a

done
```

`scripts/training` also contains an analogous preprocessing scripts that takes care of KILT files.

If you want to add unsupervised examples by sampling spans from the retrieval corpus download [DPR's preprocessed chunks](https://dl.fbaipublicfiles.com/dpr/wikipedia_split/psgs_w100.tsv.gz) and run the following:

```bash
python scripts/training/make_unsupervised_dataset.py \
    $DATASET/psgs_w100.tsv $DATASET/unsupervised \
    --format dpr --num_samples 3 --num_title_samples 1 --full_doc_n 1 --mark_pretraining

cat $DATASET/unsupervised.source >> $DATASET/train.source
cat $DATASET/unsupervised.target >> $DATASET/train.target
```

The final step is running `fairseq-preprocess`. We have prepared an easy to use script in `scripts/training/preprocess_fairseq.sh`.
The instructions can be found inside of it.

## Training

Check out `scripts/training/preprocess`!

____

## License
SEAL is licensed under the CC-BY-NC 4.0 license. The text of the license can be found [here](https://github.com/facebookresearch/SEAL/blob/main/LICENSE).
