# SEAL: <u>S</u>earch <u>E</u>ngines with <u>A</u>utoregressive <u>L</u>Ms
This repo hosts the code for our paper, SEAL.

**Michele Bevilacqua, Giuseppe Ottaviano, Patrick Lewis, Wen-tau Yih, Sebastian Riedel, Fabio Petroni**,
 *Autoregressive Search Engines: Generating Substrings as Document Identifiers.* 2022.

We propose a approach to retrieval that uses guided LM decoding to search for occurrences of ngrams of any size in an 
arbitrary large collection of documents. Constrained decoding blocks the generation of ngrams that never appear in the 
corpus: generated ngrams are always grounded in one or multiple documents in the retrieval corpus. Documents are then scored by aggregating the scores for individual generated 
"identifiers". 

We use the Ferragina Manzini index, an opportunistic, compressed suffix array as the unified data structure for constrained decoding,
retrieval and full-text storage.

## Install
We assume that `pytorch` is already available in your environment. SEAL has been tested with version 1.11.

Clone this repo with `--recursive` so that you also include the submodule in `ext`.
```commandline
git clone --recursive git@github.com:facebookresearch/SEAL.git
```

Compile and install `sdsl-lite`:
```commandline
pushd res/repos/sdsl-lite
env CFLAGS='-fPIC' CXXFLAGS='-fPIC' ./install.sh
popd
```

Install other dependencies:
```commandline
pip install -r requirements
```

Now install this library.
```commandline
pip install --ignore-installed -e .
```
## Data
We provide the weights of our Natural Questions and KILT models.
* [BART large (NQ)](URL)
* [BART large (KILT)](URL)

We also make available the indices for both the NQ and KILT retrieval corpora. Both corpora have been tokenized using 
the tokenizer from BART.
* [FM-index, `psgs_w100` (NQ)](URL)
* [FM-index (KILT)](URL)

## Retrieval
Suppose you start with the following files.
```commandline
sample.json
checkpoint.pt
sample.fm_index.fmi
sample.fm_index.oth
```
* `sample.json`: A JSON file in DPR format containing queries.
* `checkpoint.pt`: The fine-tuned BART checkpoint.
* `sample.fm_index.*`: The serialized FM-index.

### Command-line interface
To run prediction, launch the following command:

```commandline
python -m seal.search \
    --topics_format dpr --topics sample.json \
    --output_format dpr --output sample.output.json \
    --checkpoint checkpoint.pt \
    --fm_index sample.fm_index \
    --jobs 75 --progress --device cuda:0 --batch_size 20 \
    --beam 5
```
The script will generate the DPR prediction file `sample.output.json`. Other supported formats are KILT and BEIR.

### The `Searcher` class
Our codebase relies on a `pyserini`-like searcher class, that incapsulates both constrained decoding and retrieval. You
can use it programmatically:
```python
from seal.retrieval import SEALSearcher

searcher = SEALSearcher.load('sample.fm_index', 'checkpoint.pt')
searcher.include_keys = True

query = "causes of co2 increase"

for i, doc in enumerate(searcher.batch_search(query, k=3)):
    print(i, doc.score, doc.docid, *doc.text(), sep='\t')
    print("Matched:")
    for ngram, freq, score in sorted(doc.keys, reverse=True, key=lambda x:x[2]):
        print(score, freq, repr(ngram), sep='\t')
```

## Constrained decoding

### Building the FM-index (CLI)
To most straightforward way to build the FM-index is to use the script we have provided in `scripts/build_fm_index.py`! 
You only need  to put your retrieval corpus in a very simple TSV format as in the following example:
```
doc1    Doc 1   This is a sample document
doc2    Doc 2   This is another sample documents
doc3    Doc 3   And here you find the final one
```
Fields are: 
* document id
* document title
* text 

Then you can build the FM-index with:
```commandline
FILE_I=res/sample/sample_corpus.tsv
FILE_O=res/sample/sample_corpus.fm_index

python scripts/data/build_fm_index.py \
    $FILE_I $FILE_O \
    --clean --hf_model facebook/bart-large  \
    --jobs 40 --include_title \
```
The parameter `--jobs` only speeds up the tokenization at the moment. `--include_title` only makes sense if your retrieval corpus has non-empty titles.

### Building the FM-index (Python)
