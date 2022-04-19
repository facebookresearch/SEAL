Repo hosting my (`michelebevila@fb.com`) FAIR internship work. 

## Install instruction

Clone this repo with `--recursive` so that you also include submodules in `ext`.
```
git clone --recursive git@github.com:fairinternal/generative_retrieval.git
```

Compile and install `sdsl-lite`:
```
cd ext/sdsl-lite
env CFLAGS='-fPIC' CXXFLAGS='-fPIC' ./install.sh
```

Now install this library.
```
pip install --ignore-installed -e .
```

For training, you can use the version of `fairseq-py` included in `ext`.

## Prediction
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
python -m generative_retrieval.search \
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
from generative_retrieval.retrieval import GenerativeRetrievalSearcher, GenerativeRetrievalDocument

searcher = GenerativeRetrievalSearcher.load('sample.fm_index', 'checkpoint.pt')

```


## Building the FM-index
To build the FM-index, you have to put your retrieval corpus in a common format. We use a simple TSV file formatted as follows:
```
identifier  title   content
```
To do so, you may use the `scripts/data/convert_*_kb.sh` scripts (requiring `jq`).

Then you can build the FM-index with:
```
python scripts/data/build_fm_index.py corpus.tsv output --clean --hf_model facebook/bart-large --jobs 40 --include_title
```
The parameter `--jobs` only speeds up the tokenization at the moment. `--include_title` only makes sense if your retrieval corpus has non-empty titles.


