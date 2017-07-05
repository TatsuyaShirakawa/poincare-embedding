# poincare-embedding

This codes implement Poincar\'e Embedding introduced in the following paper:

[Maximilian Nickel and Douwe Kiela, "Poincar\'e Embeddings for Learning Hierarchical Representations'", arXiv preprint arXiv:1705.08039, 2017.](https://arxiv.org/abs/1705.08039)
    
## Build

```shell
cd poincare-embedding
mkdir work & cd work
cmake ..
make
```

## Data Creation

You can create WordNet noun hypernym pairs as follows (assumed you are in work directory)

```shell
python ../scripts/create_wordnet_noun_hierarchy.py ./wordnet_noun_hypernyms.tsv
```

## Run

```shell
./poincare_embedding ./wordnet_noun_hypernyms.tsv ./result_embedding.tsv -t 1 -e 5 -d 2 -l 0.001
```
