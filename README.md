# poincare-embedding

This codes implement Poincar\'e Embedding introduced in the following paper:

[Maximilian Nickel and Douwe Kiela, "Poincar\'e Embeddings for Learning Hierarchical Representations'", arXiv preprint arXiv:1705.08039, 2017.](https://arxiv.org/abs/1705.08039)

Remarks: Latest version has some bugs and we haven't reproduce appearing result yet.
    
## Build

```shell
cd poincare-embedding
mkdir work & cd work
cmake ..
make
```

## Tutorial

We assume that you are in work directory


```shell
cd poincare-embedding
mkdir work & cd work
```

### Data Creation

You can create WordNet noun hypernym pairs as follows

```shell
python ../scripts/create_wordnet_noun_hierarchy.py ./wordnet_noun_hypernyms.tsv
```

### Run

```shell
./poincare_embedding ./wordnet_noun_hypernyms.tsv ./embeddings.tsv -d 2 -t 8 -e 100 -l 0.1
```
### Plot Tree

```shell
python ../scripts/plot_tree.py mammal.n.01 ./wordnet_noun_hypernyms.tsv ./embeddings.tsv
```
