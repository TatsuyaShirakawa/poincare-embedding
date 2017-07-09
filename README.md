# poincare-embedding

This codes implement Poincar\'e Embedding introduced in the following paper:

[Maximilian Nickel and Douwe Kiela, "Poincar\'e Embeddings for Learning Hierarchical Representations'", arXiv preprint arXiv:1705.08039, 2017.](https://arxiv.org/abs/1705.08039)

Remarks: Latest version haven't reproduce appealing results yet.
    
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

and mammal subtree is created by

```shell
python ../scripts/create_mammal_subtree.py ./mammal_subtree.tsv
```

### Run

```shell
./poincare_embedding ./mammal_subtree.tsv ./embeddings.tsv -d 2 -t 8 -e 1000 -l 0.01
```
### Plot Tree

```shell
python ../scripts/plot_tree.py mammal.n.01 ./mammal_subtree.tsv ./embeddings.tsv
```
