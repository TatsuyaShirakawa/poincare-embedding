from __future__ import print_function, division, absolute_import, unicode_literals

from nltk.corpus import wordnet as wn
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import click

@click.command()
@click.argument("embedding_file")
@click.option("--max_plot", default=30)
@click.option("--left_is_parent", is_flag=True)
def main(embedding_file, max_plot, left_is_parent):

    targets = ['mammal.n.01', 'beagle.n.01', 'canine.n.02', 'german_shepherd.n.01',
               'collie.n.01', 'border_collie.n.01',
               'carnivore.n.01', 'tiger.n.02', 'tiger_cat.n.01', 'domestic_cat.n.01',
               'squirrel.n.01', 'finback.n.01', 'rodent.n.01', 'elk.n.01',
               'homo_sapiens.n.01', 'orangutan.n.01', 'bison.n.01', 'antelope.n.01',
               'even-toed_ungulate.n.01', 'ungulate.n.01', 'elephant.n.01', 'rhinoceros.n.01',
               'odd-toed_ungulate.n.01', 'mustang.n.01']

    targets = list(set([x for x in targets]))
    print(len(targets), ' targets found')

    # load embeddings
    print("read embedding_file:", embedding_file)
    embeddings = pd.read_csv(embedding_file, header=None, sep="\t", index_col=0)
#    keys = embeddings[0]
#    vals = embeddings[embeddings.columns[1:]]
#    embeddings = dict(zip(keys, vals))

    print("plot")
    fig = plt.figure(figsize=(10,10))
    ax = plt.gca()
    ax.cla() # clear things for fresh plot

    ax.set_xlim((-1.1, 1.1))
    ax.set_ylim((-1.1, 1.1))

    circle = plt.Circle((0,0), 1., color='black', fill=False)
    ax.add_artist(circle)

    for n in targets:
        x, y = embeddings.ix[n]
        ax.plot(x, y, 'o', color='y')
        ax.text(x+0.01, y+0.01, n, color='b')
    plt.show()

if __name__ == '__main__':
    main()
