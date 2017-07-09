from __future__ import print_function, division, absolute_import, unicode_literals

import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import click

@click.command()
@click.argument("root_name")
@click.argument("data_file")
@click.argument("embedding_file")
@click.option("--max_plot", default=30)
@click.option("--left_is_parent", is_flag=True)
def main(root_name, data_file, embedding_file, max_plot, left_is_parent):

    # load relations
    print("read data_file:", data_file)
    relations = pd.read_csv(data_file, header=None, sep="\t")

    # find children
    print("find children of", root_name)
    lhds, rhds = relations[0], relations[1]

    if not left_is_parent:
        targets = set(lhd for i, lhd in enumerate(lhds) if rhds.ix[i] == root_name)
    else:
        targets = set(rhd for i, rhd in enumerate(rhds) if lhds.ix[i] == root_name)
    if len(targets) + 1 > max_plot:
        targets = random.sample(targets, max_plot-1)
    targets.append(root_name)
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
