from __future__ import print_function, division, absolute_import, unicode_literals

import numpy as np
import pandas as pd
import click

@click.command()
@click.argument("embedding_file")
def main(embedding_file):

    # load embeddings
    print("read embedding_file:", embedding_file)
    embeddings = pd.read_csv(embedding_file, header=None, sep="\t", index_col=0)
    keys, vals = [], []
    for key, val in embeddings.iterrows():
        keys.append(key)
        vals.append(np.array(val))
    vals = np.vstack(vals)

    norms = np.sqrt((vals * vals).sum(1))

    indices = np.argsort(norms)

    for i in range(20):
        print(i+1, keys[indices[i]], norms[indices[i]], sep='\t')
    print('...')
    for i in range(20):
        print(len(indices)-20+i, keys[indices[-20+i]], norms[indices[-20+i]], sep='\t')

if __name__ == '__main__':
    main()
