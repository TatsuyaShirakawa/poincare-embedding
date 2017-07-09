from __future__ import print_function, division, unicode_literals, absolute_import

import random
from nltk.corpus import wordnet as wn
import click

def transitive_closure(synsets):

    hypernyms = set([])
    for s in synsets:
        paths = s.hypernym_paths()
        for path in paths:
            hypernyms.update((s,h) for h in path[1:] if h.pos() == 'n')
    return hypernyms

@click.command()
@click.argument('result_file')
@click.option('--shuffle', is_flag=True)
@click.option('--sep', default='\t')
@click.option('--target', default='mammal.n.01')
def main(result_file, shuffle, sep, target):

    target = wn.synset(target)
    print('target:', target)

    words = wn.words()

    nouns = set([])
    for word in words:
        nouns.update(wn.synsets(word, pos='n'))

    print( len(nouns), 'nouns')

    hypernyms = []
    for noun in nouns:
        paths = noun.hypernym_paths()
        for path in paths:
            try:
                pos = path.index(target)
                for i in range(pos, len(path)-1):
                    hypernyms.append((noun, path[i]))
            except Exception:
                continue

    hypernyms = list(set(hypernyms))
    print( len(hypernyms), 'hypernyms' )

    if not shuffle:
        random.shuffle(hypernyms)
    with open(result_file, 'w') as fout:
        for n1, n2 in hypernyms:
            print(n1.name(), n2.name(), sep=sep, file=fout)

if __name__ == '__main__':

    main()

