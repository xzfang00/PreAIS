import numpy as np
from itertools import product
from collections import Counter


def kmer_feature(rna_sequences, k):

    kmers_set = [''.join(p) for p in product('ACGU', repeat=k)]
    def kmer(sequence, k):

        kmers = [sequence[i:i + k] for i in range(len(sequence) - k + 1)]
        return Counter(kmers)
    feature_list = []

    for sequence in rna_sequences:
        kmers = kmer(sequence, k)
        feature_vector = np.array([kmers.get(kmer, 0) for kmer in kmers_set])
        feature_list.append(feature_vector)
    return np.array(feature_list)

