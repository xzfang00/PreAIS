from feature_extract.K_mer import kmer_feature
import numpy as np
import pandas as pd


def extract_sequences(file_path):
    # .txt
    sequences = []
    with open(file_path, 'r') as file:
        for line in file:
            if not line.startswith(">"):
                sequences.append(line.strip())
    return sequences


def merge_sequences_and_generate_labels(pos_seq, neg_seq):
    all_sequences = pos_seq + neg_seq
    pos_labels = [1] * len(pos_seq)
    neg_labels = [0] * len(neg_seq)
    all_labels = pos_labels + neg_labels
    return all_sequences, all_labels


def min_max_normalization(data):
    min_vals = data.min(axis=0)
    max_vals = data.max(axis=0)
    return (data - min_vals) / (max_vals - min_vals)


def process_data(file_neg_path, file_pos_path):

    neg_seq = extract_sequences(file_neg_path)
    pos_seq = extract_sequences(file_pos_path)

    seq, label = merge_sequences_and_generate_labels(pos_seq, neg_seq)

    train_data = kmer_feature(seq, k=3)

    return train_data, np.array(label, dtype=int)


def extract_xlsx_sequences(file_path):

    df = pd.read_excel(file_path)

    sequences = df['RNA sequence (101nt)'].tolist()
    labels = df['lable'].tolist()
    test_data = kmer_feature(sequences, k=3)

    return test_data, np.array(labels, dtype=int)
