import numpy as np
import torch


def convert_to_edge_index(M):
    M = M.triu(diagonal=1)
    edge_index = M.nonzero(as_tuple=False).t().contiguous()
    edge_weights = M[edge_index[0], edge_index[1]].view(-1, 1)
    return edge_index, edge_weights


def get_sc_reduction_graph(train_dataset, val_dataset, test_dataset, d):
    def reduce_rank_sc(sc, d):
        eps = 0.001
        D = torch.diag(sc.sum(1))
        L = D - sc
        eigv, V = np.linalg.eig(L)
        Ld = V[:, :d] @ np.diag(eigv)[:d, :d] @ V.T[:d, :]
        Ad = D - Ld
        Ad[Ad < eps] = 0
        return Ad

    for subject in train_dataset:
        sc_r = reduce_rank_sc(subject.sc, d)
        subject.sc = sc_r
        subject.edge_index_sc, subject.edge_weight_sc = convert_to_edge_index(sc_r)

    for subject in val_dataset:
        sc_r = reduce_rank_sc(subject.sc, d)
        subject.sc = sc_r
        subject.edge_index_sc, subject.edge_weight_sc = convert_to_edge_index(sc_r)

    for subject in test_dataset:
        sc_r = reduce_rank_sc(subject.sc, d)
        subject.sc = sc_r
        subject.edge_index_sc, subject.edge_weight_sc = convert_to_edge_index(sc_r)

    return train_dataset, val_dataset, test_dataset


def get_average_sc_graph(train_dataset, val_dataset, test_dataset):
    n = len(train_dataset)
    sc_avg = np.zeros((87, 87), dtype=np.float32)
    for i in range(n):
        sc_avg = train_dataset[i].sc + sc_avg

    sc_avg = sc_avg / n
    edge_index_avg, edge_weight_avg = convert_to_edge_index(sc_avg)

    for subject in train_dataset:
        subject.sc = sc_avg
        subject.edge_index_sc = edge_index_avg
        subject.edge_weight_sc = edge_weight_avg

    for subject in val_dataset:
        subject.sc = sc_avg
        subject.edge_index_sc = edge_index_avg
        subject.edge_weight_sc = edge_weight_avg

    for subject in test_dataset:
        subject.sc = sc_avg
        subject.edge_index_sc = edge_index_avg
        subject.edge_weight_sc = edge_weight_avg

    return train_dataset, val_dataset, test_dataset
