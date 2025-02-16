import torch
import torch.nn.functional as F


def supervised_contrastive_loss(graph_emb_1, graph_emb_2, data_y, tau):

    Z = F.normalize(torch.cat((graph_emb_1, graph_emb_2), dim=0), p=2, dim=1)
    y = torch.cat((data_y, data_y), dim=0)

    # the 0's -> Male labels
    N = y == 0

    mask = torch.ones_like(Z[N] @ Z[N].T) - torch.eye((Z[N] @ Z[N].T).size(0)).to(
        "cuda"
    )
    Z_negatives = torch.exp(((Z[N] @ Z[N].T) / tau) * mask)

    mask = torch.ones_like(Z[N] @ Z.T) - torch.eye((Z @ Z.T).size(0))[
        : Z[N].shape[0], :
    ].to("cuda")
    m = torch.sum(torch.exp((Z[N] @ Z.T) / tau) * mask, dim=1)
    loss_n = (torch.log(Z_negatives / m) / N.sum()).sum()

    # the 1's -> Female labels
    P = y == 1
    mask = torch.ones_like(Z[P] @ Z[P].T) - torch.eye((Z[P] @ Z[P].T).size(0)).to(
        "cuda"
    )
    Z_positives = torch.exp(((Z[P] @ Z[P].T) / tau) * mask)
    mask = torch.ones_like(Z[P] @ Z.T) - torch.eye((Z @ Z.T).size(0))[
        : Z[P].shape[0], :
    ].to("cuda")

    m = torch.sum(torch.exp((Z[P] @ Z.T) / tau) * mask, dim=1)

    loss_p = (torch.log(Z_positives / m) / P.sum()).sum()

    return -(loss_n + loss_p)


def calculate_accuracy(logits, labels):
    probs = torch.softmax(logits, dim=1)
    preds = torch.argmax(probs, dim=1)
    correct = (preds == labels).float().sum()
    accuracy = correct / labels.shape[0]
    return accuracy.item()


def calculate_precision_recall_f1(logits, labels):
    # Convert logits to probabilities and then to binary predictions
    probs = torch.softmax(logits, dim=1)
    preds = torch.argmax(probs, dim=1)

    # True positives (TP), false positives (FP), and false negatives (FN)
    TP = ((preds == 1) & (labels == 1)).float().sum()
    FP = ((preds == 1) & (labels == 0)).float().sum()
    FN = ((preds == 0) & (labels == 1)).float().sum()

    # Precision, recall, and F1-score
    precision = TP / (TP + FP) if TP + FP > 0 else torch.tensor(0.0)
    recall = TP / (TP + FN) if TP + FN > 0 else torch.tensor(0.0)
    f1_score = (
        2 * (precision * recall) / (precision + recall)
        if precision + recall > 0
        else torch.tensor(0.0)
    )

    return precision.item(), recall.item(), f1_score.item()


def similarity(x, y):
    return F.cosine_similarity(x.unsqueeze(1), y.unsqueeze(0), dim=2)
