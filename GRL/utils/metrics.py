import torch

# def calculate_accuracy(logits, labels):
#     probs = torch.softmax(logits, dim=1)
#     preds = torch.argmax(probs, dim=1)
#     labels = torch.argmax(labels, dim=1)
#     correct = (preds == labels).float().sum()
#     accuracy = correct / labels.shape[0]
#     return accuracy.item()

# def calculate_precision_recall_f1(logits, labels):
#     # Convertir logits a probabilidades y luego a predicciones binarias
#     probs = torch.softmax(logits, dim=1)
#     preds = torch.argmax(probs, dim=1)
#     labels = torch.argmax(labels, dim=1)

#     # Verdaderos positivos (TP), falsos positivos (FP), y falsos negativos (FN)
#     TP = ((preds == 1) & (labels == 1)).float().sum()
#     FP = ((preds == 1) & (labels == 0)).float().sum()
#     FN = ((preds == 0) & (labels == 1)).float().sum()

#     # Precisión, recall y F1-score
#     precision = TP / (TP + FP) if TP + FP > 0 else torch.tensor(0.0)
#     recall = TP / (TP + FN) if TP + FN > 0 else torch.tensor(0.0)
#     f1_score = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else torch.tensor(0.0)

#     return precision.item(), recall.item(), f1_score.item()

# def calculate_accuracy(logits, labels):
#     probs = torch.sigmoid(logits)  # Calcula la probabilidad
#     preds = (probs >= 0.5).float()  # Convierte las probabilidades a predicciones binarias
#     correct = (preds == labels).float().sum()  # Cuenta las predicciones correctas
#     accuracy = correct / labels.shape[0]  # Calcula la precisión
#     return accuracy.item()

# def calculate_precision_recall_f1(logits, labels):
#     probs = torch.sigmoid(logits)  # Convierte logits a probabilidades
#     preds = (probs >= 0.5).float()  # Convierte probabilidades a predicciones binarias

#     # Verdaderos positivos (TP), falsos positivos (FP), y falsos negativos (FN)
#     TP = ((preds == 1) & (labels == 1)).float().sum()
#     FP = ((preds == 1) & (labels == 0)).float().sum()
#     FN = ((preds == 0) & (labels == 1)).float().sum()

#     # Precisión, recall y F1-score
#     precision = TP / (TP + FP) if TP + FP > 0 else torch.tensor(0.0)
#     recall = TP / (TP + FN) if TP + FN > 0 else torch.tensor(0.0)
#     f1_score = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else torch.tensor(0.0)

#     return precision.item(), recall.item(), f1_score.item()


def calculate_accuracy(logits, labels):
    probs = torch.softmax(logits, dim=1)
    preds = torch.argmax(probs, dim=1)
    # labels = torch.argmax(labels, dim=1)
    correct = (preds == labels).float().sum()
    accuracy = correct / labels.shape[0]
    return accuracy.item()


def calculate_precision_recall_f1(logits, labels):
    # Convertir logits a probabilidades y luego a predicciones binarias
    probs = torch.softmax(logits, dim=1)
    preds = torch.argmax(probs, dim=1)
    # labels = torch.argmax(labels, dim=1)

    # Verdaderos positivos (TP), falsos positivos (FP), y falsos negativos (FN)
    TP = ((preds == 1) & (labels == 1)).float().sum()
    FP = ((preds == 1) & (labels == 0)).float().sum()
    FN = ((preds == 0) & (labels == 1)).float().sum()

    # Precisión, recall y F1-score
    precision = TP / (TP + FP) if TP + FP > 0 else torch.tensor(0.0)
    recall = TP / (TP + FN) if TP + FN > 0 else torch.tensor(0.0)
    f1_score = (
        2 * (precision * recall) / (precision + recall)
        if precision + recall > 0
        else torch.tensor(0.0)
    )

    return precision.item(), recall.item(), f1_score.item()
