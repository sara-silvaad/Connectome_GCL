import torch.optim as optim
from config import LR_ft


def set_optimizer(
    encoder, classifier, train_encoder=True, train_classifier=True, lr=LR_ft
):
    if not train_encoder:
        for param in encoder.parameters():
            param.requires_grad = False
    else:
        for param in encoder.parameters():
            param.requires_grad = True

    if not train_classifier:
        for param in classifier.parameters():
            param.requires_grad = False
    else:
        for param in classifier.parameters():
            param.requires_grad = True

    parameters = []
    if train_encoder:
        parameters += list(encoder.parameters())
    if train_classifier:
        parameters += list(classifier.parameters())

    optimizer = optim.Adam(parameters, lr=LR_ft)
    return optimizer
