import torch
from utils.metrics import calculate_accuracy, calculate_precision_recall_f1


def test_model_encoder_decoder_SCFC(
    model, path_to_checkpoint, test_loader, lamb, criterion_recon, criterion_classif
):

    checkpoint = torch.load(path_to_checkpoint)
    model.load_state_dict(checkpoint["model_state_dict"])

    model.eval()
    test_loss = 0
    test_accuracy = 0
    test_precision = 0
    test_recall = 0
    test_f1 = 0
    test_mse = 0

    with torch.no_grad():
        for data in test_loader:
            decoded, _, classif_logits, graph_emb, _ = model(data)

            (
                decoded,
                classif_logits,
                graph_emb,
            ) = (
                decoded.to("cpu"),
                classif_logits.to("cpu"),
                graph_emb.to("cpu"),
            )

            t_loss_recon = criterion_recon(decoded, data.fc.view(-1, data.fc.shape[-1]))
            t_loss_classif = criterion_classif(classif_logits, data.y.long())
            t_loss = t_loss_recon + lamb * t_loss_classif
            test_loss += t_loss.item()

            test_accuracy += calculate_accuracy(classif_logits, data.y)
            macro_precision, macro_recall, macro_f1_score = (
                calculate_precision_recall_f1(classif_logits, data.y)
            )
            test_precision += macro_precision
            test_recall += macro_recall
            test_f1 += macro_f1_score
            test_mse += criterion_recon(
                decoded, data.fc.view(-1, data.fc.shape[-1])
            ).item()

    # Calculating averages
    test_loss_avg = test_loss / len(test_loader)
    test_accuracy_avg = test_accuracy / len(test_loader)
    test_precision_avg = test_precision / len(test_loader)
    test_recall_avg = test_recall / len(test_loader)
    test_f1_avg = test_f1 / len(test_loader)
    test_mse_avg = test_mse / len(test_loader)

    # Compiling results into a dictionary
    results = {
        "test_loss": test_loss_avg,
        "test_accuracy": test_accuracy_avg,
        "test_precision": test_precision_avg,
        "test_recall": test_recall_avg,
        "test_f1": test_f1_avg,
        "test_mse": test_mse_avg,
    }

    print(
        f"Test Loss: {test_loss_avg:.2f}, Test Acc: {test_accuracy_avg:.2f}, Test Precision: {test_precision_avg:.2f}, Test Recall: {test_recall_avg:.2f}, Test F1: {test_f1_avg:.2f}, Test MSE: {test_mse_avg:.2f}"
    )
    return results


def test_model_classifier(model, path_to_checkpoint, test_loader, criterion_classif):

    checkpoint = torch.load(path_to_checkpoint)
    model.load_state_dict(checkpoint["model_state_dict"])

    model.eval()
    test_loss = 0
    test_accuracy = 0
    test_precision = 0
    test_recall = 0
    test_f1 = 0

    with torch.no_grad():
        for data in test_loader:
            classif_logits, _, _ = model(data)

            classif_logits = classif_logits.to("cpu")

            t_loss = criterion_classif(classif_logits, data.y.long())
            test_loss += t_loss.item()

            acc = calculate_accuracy(classif_logits, data.y.long())
            test_accuracy += acc

            precision, recall, f1_score = calculate_precision_recall_f1(
                classif_logits, data.y
            )
            test_precision += precision
            test_recall += recall
            test_f1 += f1_score

    # Calculating averages
    test_loss_avg = test_loss / len(test_loader)
    test_accuracy_avg = test_accuracy / len(test_loader)
    test_precision_avg = test_precision / len(test_loader)
    test_recall_avg = test_recall / len(test_loader)
    test_f1_avg = test_f1 / len(test_loader)

    # Compiling results into a dictionary
    results = {
        "test_loss": test_loss_avg,
        "test_accuracy": test_accuracy_avg,
        "test_precision": test_precision_avg,
        "test_recall": test_recall_avg,
        "test_f1": test_f1_avg,
    }

    print(
        f"Test Loss: {test_loss_avg:.2f}, Test Acc: {test_accuracy_avg:.2f}, "
        f"Test Precision: {test_precision_avg:.2f}, Test Recall: {test_recall_avg:.2f}, "
        f"Test F1: {test_f1_avg:.2f}"
    )
    return results


def test_model(
    model,
    path_to_checkpoint,
    test_loader,
    criterion_recon,
    criterion_classif,
    lamb=None,
):
    if model.model_name in ["EncoderDecoderSCFC", "EncoderDecoderSCAlternateDA"]:
        results = test_model_encoder_decoder_SCFC(
            model,
            path_to_checkpoint,
            test_loader,
            lamb,
            criterion_recon,
            criterion_classif,
        )
        return results

    elif model.model_name in [
        "EncoderClassifierSC",
        "EncoderClassifierSCAlternateDA",
        "FullyConnected",
    ]:
        results = test_model_classifier(
            model, path_to_checkpoint, test_loader, criterion_classif
        )
        return results
