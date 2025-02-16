import torch
from utils.metrics import (
    calculate_accuracy,
    calculate_precision_recall_f1,
    supervised_contrastive_loss,
)


def test_pre(model, test_loader, lambda_val, data_aug, tau, loss_recon=None):
    model.eval()
    test_loss = 0

    with torch.no_grad():
        for data in test_loader:

            if data_aug == "featureMasking_edgeDropping":
                _, graph_emb_1, graph_emb_2 = model(data)
                t_loss = supervised_contrastive_loss(
                    graph_emb_1, graph_emb_2, data.y, tau=tau
                )

            elif data_aug == "featureMasking_edgeDropping_Decoder":
                reconstruction, _, graph_emb_1, graph_emb_2 = model(data)
                t_reconstruction_loss = loss_recon(reconstruction, data.fc.to("cuda"))
                t_loss = supervised_contrastive_loss(
                    graph_emb_1, graph_emb_2, data.y, tau=tau
                )

            elif data_aug == "NoDA_Decoder":
                reconstruction, _, graph_emb_1, graph_emb_2 = model(data)
                t_reconstruction_loss = loss_recon(reconstruction, data.fc.to("cuda"))
                t_loss = supervised_contrastive_loss(
                    graph_emb_1, graph_emb_2, data.y, tau=tau
                )

            elif data_aug == "NoDA":
                _, graph_emb_1, graph_emb_2 = model(data)
                t_loss = supervised_contrastive_loss(
                    graph_emb_1, graph_emb_2, data.y, tau=tau
                )

            if loss_recon is not None:
                t_loss = t_loss + lambda_val * t_reconstruction_loss

            test_loss += t_loss.item()

    # Calculating averages
    test_loss_avg = test_loss / len(test_loader)

    # Compiling results into a dictionary
    results = {
        "test_loss": test_loss_avg,
    }

    print(f"Test Loss: {test_loss_avg:.2f}")
    return results


def test_ft(encoder, classifier, test_loader, criterion_classif):

    encoder.eval()
    classifier.eval()
    test_loss = 0
    test_accuracy = 0
    test_precision = 0
    test_recall = 0
    test_f1 = 0

    with torch.no_grad():
        for data in test_loader:
            # Get node embeddings from the encoder
            x, edge_index_sc, edge_weight_sc, batch = (
                data.x,
                data.edge_index_sc,
                data.edge_weight_sc,
                data.batch,
            )

            node_emb = encoder.encoder(x, edge_index_sc, edge_weight_sc)
            graph_emb = encoder.graph_embedding(node_emb, batch)

            # Get predictions from the classifier
            predictions = torch.squeeze(classifier(graph_emb)).float()

            # Calculate loss
            t_loss = criterion_classif(predictions, data.y.long())
            test_loss += t_loss.item()

            # Calculate metrics
            acc = calculate_accuracy(predictions, data.y)
            test_accuracy += acc

            precision, recall, f1_score = calculate_precision_recall_f1(
                predictions, data.y
            )
            test_precision += precision
            test_recall += recall
            test_f1 += f1_score

    # Calculate averages
    test_loss_avg = test_loss / len(test_loader)
    test_accuracy_avg = test_accuracy / len(test_loader)
    test_precision_avg = test_precision / len(test_loader)
    test_recall_avg = test_recall / len(test_loader)
    test_f1_avg = test_f1 / len(test_loader)

    # Compile results into a dictionary
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
