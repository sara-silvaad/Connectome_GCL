import torch
from utils.metrics import calculate_accuracy, calculate_precision_recall_f1
from utils.plots import save_model
import os
from torch.utils.tensorboard import SummaryWriter


def train_and_evaluate_encoder_decoder_SCFC(
    model,
    train_loader,
    val_loader,
    lamb,
    epochs,
    optimizer,
    criterion_recon,
    criterion_classif,
    early_stopping_patience,
    path,
    filename,
    ckpt_path=None,
):

    if ckpt_path is not None:
        checkpoint = torch.load(ckpt_path)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        ckpt_epoch = checkpoint["epoch"]
    else:
        ckpt_epoch = 0

    ckpt_path = f"{path}/ckpt/"
    logs_dir = f"{path}/logs/"

    vis_rate = 50

    os.makedirs(ckpt_path, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    writer = SummaryWriter(log_dir=logs_dir)

    results = {
        "train_loss": [],
        "train_accuracy": [],
        "train_precision": [],
        "train_recall": [],
        "train_f1": [],
        "train_mse": [],
        "val_loss": [],
        "val_accuracy": [],
        "val_precision": [],
        "val_recall": [],
        "val_f1": [],
        "val_mse": [],
    }

    best_val_loss = float("inf")
    epochs_no_improve = 0
    best_model = {
        "model": model,
        "epoch": ckpt_epoch,
        "val_accuracy": 0,
        "val_loss": 0,
        "optimizer": optimizer,
    }

    for epoch in range(epochs - ckpt_epoch):  # Training and Validation loop
        epoch = epoch + ckpt_epoch

        # Training phase
        model.train()
        total_loss = 0
        train_accuracy = 0
        train_precision = 0
        train_recall = 0
        train_f1 = 0
        train_mse = 0
        # total_loss_class = 0

        graph_emb_total = []
        labels_total = []
        for data in train_loader:
            optimizer.zero_grad()

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
            loss_recon = criterion_recon(decoded, data.fc)
            loss_classif = criterion_classif(classif_logits, data.y.long())
            loss = loss_recon + lamb * loss_classif

            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            # total_loss_class += loss_classif.item()

            train_accuracy += calculate_accuracy(classif_logits, data.y)
            macro_precision, macro_recall, macro_f1_score = (
                calculate_precision_recall_f1(classif_logits, data.y)
            )
            train_precision += macro_precision
            train_recall += macro_recall
            train_f1 += macro_f1_score
            train_mse += criterion_recon(decoded, data.fc).item()

            graph_emb_total.append(graph_emb)
            labels_total.append(data.y)

        train_loss_avg = total_loss / len(train_loader)
        train_accuracy_avg = train_accuracy / len(train_loader)
        train_f1_avg = train_f1 / len(train_loader)
        train_precision_avg = train_precision / len(train_loader)
        train_recall_avg = train_recall / len(train_loader)
        train_mse_avg = train_mse / len(train_loader)

        writer.add_scalar("Loss/Train", train_loss_avg, epoch)
        writer.add_scalar("Accuracy/Train", train_accuracy_avg, epoch)
        writer.add_scalar("F1/Train", train_f1_avg, epoch)
        writer.add_scalar("Precision/Train", train_precision_avg, epoch)
        writer.add_scalar("Recall/Train", train_recall_avg, epoch)

        # Validation phase
        model.eval()
        val_loss = 0
        val_accuracy = 0
        val_precision = 0
        val_recall = 0
        val_f1 = 0
        val_mse = 0

        graph_emb_total = []
        labels_total = []
        with torch.no_grad():
            for data in val_loader:
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
                v_loss_recon = criterion_recon(decoded, data.fc)
                v_loss_classif = criterion_classif(classif_logits, data.y.long())
                v_loss = v_loss_recon + lamb * v_loss_classif
                val_loss += v_loss.item()

                val_accuracy += calculate_accuracy(classif_logits, data.y)
                macro_precision, macro_recall, macro_f1_score = (
                    calculate_precision_recall_f1(classif_logits, data.y)
                )
                val_precision += macro_precision
                val_recall += macro_recall
                val_f1 += macro_f1_score
                val_mse += criterion_recon(
                    decoded, data.fc.view(-1, data.fc.shape[-1])
                ).item()

                graph_emb_total.append(graph_emb)
                labels_total.append(data.y)

        val_loss_avg = val_loss / len(val_loader)
        val_accuracy_avg = val_accuracy / len(val_loader)
        val_f1_avg = val_f1 / len(val_loader)
        val_precision_avg = val_precision / len(val_loader)
        val_recall_avg = val_recall / len(val_loader)
        val_mse_avg = val_mse / len(val_loader)

        writer.add_scalar("Loss/Val", val_loss_avg, epoch)
        writer.add_scalar("Accuracy/Val", val_accuracy_avg, epoch)
        writer.add_scalar("F1/Val", val_f1_avg, epoch)
        writer.add_scalar("Precision/Val", val_precision_avg, epoch)
        writer.add_scalar("Recall/Val", val_recall_avg, epoch)

        # Append epoch results to the results dict
        results["train_loss"].append(train_loss_avg)
        results["train_accuracy"].append(train_accuracy_avg)
        results["train_precision"].append(train_precision_avg)
        results["train_recall"].append(train_recall_avg)
        results["train_f1"].append(train_f1_avg)
        results["train_mse"].append(train_mse_avg)
        results["val_loss"].append(val_loss_avg)
        results["val_accuracy"].append(val_accuracy_avg)
        results["val_precision"].append(val_precision_avg)
        results["val_recall"].append(val_recall_avg)
        results["val_f1"].append(val_f1_avg)
        results["val_mse"].append(val_mse_avg)

        print(
            f"Epoch {epoch+1}, Train Loss: {train_loss_avg:.2f}, Train Acc: {train_accuracy_avg:.2f}, Train Precision: {train_precision_avg:.2f}, Train Recall: {train_recall_avg:.2f},Train F1: {train_f1_avg:.2f}, Train MSE: {train_mse_avg:.2f}, Val Loss: {val_loss_avg:.2f}, Val Acc: {val_accuracy_avg:.2f}, Val Precision: {val_precision_avg:.2f}, Val Recall: {val_recall_avg:.2f}, Val F1: {val_f1_avg:.2f}, Val MSE: {val_mse_avg:.2f}"
        )

        # Check for early stopping
        if val_loss_avg < best_val_loss:
            best_val_loss = val_loss_avg
            epochs_no_improve = 0
            best_model["model"] = model
            best_model["epoch"] = epoch
            best_model["val_accuracy"] = val_accuracy_avg
            best_model["val_loss"] = val_loss_avg
            best_model["optimizer"] = optimizer
            save_model(ckpt_path, f"best_{filename}", best_model)
        else:
            epochs_no_improve += 1

        save_model(ckpt_path, f"last_{filename}", best_model)

        if epochs_no_improve == early_stopping_patience:
            print(f"Early stopping triggered after {epoch+1} epochs.")
            break

    writer.flush()
    return results


def train_and_evaluate_classifier(
    model,
    train_loader,
    val_loader,
    epochs,
    optimizer,
    criterion_classif,
    early_stopping_patience,
    path,
    filename,
    ckpt_path,
):

    if ckpt_path is not None:
        checkpoint = torch.load(ckpt_path)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        ckpt_epoch = checkpoint["epoch"]
    else:
        ckpt_epoch = 0

    ckpt_path = f"{path}/ckpt/"
    logs_dir = f"{path}/logs/"

    vis_rate = 200

    os.makedirs(ckpt_path, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    writer = SummaryWriter(log_dir=logs_dir)

    results = {
        "train_loss": [],
        "train_accuracy": [],
        "train_precision": [],
        "train_recall": [],
        "train_f1": [],
        "val_loss": [],
        "val_accuracy": [],
        "val_precision": [],
        "val_recall": [],
        "val_f1": [],
    }

    best_val_loss = float("inf")
    epochs_no_improve = 0
    best_model = {
        "model": model,
        "epoch": ckpt_epoch,
        "val_accuracy": 0,
        "val_loss": 0,
        "optimizer": optimizer,
    }

    for epoch in range(epochs - ckpt_epoch):  # Training and Validation loop
        epoch = epoch + ckpt_epoch

        # Training loop
        model.train()
        total_loss = 0
        train_accuracy = 0
        train_precision = 0
        train_recall = 0
        train_f1 = 0

        graph_emb_total = []
        labels_total = []

        for data in train_loader:
            optimizer.zero_grad()

            classif_logits, _, graph_emb = model(data)

            loss = criterion_classif(classif_logits.to("cpu"), data.y.long())

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            train_accuracy += calculate_accuracy(
                classif_logits.to("cpu"), data.y.long()
            )
            macro_precision, macro_recall, macro_f1_score = (
                calculate_precision_recall_f1(classif_logits.to("cpu"), data.y)
            )
            train_precision += macro_precision
            train_recall += macro_recall
            train_f1 += macro_f1_score

            graph_emb_total.append(graph_emb)
            labels_total.append(data.y)

            torch.cuda.empty_cache()

        train_loss_avg = total_loss / len(train_loader)
        train_accuracy_avg = train_accuracy / len(train_loader)
        train_f1_avg = train_f1 / len(train_loader)
        train_precision_avg = train_precision / len(train_loader)
        train_recall_avg = train_recall / len(train_loader)

        # Tensorboard Logging
        if model.model_name != "FullyConnected":
            if epoch % vis_rate == 0:
                graph_emb_total = torch.cat(graph_emb_total, dim=0)
                labels_total = torch.cat(labels_total, dim=0)
                writer.add_embedding(
                    graph_emb_total,
                    metadata=labels_total,
                    label_img=None,
                    global_step=epoch,
                    tag="Traning graph embedding",
                    metadata_header=None,
                )

        writer.add_scalar("Loss/Train", train_loss_avg, epoch)
        writer.add_scalar("Accuracy/Train", train_accuracy_avg, epoch)
        writer.add_scalar("F1/Train", train_f1_avg, epoch)
        writer.add_scalar("Precision/Train", train_precision_avg, epoch)
        writer.add_scalar("Recall/Train", train_recall_avg, epoch)

        # Validation phase
        model.eval()
        val_loss = 0
        val_accuracy = 0
        val_precision = 0
        val_recall = 0
        val_f1 = 0
        graph_emb_total = []
        labels_total = []
        with torch.no_grad():
            for data in val_loader:
                classif_logits, _, graph_emb = model(data)

                v_loss = criterion_classif(classif_logits.to("cpu"), data.y.long())
                val_loss += v_loss.item()
                val_accuracy += calculate_accuracy(
                    classif_logits.to("cpu"), data.y.long()
                )
                macro_precision, macro_recall, macro_f1_score = (
                    calculate_precision_recall_f1(classif_logits.to("cpu"), data.y)
                )
                val_precision += macro_precision
                val_recall += macro_recall
                val_f1 += macro_f1_score
                torch.cuda.empty_cache()

                labels_total.append(data.y)
                graph_emb_total.append(graph_emb)

        val_loss_avg = val_loss / len(val_loader)
        val_accuracy_avg = val_accuracy / len(val_loader)
        val_f1_avg = val_f1 / len(val_loader)
        val_precision_avg = val_precision / len(val_loader)
        val_recall_avg = val_recall / len(val_loader)

        # Tensorboard Logging
        if model.model_name != "FullyConnected":
            if epoch % vis_rate == 0:
                graph_emb_total = torch.cat(graph_emb_total, dim=0)
                labels_total = torch.cat(labels_total, dim=0)
                writer.add_embedding(
                    graph_emb_total,
                    metadata=labels_total,
                    label_img=None,
                    global_step=epoch,
                    tag="Validation graph embedding",
                    metadata_header=None,
                )

        writer.add_scalar("Loss/Val", val_loss_avg, epoch)
        writer.add_scalar("Accuracy/Val", val_accuracy_avg, epoch)
        writer.add_scalar("F1/Val", val_f1_avg, epoch)
        writer.add_scalar("Precision/Val", val_precision_avg, epoch)
        writer.add_scalar("Recall/Val", val_recall_avg, epoch)

        # Append epoch results to the results dict
        results["train_loss"].append(train_loss_avg)
        results["train_accuracy"].append(train_accuracy_avg)
        results["train_precision"].append(train_precision_avg)
        results["train_recall"].append(train_recall_avg)
        results["train_f1"].append(train_f1_avg)
        results["val_loss"].append(val_loss_avg)
        results["val_accuracy"].append(val_accuracy_avg)
        results["val_precision"].append(val_precision_avg)
        results["val_recall"].append(val_recall_avg)
        results["val_f1"].append(val_f1_avg)

        print(
            f"Epoch {epoch+1}, Train Loss: {train_loss_avg:.2f}, Train Acc: {train_accuracy_avg:.2f}, Train Precision: {train_precision_avg:.2f}, Train Recall: {train_recall_avg:.2f}, Train F1: {train_f1_avg:.2f}, Val Loss: {val_loss_avg:.2f}, Val Acc: {val_accuracy_avg:.2f}, Val Precision: {val_precision_avg:.2f}, Val Recall: {val_recall_avg:.2f}, Val F1: {val_f1_avg:.2f}"
        )

        # Early Stopping Check
        if val_loss_avg < best_val_loss:
            best_val_loss = val_loss_avg
            epochs_no_improve = 0
            best_model["model"] = model
            best_model["epoch"] = epoch
            best_model["val_accuracy"] = val_accuracy_avg
            best_model["val_loss"] = val_loss_avg
            best_model["optimizer"] = optimizer
            save_model(ckpt_path, f"best_{filename}", best_model)
        else:
            epochs_no_improve += 1

        save_model(ckpt_path, f"last_{filename}", best_model)

        if epochs_no_improve == early_stopping_patience:
            print(f"Early stopping triggered after {epoch+1} epochs.")
            break

    writer.flush()
    return results


def train_and_evaluate(
    model,
    train_loader,
    val_loader,
    epochs,
    optimizer,
    criterion_recon,
    criterion_classif,
    early_stopping_patience,
    path,
    filename,
    ckpt_path=None,
    lamb=None,
):

    if model.model_name in ["EncoderDecoderSCFC", "EncoderDecoderSCAlternateDA"]:
        results = train_and_evaluate_encoder_decoder_SCFC(
            model,
            train_loader,
            val_loader,
            lamb,
            epochs,
            optimizer,
            criterion_recon,
            criterion_classif,
            early_stopping_patience,
            path,
            filename,
            ckpt_path,
        )
        return results

    if model.model_name in [
        "EncoderClassifierSC",
        "EncoderClassifierSCAlternateDA",
        "FullyConnected",
    ]:
        results = train_and_evaluate_classifier(
            model,
            train_loader,
            val_loader,
            epochs,
            optimizer,
            criterion_classif,
            early_stopping_patience,
            path,
            filename,
            ckpt_path,
        )
        return results
