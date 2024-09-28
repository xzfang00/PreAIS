from sklearn.metrics import matthews_corrcoef


def calculate_acc_sn_sp(val_labels, val_preds):

    TP = TN = FP = FN = 0
    for pred, label in zip(val_preds, val_labels):
        if pred == 1 and label == 1:
            TP += 1
        elif pred == 0 and label == 0:
            TN += 1
        elif pred == 1 and label == 0:
            FP += 1
        else:
            FN += 1


    total = TP + TN + FP + FN


    acc = (TP + TN) / total if total > 0 else 0
    sn = TP / (TP + FN) if (TP + FN) > 0 else 0
    sp = TN / (TN + FP) if (TN + FP) > 0 else 0

    return acc, sn, sp


def calculate_mcc(val_labels, val_preds):

    return matthews_corrcoef(val_labels, val_preds)


if __name__ == "__main__":

    val_labels = [0, 1, 1, 0, 1, 0, 1, 0]
    val_preds = [0, 1, 0, 0, 1, 1, 1, 0]

    acc, sn, sp = calculate_acc_sn_sp(val_labels, val_preds)
    print(f"Accuracy: {acc:.4f}, Sensitivity (Sn): {sn:.4f}, Specificity (Sp): {sp:.4f}")

    mcc = calculate_mcc(val_labels, val_preds)
    print(f"MCC: {mcc:.4f}")
