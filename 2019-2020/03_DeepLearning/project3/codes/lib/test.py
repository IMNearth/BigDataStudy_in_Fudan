import torch
import torch.nn.functional as F

import numpy as np
from sklearn.metrics import accuracy_score

def get_accuracy(model, testloader, device):

    model.eval()

    pred_labels = np.array([])
    true_labels = np.array([])

    for data_batch in testloader:
        points, batch_labels = data_batch["points"].to(device), data_batch["label"].to(device)

        logits = F.softmax(model(points), dim=1)
        logits = logits.detach().cpu().numpy()
        batch_labels = batch_labels.squeeze().detach().cpu().numpy()

        logits_labels = np.argmax(logits, axis=1)
        pred_labels = np.concatenate((pred_labels, logits_labels))
        true_labels = np.concatenate((true_labels, batch_labels))

    acc = accuracy_score(true_labels, pred_labels)

    return acc
