import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

def test_model(model, testloader):
    model.eval()

    pred_labels = np.array([])
    true_labels = np.array([])

    for data_batch in testloader:
        images, batch_labels = data_batch
        if torch.cuda.is_available():
            images = images.cuda()

        logits = F.softmax(model(images), dim=1)
        logits = logits.detach().cpu().numpy()
        batch_labels = batch_labels.detach().cpu().numpy()

        logits_labels = np.argmax(logits, axis=1)
        pred_labels = np.concatenate((pred_labels, logits_labels))
        true_labels = np.concatenate((true_labels, batch_labels))
    

    acc = accuracy_score(true_labels, pred_labels)
    f1 = f1_score(true_labels, pred_labels, average='macro')

    return f1, acc
