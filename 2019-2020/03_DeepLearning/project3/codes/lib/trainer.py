import torch
import torch.nn as nn

import os
import sys
import time
from tensorboardX import SummaryWriter

from lib.test import get_accuracy
from lib.utils import plot_intermediate_results, random_point_dropout


def train(model, trainloader, optimer, criterion, testloader, args, scheduler=None):

    cur_time = time.strftime("%m-%dT%H-%M", time.localtime())
    log_path = os.path.join(args.save_path, "log")
    if not os.path.exists(log_path):
        os.mkdir(log_path)
    writer = SummaryWriter(log_dir=os.path.join(log_path, cur_time))

    output_buffer = {"train_batch/loss": [], "train/loss": [], "test/accuracy": []}

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.cuda() if torch.cuda.is_available() else model

    cur_batch=0

    best_acc, best_epoch = 0, 0
    for cur_epoch in range(args.max_epoch):
        model.train()
        
        loss = 0
        time1 = time.time()
        for data_batch in trainloader:
            if args.data_arg: 
                data_batch["points"] = random_point_dropout(data_batch["points"], max_dropout_ratio=0.95)
            points, labels = data_batch["points"].to(device), data_batch["label"].to(device)

            preds = model(points, features=None) # (B, num_classes)

            optimer.zero_grad()
            batch_loss = criterion(preds, labels.long().squeeze())
            batch_loss.backward()
            optimer.step()

            loss += batch_loss
            writer.add_scalar("train_batch/loss", batch_loss, cur_batch)
            output_buffer["train_batch/loss"].append(batch_loss.detach().cpu().item())
            cur_batch += 1
            # print(batch_loss)

        if scheduler is not None:
            scheduler.step()
        
        writer.add_scalar("train/loss", loss, cur_epoch)
        output_buffer["train/loss"].append(loss.detach().cpu().item())
        print("Epoch [{}/{}] Train loss: [{:.5f}]".format(cur_epoch, args.max_epoch, loss), end=" ")

        acc = get_accuracy(model, testloader, device)
        writer.add_scalar("test/accuracy", acc, cur_epoch)
        output_buffer["test/accuracy"].append(acc)

        lr = scheduler.get_lr() if scheduler is not None else args.lr
        print( "Test: ACC: [{:.5f}], Time: [{:.3f}]s, LR: [{}]".format(acc, time.time()-time1, lr), end="\r")

        if acc > best_acc:
            best_acc, best_epoch = acc, cur_epoch
            save_model(model, path=os.path.join(args.save_path, args.classifier, "BEST_"+ cur_time + ".pt"))

    model.eval() 
    writer.close()

    print("\n")
    print("Best Outcome: [Epoch: {0}],  ACC: [{1:.5f}]".format(best_epoch, best_acc))

    save_model(model, path=os.path.join(args.save_path, args.classifier, "LAST_" + cur_time + ".pt"), pt=True)
    
    plot_intermediate_results(
        [output_buffer["train_batch/loss"]],
        [output_buffer["train/loss"]],
        [output_buffer["test/accuracy"]],
        labels="PointNet2",
        args=args,
        plus=cur_time)

    save_intermediate_results(output_buffer, path=os.path.join(args.save_path, args.classifier, 
                                                            "intermediate_results_{}_{}_{}.pkl".format(args.loss, args.optim, cur_time)), pt=True)

    return model
            

def save_model(model, path, pt=False):
    from pathlib import Path

    file_path = Path(path)
    file_path.touch(exist_ok=True)

    torch.save(model.state_dict(), file_path)
    if pt: print("Saving model into {}".format(file_path))


def save_intermediate_results(output_buffer, path, pt=False):
    from pathlib import Path
    import pickle

    file_path = Path(path)
    file_path.touch(exist_ok=True)

    with open(file_path, "wb") as fp:
        pickle.dump(output_buffer, fp)
    
    if pt: print("Saving intermediate results into {}".format(file_path))
