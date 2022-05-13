import torch
import torch.nn.functional as F
import numpy as np

import pickle
import time
from tensorboardX import SummaryWriter

from tqdm import tqdm
from colorama import Fore
import os

from pkg.engine.test import test_model


def train(model, trainloader, criterion, optimer, testloader, args, scheduler=None):
    max_epoch = args.max_epoch
    best_acc, best_f1, best_epoch = 0, 0, 0
    cur_time = time.strftime("%m-%dT%H-%M", time.localtime())

    paramstr = f"{args.activation}_device{args.device}_bs{args.batch_size}"

    # checkpoint directory
    ckpt_dir = os.path.join(args.save_path, args.classifier)
    if not os.path.exists(ckpt_dir): os.mkdir(ckpt_dir)
    ckpt_dir = os.path.join(ckpt_dir, cur_time + "_" +args.data_aug.upper())
    os.mkdir(ckpt_dir)
    writer = SummaryWriter(log_dir=ckpt_dir)
    print(f"Find checkpoints at {ckpt_dir}")
    
    # model to device
    DEVICE = torch.device(f"cuda:{args.device}") \
        if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(DEVICE)

    pbar = tqdm(range(max_epoch), 
                desc="Train", unit='ep',
                bar_format="{l_bar}%s{bar}%s{r_bar}" % (Fore.BLUE, Fore.RESET))
    output_buffer = {"train_batch/loss": [], "train/loss": [], "test/accuracy": [], "test/f1-score": []}
    cur_batch=0
    
    for cur_epoch in pbar:
        model.train()
        
        loss = 0
        for data_batch in trainloader:
            images, labels = data_batch
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            # added cutmix here
            r = np.random.rand(1)
            if args.data_aug == "cutmix" and args.beta > 0 and r < args.cutmix_prob:
                # generate mixed sample
                lam = np.random.beta(args.beta, args.beta)
                rand_index = torch.randperm(images.size()[0]).cuda()
                target_a = labels
                target_b = labels[rand_index]
                bbx1, bby1, bbx2, bby2 = rand_bbox(images.size(), lam)
                images[:, :, bbx1:bbx2, bby1:bby2] = images[rand_index, :, bbx1:bbx2, bby1:bby2]
                # adjust lambda to exactly match pixel ratio
                lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (images.size()[-1] * images.size()[-2]))
                # compute output
                output = model(images)
                batch_loss = criterion(output, target_a) * lam + criterion(output, target_b) * (1. - lam)
            elif args.data_aug == "cutout" and args.beta > 0 and r < args.cutout_prob:
                # generate cut sample
                lam = np.random.beta(args.beta, args.beta)
                bbx1, bby1, bbx2, bby2 = rand_bbox(images.size(), lam)
                images[:, :, bbx1:bbx2, bby1:bby2] = 0
                # compute output
                output = model(images)
                batch_loss = criterion(output, labels)
            elif args.data_aug == "mixup" and args.beta > 0 and r < args.mixup_prob:
                # generate mixed sample
                lam = np.random.beta(args.beta, args.beta)
                rand_index = torch.randperm(images.size()[0]).cuda()
                target_a = labels
                target_b = labels[rand_index]
                images = lam * images + (1 - lam) * images[rand_index, :]
                # compute output
                output = model(images)
                batch_loss = criterion(output, target_a) * lam + criterion(output, target_b) * (1. - lam)
            else:
                # compute output
                output = model(images)
                batch_loss = criterion(output, labels)

            optimer.zero_grad()
            batch_loss.backward()
            optimer.step()
            
            loss += batch_loss
            writer.add_scalar("train_batch/loss", batch_loss, cur_batch)
            output_buffer["train_batch/loss"].append(batch_loss.detach().cpu().item())
            cur_batch += 1
        
        if scheduler is not None:
            scheduler.step()
        
        writer.add_scalar("train/loss", loss, cur_epoch)
        output_buffer["train/loss"].append(loss.detach().cpu().item())
        cur_lr = scheduler.get_last_lr()[0] if scheduler is not None else args.lr
        info_str = "EP {0}, Loss {1:.2f}, LR {2:.4f} ".format(cur_epoch, loss, cur_lr)

        f1, acc = test_model(model, testloader)
        writer.add_scalar("test/accuracy", acc, cur_epoch)
        writer.add_scalar("test/f1-score", f1,  cur_epoch)
        output_buffer["test/accuracy"].append(acc)
        output_buffer["test/f1-score"].append(f1)
        info_str += "[Test] F1 {0:.3f} ACC {1:.3f} ".format(f1, acc)

        pbar.set_postfix_str(info_str)

        for name, layer in model.named_parameters():
            writer.add_histogram("{}_grad".format(name), layer.grad.detach().cpu().data.numpy(), cur_epoch)
            writer.add_histogram("{}_data".format(name), layer.detach().cpu().data.numpy(), cur_epoch)

        if acc > best_acc:
            best_acc, best_f1, best_epoch = acc, f1, cur_epoch
            clean_dir(ckpt_dir)
            save_model(model, path=os.path.join(ckpt_dir, f"BEST_Acc{best_acc:.3f}_{paramstr}.pt"))
    
    model.eval() 
    writer.close()

    print("\n")
    print("Best Outcome: [Epoch: {0}], [F1: {1:.5f}, ACC: {2:.5f}]"
          .format(best_epoch, best_f1, best_acc))
    save_model(model, path=os.path.join(ckpt_dir, f"EP{max_epoch}_Acc{acc:.3f}_{paramstr}.pt"), pt=True)
    # added
    save_intermediate_results(output_buffer, path=os.path.join(
        ckpt_dir, "intermediate_results.pkl"), pt=True)

    return model


def save_model(model, path, pt=False):
    from pathlib import Path

    file_path = Path(path)
    file_path.touch(exist_ok=True)

    torch.save(model.state_dict(), file_path)
    if pt: print("Saving model into {}".format(file_path))


def save_intermediate_results(output_buffer, path, pt=False):
    from pathlib import Path

    file_path = Path(path)
    file_path.touch(exist_ok=True)

    with open(file_path, "wb") as fp:
        pickle.dump(output_buffer, fp)
    
    if pt: print("Saving intermediate results into {}".format(file_path))


def clean_dir(save_dir):
    for fn in os.listdir(save_dir):
        if "BEST" in fn:
            os.remove(os.path.join(save_dir, fn))


def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

