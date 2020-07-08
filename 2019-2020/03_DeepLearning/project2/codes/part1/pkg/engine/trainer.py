import torch
import torch.nn.functional as F
import pickle
import time
from tensorboardX import SummaryWriter

from tqdm import tqdm
from colorama import Fore
import os

from pkg.engine.test import test_model


def train(model, trainloader, optimer, testloader, args, scheduler=None):
    max_epoch = args.max_epoch
    best_acc, best_f1, best_epoch = 0, 0, 0

    cur_time = time.strftime("%m-%dT%H-%M", time.localtime())
    log_path = os.path.join(args.save_path, "log", args.classifier)
    if not os.path.exists(log_path):
        os.mkdir(log_path)
    paramstr = "activation_{}_".format(args.activation)
    writer = SummaryWriter(log_dir=os.path.join(log_path, cur_time + "_" + paramstr))
    
    if torch.cuda.is_available():
        model = model.cuda()

    # pbar = tqdm(range(max_epoch), 
    #             desc="Training",
    #             unit='epoch',
    #             bar_format="{l_bar}%s{bar}%s{r_bar}" % (Fore.BLUE, Fore.RESET))
    
    output_buffer = {"train_batch/loss": [], "train/loss": [], "test/accuracy": [], "test/f1-score": []}
    cur_batch=0
    
    for cur_epoch in range(max_epoch):
        model.train()
        
        loss = 0
        for data_batch in trainloader:
            images, labels = data_batch
            if torch.cuda.is_available():
                images, labels = images.cuda(), labels.cuda()
            
            x = model(images)
            if args.loss == "cross_entropy":
                batch_loss = F.cross_entropy(x, labels)
            elif args.loss == "margin":
                x = F.softmax(x, dim=1)
                batch_loss = F.multi_margin_loss(x, labels)
            else:
                raise NotImplementedError

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
        print("Epoch [{0}] Train loss: [{1:.5f}]".format(cur_epoch, loss), end=" ")

        f1, acc = test_model(model, testloader)
        writer.add_scalar("test/accuracy", acc, cur_epoch)
        writer.add_scalar("test/f1-score", f1,  cur_epoch)
        output_buffer["test/accuracy"].append(acc)
        output_buffer["test/f1-score"].append(f1)
        print( "Test: [F1: {0:.5f}, ACC: {1:.5f}]".format(f1, acc), end="\r")

        for name, layer in model.named_parameters():
            writer.add_histogram("{}_grad".format(name), layer.grad.detach().cpu().data.numpy(), cur_epoch)
            writer.add_histogram("{}_data".format(name), layer.detach().cpu().data.numpy(), cur_epoch)

        if acc > best_acc:
            best_acc, best_f1, best_epoch = acc, f1, cur_epoch
            save_model(model, path=os.path.join(args.save_path, args.classifier, "BEST_" + paramstr + "__" + cur_time+".pt"))
    
    model.eval() 
    writer.close()

    print("\n")
    print("Best Outcome: [Epoch: {0}], [F1: {1:.5f}, ACC: {2:.5f}]"
          .format(best_epoch, best_f1, best_acc))
    save_model(model, path=os.path.join(args.save_path, args.classifier, paramstr + "__" +cur_time+".pt"), pt=True)
    # added
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

    file_path = Path(path)
    file_path.touch(exist_ok=True)

    with open(file_path, "wb") as fp:
        pickle.dump(output_buffer, fp)
    
    if pt: print("Saving intermediate results into {}".format(file_path))

