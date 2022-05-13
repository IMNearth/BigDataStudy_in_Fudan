import os
import sys
import logging

import time
import random
import argparse

import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader

import pkg.utils as U
from pkg.model import YoloV3, YoloV3Loss
from pkg.config import cfg
from pkg.eval import Evaluator

from tensorboardX import SummaryWriter


class Trainer(object):

    # possible_sizes = [x * 32 for x in range(10,20)]

    def __init__(self, weight_path, output_path, resume, gpu_id):
        U.tools.init_seeds(0)
        
        self.device = U.select_device(gpu_id)
        self.start_epoch = 0
        self.best_mAP = 0.0
        self.epochs = cfg.TRAIN["EPOCHS"]
        self.weight_path = weight_path
        self.multi_scale_train = cfg.TRAIN["MULTI_SCALE_TRAIN"]

        time_str = time.strftime('%Y-%m%d-%H:%M', time.localtime(time.time()))
        self.output_path = os.path.join(output_path, time_str)
        if not os.path.exists(self.output_path):
            os.mkdir(self.output_path)
        
        self.writer = SummaryWriter(self.output_path)
        month_date = "-".join(time_str.split("-")[:2])
        self.logger = U.setup_logger("YoloV3", save_dir=output_path, filename=f"log_{time_str}.txt")
    
        self.logger.info("-----------------------------")
        self.logger.info("----      Preparing      ----")
        self.logger.info("-----------------------------")
        
        self.trainset = U.VocDataset(
            anno_file_type="train", 
            img_size=cfg.TRAIN["TRAIN_IMG_SIZE"]
        )
        self.train_loader = DataLoader(
            self.trainset,
            batch_size=cfg.TRAIN["BATCH_SIZE"],
            num_workers=cfg.TRAIN["NUMBER_WORKERS"],
            shuffle=True
        )
        self.logger.info("[1] Dataset successfully loaded!")
        
        self.model = YoloV3().to(self.device)
        self.__load_model_weights(resume)
        self.logger.info("[2] Model successfully initialized!")

        self.optimizer = optim.SGD(
            self.model.parameters(), 
            lr=cfg.TRAIN["LR_INIT"],
            momentum=cfg.TRAIN["MOMENTUM"], 
            weight_decay=cfg.TRAIN["WEIGHT_DECAY"]
        )
        self.logger.info("[3] Optimizer successfully assigned!")

        self.scheduler = U.CosineDecayLR(
            self.optimizer,
            T_max=self.epochs*len(self.train_loader),
            lr_init=cfg.TRAIN["LR_INIT"],
            lr_min=cfg.TRAIN["LR_END"],
            warmup=cfg.TRAIN["WARMUP_EPOCHS"]*len(self.train_loader)
        )
        self.logger.info("[4] Scheduler successfully assigned!")

        self.criterion = YoloV3Loss(
            anchors=cfg.MODEL["ANCHORS"], 
            strides=cfg.MODEL["STRIDES"],
            iou_threshold_loss=cfg.TRAIN["IOU_THRESHOLD_LOSS"]
        )
        self.logger.info("[5] Crtterion successfully built!")

    def __load_model_weights(self, resume):
        if resume:
            last_weight = os.path.join(self.output_path, "last.pt")
            chkpt = torch.load(last_weight, map_location=self.device)
            self.model.load_state_dict(chkpt['model'])

            self.start_epoch = chkpt['epoch'] + 1
            if chkpt['optimizer'] is not None:
                self.optimizer.load_state_dict(chkpt['optimizer'])
                self.best_mAP = chkpt['best_mAP']
            del chkpt
        else:
            self.model.load_darknet_weights(self.weight_path)


    def __save_model_weights(self, epoch, mAP):
        if mAP > self.best_mAP:
            self.best_mAP = mAP
        
        best_weight = os.path.join(self.output_path, f"best_mAP:{self.best_mAP*100:.1f}.pt")
        last_weight = os.path.join(self.output_path, f"last_mAP:{mAP*100:.1f}.pt")
        
        chkpt = {
            'epoch': epoch,
            'best_mAP': self.best_mAP,
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }
        U.clean_dir(self.output_path, name_keys=["last"])
        torch.save(chkpt, last_weight)

        if self.best_mAP == mAP:
            U.clean_dir(self.output_path, name_keys=["best"])
            torch.save(chkpt['model'], best_weight)

        if epoch > 0 and epoch % 10 == 0:
            torch.save(chkpt, os.path.join(self.output_path, 'backup_epoch%g.pt'%epoch))
        del chkpt

    def train(self):
        logger = self.logger
        writer = self.writer
        
        logger.info("-----------------------------")
        logger.info("---- Now start training! ----")
        logger.info("-----------------------------")
        logger.info("Train datasets number is : {}".format(len(self.trainset)))

        for epoch in range(self.start_epoch+1, self.epochs+1):
            self.model.train()

            mloss = torch.zeros(4)
            for i, (imgs, label_sbbox, label_mbbox, label_lbbox, sbboxes, mbboxes, lbboxes) \
                in enumerate(self.train_loader):

                cur_iter = len(self.train_loader)*epoch + i
                self.scheduler.step(cur_iter)

                imgs = imgs.to(self.device)
                label_sbbox = label_sbbox.to(self.device)
                label_mbbox = label_mbbox.to(self.device)
                label_lbbox = label_lbbox.to(self.device)
                sbboxes = sbboxes.to(self.device)
                mbboxes = mbboxes.to(self.device)
                lbboxes = lbboxes.to(self.device)

                p, p_d = self.model(imgs)

                loss, loss_giou, loss_conf, loss_cls = self.criterion(p, p_d, 
                    label_sbbox, label_mbbox, label_lbbox, sbboxes, mbboxes, lbboxes)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # Update running mean of tracked metrics
                loss_items = torch.tensor([loss_giou, loss_conf, loss_cls, loss])
                mloss = (mloss * i + loss_items) / (i + 1)

                # Print batch results
                if i % 10 == 0:
                    info_str = f"Epoch: {epoch:2d}/{self.epochs}," + \
                        f" Batch: {i+1:2d}/{len(self.train_loader)}," + \
                        f" Image Size: {self.trainset.img_size}," + \
                        f" Loss_GIoU: {loss_giou:.3f}({mloss[0]:.2f})," + \
                        f" Loss_Conf: {loss_conf:.3f}({mloss[1]:.2f})," + \
                        f" Loss_Cls : {loss_cls:.3f}({mloss[2]:.2f})," + \
                        f" Total Loss: {loss:.3f}({mloss[3]:.2f})," + \
                        f" LR: {self.optimizer.param_groups[0]['lr']:g} ."
                    logger.info(info_str)

                    writer.add_scalar("single_iter/total_loss", loss, cur_iter)
                    writer.add_scalar("single_iter/loss_giou", loss_giou, cur_iter)
                    writer.add_scalar("single_iter/loss_conf", loss_conf, cur_iter)
                    writer.add_scalar("single_iter/loss_cls", loss_cls, cur_iter)

                    writer.add_scalar("running_mean/total_loss", mloss[3], cur_iter)
                    writer.add_scalar("running_mean/loss_giou", mloss[0], cur_iter)
                    writer.add_scalar("running_mean/loss_conf", mloss[1], cur_iter)
                    writer.add_scalar("running_mean/loss_cls", mloss[2], cur_iter)

                # multi-sclae training every 10 batches
                if self.multi_scale_train and (i+1) % 10 == 0:
                    # self.trainset.img_size = random.choice(self.possible_sizes)
                    self.trainset.img_size = random.choice(range(10,20)) * 32

            mAP = 0.0
            if epoch >= 20:
                logger.info('*'*20 + " Validate " + '*'*20)
                with torch.no_grad():
                    APs = Evaluator(self.model).APs_voc()
                    for i in APs:
                        logger.info("{} --> mAP : {:.3f}".format(i, APs[i]))
                        mAP += APs[i]
                    mAP = mAP / self.trainset.num_classes
                    
                    logger.info('mAP:%g'%(mAP))
                    writer.add_scalar("eval/mAP", mAP, epoch)

            self.__save_model_weights(epoch, mAP)
            logger.info('BEST mAP : %g' % (self.best_mAP))



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--weight_path', type=str, default='./pretrained/darknet53_448.weights', help='weight file path')
    parser.add_argument('--output_path', type=str, default='./outputs', help='output path')
    parser.add_argument('--resume', action='store_true', default=False,  help='resume training flag')
    parser.add_argument('--gpu_id', type=int, default=0, help='gpu id')
    opt = parser.parse_args()

    trainer = Trainer(
        weight_path=opt.weight_path, 
        output_path=opt.output_path,
        resume=opt.resume, 
        gpu_id=opt.gpu_id
    )
    trainer.train()

