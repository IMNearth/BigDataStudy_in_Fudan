import os
import argparse
import cv2
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

import pkg.utils as U
from pkg.model import YoloV3
from pkg.config import cfg
from pkg.eval import Evaluator
from pkg.utils.visualize import *


class Tester(object):
    def __init__(self, 
        weight_path, output_path, 
        conf_thresh=None, iou_thresh=0.5, gpu_id=0, visual=None, do_eval=False
    ):
        
        self.weight_path = weight_path
        self.output_path = output_path
        self.iou_thresh = iou_thresh
        self.device = U.select_device(gpu_id)
        self.visual = visual
        self.do_eval = do_eval
        self.test_data_path = os.path.join(cfg.TEST_DATA_PATH, 'VOC2007')
        
        self.conf_threshold = cfg.TEST["CONF_THRESH"] if conf_thresh is None else conf_thresh
        self.classes = cfg.DATA["CLASSES"]                      # VOC CLASS
        self.num_class = cfg.DATA["NUM"]                        # VOC CLASS NUM: 20
        self.nms_threshold = cfg.TEST["NMS_THRESH"]             # 0.5
        self.multi_scale_test = cfg.TEST["MULTI_SCALE_TEST"]    # False
        self.flip_test = cfg.TEST["FLIP_TEST"]                  # False

        self.model = YoloV3().to(self.device)
        self.__load_model_weights(weight_path)

        self.evalter = Evaluator(self.model, visiual=False)

    def __load_model_weights(self, weight_path):
        logger.info("Loading weight file from : {}".format(weight_path))

        weight = os.path.join(weight_path)
        chkpt = torch.load(weight, map_location=self.device)
        self.model.load_state_dict(chkpt['model'])
        
        logger.info("Loading weight file is done!")
        del chkpt

    def test(self):
        if self.visual:
            img_inds_file = os.path.join("outputs/images/other_images", "other.txt")
            # img_inds_file = os.path.join(self.test_data_path,  'ImageSets', 'Main', 'test.txt')
            with open(img_inds_file, 'r') as f:
                lines = f.readlines()
                img_inds = [line.strip() for line in lines]

            for img_ind in tqdm(img_inds):
                img_path = os.path.join("outputs/images/other_images", img_ind+'.jpeg')
                # img_path = os.path.join(self.test_data_path, 'JPEGImages', img_ind+'.jpg')
                info_str = "Test images : {}".format(img_path)

                img = cv2.imread(img_path)
                assert img is not None

                bboxes_prd = self.evalter.get_bbox(img)
                if bboxes_prd.shape[0] != 0:
                    boxes = bboxes_prd[..., :4]
                    class_inds = bboxes_prd[..., 5].astype(np.int32)
                    scores = bboxes_prd[..., 4]

                    visualize_boxes(
                        image=img, 
                        boxes=boxes, 
                        labels=class_inds, 
                        probs=scores, 
                        class_labels=self.classes
                    )
                    save_path = os.path.join(f"{self.output_path}/images", "processed_"+img_ind+'.jpg')

                    cv2.imwrite(save_path, img)
                    info_str += " ==> Saved images : {}".format(save_path)
                logger.info(info_str)

        if self.do_eval:
            mAP = 0
            logger.info('*' * 20 + " Validate " + '*' * 20)

            with torch.no_grad():
                APs = Evaluator(self.model).APs_voc(
                    multi_test=self.multi_scale_test, 
                    flip_test=self.flip_test,
                    iou_thresh=self.iou_thresh
                )

                for i in APs:
                    logger.info("{} --> mAP : {:.4f}".format(i, APs[i]))
                    mAP += APs[i]
                mAP = mAP / self.num_class
                logger.info('mAP:%g' % (mAP))



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--weight-path', type=str, default='outputs/best_param/best_84.0.pt', help='weight file path')
    parser.add_argument('--output-path', type=str, default='./outputs', help='output path')
    parser.add_argument('--visual', action='store_true', default=False, help='test data path or None')
    parser.add_argument('--eval', action='store_true', default=False, help='eval the mAP or not')
    parser.add_argument('--gpu-id', type=int, default=0, help='gpu id')
    parser.add_argument('--conf-thresh', type=float, default=0.01, help='confidence threshold')
    parser.add_argument('--iou-thresh', type=float, default=0.5, help='evaluation iou threshold')
    opt = parser.parse_args()

    global logger
    logger = U.setup_logger("YoloV3", save_dir=opt.output_path, filename="log_eval.txt")
    logger.info(opt)

    tester = Tester(
        weight_path=opt.weight_path,
        output_path=opt.output_path,
        gpu_id=opt.gpu_id,
        do_eval=opt.eval,
        visual=opt.visual,
        conf_thresh=opt.conf_thresh,
        iou_thresh=opt.iou_thresh,
    )
    tester.test()
    logger.info('\n')
