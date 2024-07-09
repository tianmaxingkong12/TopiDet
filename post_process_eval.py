# encoding=utf-8
import os
import json
import copy
import datetime

import wandb
import pandas as pd
import torch
torch.multiprocessing.set_sharing_strategy('file_system')  # ulimit -SHn 51200
import misc_utils as utils

from dataloader.dataloaders import test_dataloader
from options import opt, config
from utils.post_process import post_process_analyse, nms
from utils.utils import raise_exception
from options.helper import init_log
from network import get_model



confidence_thresholds = [0.05] #1
nms_iou_thresholds = [1.0,0.9,0.8,0.75,0.7,0.65,0.60,0.55,0.5,0.45,0.4,0.35,0.3,0.2,0.1,0.0] #16
# match_iou_thresholds = [0.3,0.4,0.5,0.75] #4
match_iou_thresholds = [0.5]
cat_names = ["sheep"]
gt_path = "./datasets/VOC07_12/annotations/instances_test2007.json"


if __name__ == '__main__':
    if not opt.load and 'LOAD' not in config.MODEL:
        print('Usage: eval.py [--tag TAG] --load LOAD')
        raise_exception('eval.py: the following arguments are required: --load')
    convert_COCO_label = False
    _config = copy.deepcopy(config)
    c_t = confidence_thresholds[0]
    nms_t = 1
    _config = copy.deepcopy(config)
    Model = get_model(_config.MODEL.NAME)
    _config.TEST.NMS_THRESH = nms_t
    _config.TEST.CONF_THRESH = c_t
    model = Model(_config,box_detections_per_img=1e6)
    model = model.to(device=opt.device)
    if opt.load:
        which_epoch = model.load(opt.load)
    elif 'LOAD' in _config.MODEL:
        which_epoch = model.load(_config.MODEL.LOAD)
    else:
        which_epoch = 0
    model.eval()        
    print('Start evaluating...')
    EXP_ID = datetime.datetime.strftime(datetime.datetime.now(),"%Y%m%d-%H%M%S")
    log_root = os.path.join("./logs", opt.tag, EXP_ID)
    _config.LOG.ROOT_DIR = log_root
    logger = init_log(log_root)
    run_dir = os.path.join(log_root,"runs")
    utils.try_make_dir(run_dir)
    logger.info('===========================================')
    if test_dataloader is not None:
        logger.info('test_trasforms: ' +str(test_dataloader.dataset.transforms))
    logger.info('===========================================')
    # 测试集为test时输出详细信息
    coco_result = model.save_preds(test_dataloader, convert_COCO_label)
    for n_i_t in nms_iou_thresholds:
        process_result = nms(coco_result,n_i_t)
        output_file = os.path.join(log_root, "pred_{}.json".format(n_i_t))
        with open(output_file, "w") as f:
            json.dump(process_result, f)
        for m_i_t in match_iou_thresholds:
            info = dict()
            info["dataset"] = "VOC-FasterRCNN-postprocess"
            info["cat_names"] = cat_names
            info["confidence_threshold"] = c_t
            info["nms_iou_threshold"] = n_i_t
            info["match_iou_threshold"] = m_i_t
            post_process_analyse(gt_path, output_file, info)

            if opt.start_wandb and "WANDB" in _config:
                _config.WANDB.PROJECT_NAME = "PostProcess2"
                _config.WANDB.EXP_ID = EXP_ID
                _config.WANDB.GROUP_NAME = "post_process"
                _config.WANDB.JOB_TYPE = "evaluate"
                wandb_dir = os.path.join(run_dir, "wandb")
                utils.try_make_dir(wandb_dir)
                run = wandb.init(
                        project = _config.WANDB.PROJECT_NAME,
                        group= _config.WANDB.GROUP_NAME,
                        job_type= _config.WANDB.JOB_TYPE,
                        name = "{}_conf{}_nms{}_iou{}".format(cat_names[0],c_t,n_i_t,m_i_t),
                        config= _config,
                        dir = wandb_dir
                    )
                for k,v in info.items():
                    if k not in ["recall", "precision","f1-score", "scores"]:
                        wandb.config[k] = v
                # wandb.config(info)
                ## precision, recall, f1-score, scores
                data = [[x, y, z, a] for (x, y, z, a) in zip(info["recall"], info["precision"], info["f1-score"], info["scores"])]
                table = wandb.Table(data=data, columns = ["recall", "precision", "f1-score", "confidence"])
                wandb.log({"PR-curve" : wandb.plot.line(table,
                                "recall", "precision", title="PR-curve")})
                wandb.log({"P-confidence" : wandb.plot.line(table,
                                "confidence", "precision", title="P-confidence")})
                wandb.log({"R-confidence" : wandb.plot.line(table,
                                "confidence", "recall", title="R-confidence")})
                wandb.log({"f1-confidence" : wandb.plot.line(table,
                                "confidence", "f1-score", title="f1-confidence")})
                data2 = [[x, y] for (x, y) in zip(info["recThrs"], info["Precision_101"])]
                table2 = wandb.Table(data=data2, columns = ["recThrs","Precision_101"])
                wandb.log({"101_AP": wandb.plot.line(table2,"recThrs","Precision_101",title="101_AP")})
                run.finish()
                wandb.finish()

    
    
    



    

