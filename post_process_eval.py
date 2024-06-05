# encoding=utf-8
import torch
import datetime
import pandas as pd
torch.multiprocessing.set_sharing_strategy('file_system')  # ulimit -SHn 51200

from dataloader.dataloaders import test_dataloader, train_dataset, test_dataset
from options import opt, config
from mscv.summary import write_loss, write_image
from mscv.image import tensor2im
import wandb

from utils import *
from options.helper import init_log

import misc_utils as utils
import pdb
from network import get_model
import misc_utils as utils
import copy


def evaluate(model, dataloader, epoch, writer, logger, data_name='val'):
    # 每个模型的evaluate方式不同
    metrics = model.evaluate(dataloader, epoch, writer, logger, data_name)
    return metrics

confidence_thresholds = [0.05,0.1,0.2,0.3,0.4,0.5] #6
nms_iou_thresholds = [1.0,0.75,0.65,0.5,0.3,0.0] #6
match_iou_thresholds = [0.5,0.75,0.95] #3
cat_names = ["sheep"]
gt_path = "./datasets/VOC07_12/annotations/instances_test2007.json"
if __name__ == '__main__':
    if not opt.load and 'LOAD' not in config.MODEL:
        print('Usage: eval.py [--tag TAG] --load LOAD')
        raise_exception('eval.py: the following arguments are required: --load')
    # if config.DATA.DATASET == ["voc_coco"]:
    #     config.DATA.NUM_CLASSESS = 90 ##对于COCO
    convert_COCO_label = False
    # if config.DATA.NUM_CLASSESS == 80:
    #     convert_COCO_label = True
    for c_t in confidence_thresholds:
        for n_i_t in nms_iou_thresholds:
            _config = copy.deepcopy(config)
            Model = get_model(_config.MODEL.NAME)
            _config.TEST.NMS_THRESH = n_i_t
            _config.TEST.CONF_THRESH = c_t
            model = Model(_config)
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
            output_file = os.path.join(log_root, "pred.json")
            with open(output_file, "w") as f:
                json.dump(coco_result, f)
            for m_i_t in match_iou_thresholds:
                info = dict()
                info["dataset"] = "VOC-FasterRCNN-postprocess"
                info["cat_names"] = cat_names
                info["confidence_threshold"] = c_t
                info["nms_iou_threshold"] = n_i_t
                info["match_iou_threshold"] = m_i_t
                post_process_analyse(gt_path, output_file, info)

                if opt.start_wandb and "WANDB" in _config:
                    _config.WANDB.PROJECT_NAME = ""
                    _config.WANDB.EXP_ID = EXP_ID
                    _config.WANDB.GROUP_NAME = "post_process"
                    _config.WANDB.JOB_TYPE = "evaluate"
                    wandb_dir = os.path.join(run_dir, "wandb")
                    utils.try_make_dir(wandb_dir)
                    run = wandb.init(
                            project = _config.WANDB.PROJECT_NAME,
                            group= _config.WANDB.GROUP_NAME,
                            job_type= _config.WANDB.JOB_TYPE,
                            name = "conf{}_nms{}_iou{}".format(c_t,n_i_t,m_i_t),
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
                    run.finish()
                    wandb.finish()

    
    
    



    

