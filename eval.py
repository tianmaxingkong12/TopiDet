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

from PIL import Image
from utils import *
from options.helper import init_log

import misc_utils as utils
import pdb


def evaluate(model, dataloader, epoch, writer, logger, data_name='val'):
    # 每个模型的evaluate方式不同
    metrics = model.evaluate(dataloader, epoch, writer, logger, data_name)
    return metrics


if __name__ == '__main__':
    from options import opt
    from network import get_model
    import misc_utils as utils
    from mscv.summary import create_summary_writer

    if not opt.load and 'LOAD' not in config.MODEL:
        print('Usage: eval.py [--tag TAG] --load LOAD')
        raise_exception('eval.py: the following arguments are required: --load')

    Model = get_model(config.MODEL.NAME)
    model = Model(config)
    model = model.to(device=opt.device)

    if opt.load:
        which_epoch = model.load(opt.load)
    elif 'LOAD' in config.MODEL:
        which_epoch = model.load(config.MODEL.LOAD)
    else:
        which_epoch = 0

    model.eval()
    
    print('Start evaluating...')
    
    ## TODO EXP_ID 与之前的训练的实验配置一致
    EXP_ID = datetime.datetime.strftime(datetime.datetime.now(),"%Y%m%d-%H%M%S")
    config.WANDB.EXP_ID = EXP_ID
    config.WANDB.EXP_NAME = opt.tag
    config.WANDB.JOB_TYPE = "test"
    # 初始化路径
    log_root = os.path.join("./logs", opt.tag, EXP_ID)
    
    config.LOG.ROOT_DIR = log_root
    logger = init_log(log_root)
    
    run_dir = os.path.join(log_root,"runs")
    utils.try_make_dir(run_dir)
    tensorboard_dir = os.path.join(run_dir,"tensorboard")
    utils.try_make_dir(tensorboard_dir)
    writer = create_summary_writer(tensorboard_dir)
    if opt.start_wandb and "WANDB" in config:
        wandb_dir = os.path.join(run_dir, "wandb")
        utils.try_make_dir(wandb_dir)
        run = wandb.init(
                project = config.WANDB.PROJECT_NAME,
                group= config.WANDB.GROUP_NAME,
                job_type= config.WANDB.JOB_TYPE,
                name = opt.tag,
                config= config,
                dir = wandb_dir
            )
    logger.info('===========================================')
    if test_dataloader is not None:
        logger.info('test_trasforms: ' +str(test_dataloader.dataset.transforms))
    logger.info('===========================================')
    # 测试集为test时输出详细信息
    metrics = evaluate(model, test_dataloader, which_epoch, writer, logger, 'test')
    data = pd.DataFrame(columns = ["编号","类别名称","训练集图片数","训练集框数","测试集图片数","测试集框数","测试集AP50","测试集AP75"])
    for _id, class_name in enumerate(config.DATA.CLASS_NAMES):
        data.loc[_id,"编号"] = _id
        data.loc[_id,"类别名称"] = class_name
        data.loc[_id,"训练集图片数"] = train_dataset.datasets[0].image_counter[class_name]
        data.loc[_id,"训练集框数"] = train_dataset.datasets[0].bbox_counter[class_name]
        data.loc[_id,"测试集图片数"] = test_dataset.datasets[0].image_counter[class_name]
        data.loc[_id,"测试集框数"] = test_dataset.datasets[0].bbox_counter[class_name]
        data.loc[_id,"测试集AP50"] = metrics['test/AP50_EachClass'][_id]
        data.loc[_id,"测试集AP75"] = metrics['test/AP75_EachClass'][_id]
    data.to_excel(os.path.join(log_root,"results.xlsx"),index= False)
    metric_info = {
        "train_images":len(train_dataset),
        "train_bbox": train_dataset.datasets[0].tot_bbox,
        "test_images": len(test_dataset),
        "test_bbox": test_dataset.datasets[0].tot_bbox,
        "test_mAP50":metrics["test/AP50"],
        "test_mAP75":metrics["test/AP75"],
        "test_mAP50:95":metrics["test/AP50-AP95"] 
    }
    if opt.start_wandb and "WANDB" in config:
        wandb.log({"Results":wandb.Table(dataframe = data)})
        # wandb.log(metric_info)
        run.finish()
        wandb.finish()

    
    
    



    

