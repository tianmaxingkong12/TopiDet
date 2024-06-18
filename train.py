# encoding = utf-8
import os
import pdb
import time
import numpy as np
import datetime
import math

import torch
torch.multiprocessing.set_sharing_strategy('file_system')
import torch.distributed as dist
from torch import optim
from torch.autograd import Variable

from options.helper import is_distributed, is_first_gpu, setup_multi_processes
from options import opt, config
import wandb

# 设置多卡训练
if is_distributed():
    setup_multi_processes()

from dataloader.dataloaders import train_dataloader, val_dataloader

from network import get_model
from eval import evaluate

from options.helper import init_log, load_meta, save_meta
from utils import seed_everything
from scheduler import schedulers

from mscv.summary import create_summary_writer, write_meters_loss, write_image
from mscv.image import tensor2im
# from utils.send_sms import send_notification

import misc_utils as utils
import random
import albumentations as A

from albumentations.pytorch.transforms import ToTensorV2

# 初始化
with torch.no_grad():
    # 设置随机种子
    if 'RANDOM_SEED' in config.MISC:
        seed_everything(config.MISC.RANDOM_SEED)
    

    # dataloader
    train_dataloader = train_dataloader
    val_dataloader = val_dataloader

    if is_first_gpu():
        # 初始化日志系统
        # 日志系统包括config、log.txt、checkpoint文件夹、runs:tensorboard保存路径
        # 日志系统文件夹层次
        # -logs
        # -EXP_NAME
        #   |-EXP_ID
        #       |--config.py
        #       |--log.txt
        #       |--checkpoints
        #       |--runs
        EXP_ID = datetime.datetime.strftime(datetime.datetime.now(),"%Y%m%d-%H%M%S")
        if opt.start_wandb and "WANDB" in config:
            config.WANDB.EXP_ID = EXP_ID
            config.WANDB.EXP_NAME = opt.tag
        # 初始化路径
        log_root = os.path.join("./logs", opt.tag, EXP_ID)
        checkpoint_dir = os.path.join(log_root, "checkpoints")
        run_dir = os.path.join(log_root,"runs")
        utils.try_make_dir(checkpoint_dir)
        utils.try_make_dir(run_dir)
        config.LOG.ROOT_DIR = log_root

        # 初始化日志
        logger = init_log(log_root)

        # 初始化训练的meta信息
        meta = load_meta(log_root, new=True)
        save_meta(log_root, meta) #保存训练数据信息

    # 初始化模型
    Model = get_model(config.MODEL.NAME)
    model = Model(config)

    # 转到GPU
    # model = model.to(device=opt.device)

    if opt.load:
        load_epoch = model.load(opt.load)
        start_epoch = load_epoch + 1 if opt.resume or 'RESUME' in config.MISC else 1
    elif 'LOAD' in config.MODEL:
        load_epoch = model.load(config.MODEL.LOAD)
        start_epoch = load_epoch + 1 if opt.resume or 'RESUME' in config.MISC else 1
    else:
        start_epoch = 1

    model.train()

    if is_first_gpu():
        # 开始训练
        print('Start training...')
    
    ## 默认都是加载的以epoch保存的模型
    if config.OPTIMIZE.USE_EPOCH:        
        start_step = (start_epoch - 1) * len(train_dataloader)
        global_step = start_step
        end_epoch = config.OPTIMIZE.EPOCHS
        total_steps = end_epoch * len(train_dataloader)
    else:
        start_step = (start_epoch - 1) * len(train_dataloader)
        global_step = start_step
        total_steps = config.OPTIMIZE.STEPS
        end_epoch = math.ceil(total_steps / len(train_dataloader))

    start = time.time()

    # 定义scheduler
    scheduler = model.scheduler

    if is_first_gpu():
        # tensorboard日志
        tensorboard_dir = os.path.join(run_dir,"tensorboard")
        utils.try_make_dir(tensorboard_dir)
        writer = create_summary_writer(tensorboard_dir)
        # wandb可视化训练
        if opt.start_wandb and "WANDB" in config:
            wandb_dir = os.path.join(run_dir, "wandb")
            utils.try_make_dir(wandb_dir)
            wandb.init(
                project = config.WANDB.PROJECT_NAME,
                group= config.WANDB.GROUP_NAME,
                job_type= config.WANDB.JOB_TYPE,
                name = opt.tag,
                config= config,
                dir = wandb_dir
            )
            wandb.define_metric("steps")
            wandb.define_metric("train/*",step_metric = "steps")
            wandb.define_metric("val/*",step_metric = "steps")
            if config.DATA.DATASET == ["Lesion4K"]:
                wandb.define_metric("val/AP50(11)",summary="max")
                wandb.define_metric("val/AP50", summary="max")

    else:
        writer = None

    # 在日志记录transforms
    if is_first_gpu():
        logger.info('train_trasforms: ' +str(train_dataloader.dataset.transforms))
        logger.info('===========================================')
        if val_dataloader is not None:
            logger.info('val_trasforms: ' +str(val_dataloader.dataset.transforms))
        logger.info('===========================================')

        # 在日志记录scheduler
        if config.OPTIMIZE.SCHEDULER in schedulers:
            logger.info('scheduler: (Lambda scheduler)\n' + str(schedulers[config.OPTIMIZE.SCHEDULER]))
            logger.info('===========================================')

# 训练循环
#try:
if __name__ == '__main__':
    eval_result = ''
    val_AP50 = best_AP50 = 0
    val_loss = best_loss = 100

    for epoch in range(start_epoch, end_epoch + 1):
        if is_distributed():
            train_dataloader.sampler.set_epoch(epoch)
        for iteration, sample in enumerate(train_dataloader):
            global_step += 1

            base_lr = config.OPTIMIZE.BASE_LR
            if global_step < 500:
                # 500个step从0.001->1.0
                lr = (0.001 + 0.999 / 499 * (global_step - 1)) * base_lr
                for param_group in model.optimizer.param_groups:
                    param_group['lr'] = lr

            elif global_step == 500:
                for param_group in model.optimizer.param_groups:
                    param_group['lr'] = base_lr
            elif global_step > total_steps:
                break

            # 计算剩余时间
            rate = (global_step - start_step) / (time.time() - start)
            remaining = (total_steps - global_step) / rate

            # --debug模式下只训练10个batch
            if opt.debug and iteration > 10:
                break

            sample['global_step'] = global_step
     
            #  更新网络参数
            updated = model.update(sample)
            predicted = updated.get('predicted')

            pre_msg = 'Epoch:%d' % epoch

            lr = model.optimizer.param_groups[0]['lr']
            # 显示进度条
            msg = f'lr:{round(lr, 6) : .6f} (loss) {str(model.avg_meters)} ETA: {utils.format_time(remaining)}'
            if is_first_gpu():
                utils.progress_bar(iteration, len(train_dataloader), pre_msg, msg)
            # print(pre_msg, msg)

            ## 记录训练集损失 训练集损失根据每个step记录
            if is_first_gpu():
                write_meters_loss(writer, 'train', model.avg_meters, global_step)
                if opt.start_wandb and "WANDB" in config:
                    loss_log_dict = {"steps":global_step}
                    loss_log_dict.update(model.loss_details)
                    wandb.log(loss_log_dict)
                if not config.OPTIMIZE.USE_EPOCH:
                    # 记录训练日志
                    logger.info(f'Train step: {global_step}, lr: {round(scheduler.get_lr()[0], 6) : .6f}, (loss) ' + str(model.avg_meters))
                    if not opt.no_val and global_step % config.MISC.VAL_FREQ == 0:
                        model.eval()
                        valid_loss_details = model.valid(val_dataloader)
                        valid_metrics = model.evaluate(val_dataloader, epoch, writer, logger, data_name='val')
                        train_metrics = model.evaluate(train_dataloader, epoch, writer, logger, data_name='train')
                        val_loss = valid_loss_details["val/loss"] 
                        metric_dict = {"steps":global_step}
                        metric_dict.update(valid_loss_details)
                        metric_dict.update(valid_metrics)
                        metric_dict.update(train_metrics)
                        if config.DATA.DATASET == ["Lesion4K"]:
                            val_AP50_all = valid_metrics["val/AP50"]
                            lesion_index = [0,1,3,4,5,7,8,10,11,13,14]
                            val_AP50 = sum([valid_metrics["val/AP50_EachClass"][_] for _ in lesion_index])/11
                            metric_dict.update({"val/AP50(11)":val_AP50})
                        else:
                            val_AP50 = valid_metrics["val/AP50"]
                        if is_first_gpu() and opt.start_wandb and "WANDB" in config:
                            wandb.log(metric_dict)
                        model.train()
    
                    if global_step % config.MISC.SAVE_FREQ == 0 or global_step == total_steps:  # 最后一个epoch要保存一下
                        model.save(checkpoint_dir, global_step)
                    if val_AP50 > best_AP50:
                        best_AP50 = val_AP50
                        model.save(checkpoint_dir, "best_AP50")
                    if  val_loss < best_loss:
                        best_loss = val_loss
                        model.save(checkpoint_dir, "best_loss")


        if is_first_gpu() and config.OPTIMIZE.USE_EPOCH:
            # 记录训练日志
            logger.info(f'Train epoch: {epoch}, lr: {round(scheduler.get_lr()[0], 6) : .6f}, (loss) ' + str(model.avg_meters))
            if not opt.no_val and epoch % config.MISC.VAL_FREQ == 0:
                model.eval()
                
                valid_loss_details = model.valid(val_dataloader)
                valid_metrics = model.evaluate(val_dataloader, epoch, writer, logger, data_name='val')
                train_metrics = model.evaluate(train_dataloader, epoch, writer, logger, data_name='train')
                val_loss = valid_loss_details["val/loss"]
                metric_dict = {"steps":global_step}
                metric_dict.update(valid_loss_details)
                metric_dict.update(valid_metrics)
                metric_dict.update(train_metrics)
                if config.DATA.DATASET == ["Lesion4K"]:
                    val_AP50_all = valid_metrics["val/AP50"]
                    lesion_index = [0,1,3,4,5,7,8,10,11,13,14]
                    val_AP50 = sum([valid_metrics["val/AP50_EachClass"][_] for _ in lesion_index])/11
                    metric_dict.update({"val/AP50(11)":val_AP50})
                else:
                    val_AP50 = valid_metrics["val/AP50"]
                if is_first_gpu() and opt.start_wandb and "WANDB" in config:
                    wandb.log(metric_dict)
                model.train()
    
            if epoch % config.MISC.SAVE_FREQ == 0 or epoch == config.OPTIMIZE.EPOCHS:  # 最后一个epoch要保存一下
                model.save(checkpoint_dir, epoch)
            if val_AP50 > best_AP50:
                best_AP50 = val_AP50
                model.save(checkpoint_dir, "best_AP50")
            if  val_loss < best_loss:
                best_loss = val_loss
                model.save(checkpoint_dir, "best_loss")


        if scheduler is not None:
            scheduler.step()

        #if is_distributed():
        #    dist.barrier()

    # 保存结束信息
    #if is_first_gpu():
    #    if opt.tag != 'default':
    #        with open('run_log.txt', 'a') as f:
    #            f.writelines('    Accuracy:' + eval_result + '\n')

    #    meta = load_meta()
    #    meta[-1]['finishtime'] = utils.get_time_stamp()
    #    save_meta(meta)

    if is_distributed():
        dist.destroy_process_group()
    if is_first_gpu() and opt.start_wandb and "WANDB" in config:
        wandb.finish()
"""
except Exception as e:
    
    if is_first_gpu():
        if opt.tag != 'default':
            with open('run_log.txt', 'a') as f:
                f.writelines('    Error: ' + str(e)[:120] + '\n')

        meta = load_meta()
        meta[-1]['finishtime'] = utils.get_time_stamp()
        save_meta(meta)
        # print(e)
    raise Exception('Error')  # 再引起一个异常，这样才能打印之前的trace back信息

except:  # 其他异常，如键盘中断等
    if is_first_gpu():
        meta = load_meta()
        meta[-1]['finishtime'] = utils.get_time_stamp()
        save_meta(meta)
"""
