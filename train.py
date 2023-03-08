# -*- coding: utf-8 -*- 
# @Describe : 
# @Time : 2022/07/04 15:15 
# @Author : zpx 
# @File : train.py
import os
import random

import hydra
import numpy as np
import torch
import torch.nn.functional as F
from ignite.contrib.handlers import TensorboardLogger, ProgressBar
from ignite.contrib.metrics import GpuInfo
from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator, convert_tensor
from ignite.handlers import CosineAnnealingScheduler, LRScheduler, EarlyStopping, global_step_from_engine, \
    ModelCheckpoint
from ignite.metrics import Loss, DiceCoefficient, ConfusionMatrix
from torch import nn
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader

from appendix_dataset import Appendix_Dataset_3slice, RandomGenerator
from network import LSSED
from utils import logValidAndTest, HD95Metric, DiceLoss, LSS_LOSS


@hydra.main(config_path='config', config_name='config.yaml')
def run(config):
    os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(config.gpu_num)
    os.environ['PYTHONHASHSEED'] = str(config.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed(config.seed)
    device = torch.device(config.device)
    net = LSSED(pretrained=config.pretrained, num_classes=config.num_classes, patch_size=config.patch_size).to(device)
    logger_path = '{}_{}_{}'.format(config.log_dir, config.model_name, config.gamma)
    tb_logger = TensorboardLogger(log_dir=logger_path)
    # load data
    train_loader = DataLoader(
        Appendix_Dataset_3slice(base_dir=config.root_path, split="train", transform=RandomGenerator(
            output_size=config.img_size, random_convert=True), wl=config.wl, ww=config.ww),
        batch_size=config.batch_size, num_workers=config.num_workers, pin_memory=True, shuffle=True,
        worker_init_fn=lambda worker_id: random.seed(config.seed + worker_id))
    valid_loader = DataLoader(
        Appendix_Dataset_3slice(base_dir=config.root_path, split="valid", transform=RandomGenerator(
            output_size=config.img_size, random_convert=False), wl=config.wl, ww=config.ww),
        batch_size=config.batch_size, num_workers=config.num_workers, pin_memory=True, shuffle=False,
        worker_init_fn=lambda worker_id: random.seed(config.seed + worker_id))
    test_loader = DataLoader(
        Appendix_Dataset_3slice(base_dir=config.root_path, split="test", transform=RandomGenerator(
            output_size=config.img_size, random_convert=False), wl=config.wl, ww=config.ww),
        batch_size=config.batch_size, num_workers=config.num_workers, pin_memory=True, shuffle=False,
        worker_init_fn=lambda worker_id: random.seed(config.seed + worker_id))

    dice_loss = DiceLoss(config.num_classes)
    lgem_loss = LSS_LOSS(H=config.H, Q=config.Q, net=net.seg_head, gamma=config.gamma)
    if config.num_gpus > 1:
        net = nn.DataParallel(net)

    # define optimizer
    if config.optimizer == 'AdamW':
        optimizer = torch.optim.AdamW(net.parameters(), lr=config.lr)
    elif config.optimizer == 'Adam':
        optimizer = torch.optim.Adam(net.parameters(), lr=config.lr)
    elif config.optimizer == 'RMSprop':
        optimizer = torch.optim.RMSprop(net.parameters(), lr=config.lr, weight_decay=1e-4, momentum=0.9)
    else:
        optimizer = torch.optim.SGD(net.parameters(), lr=config.lr, momentum=0.9, weight_decay=1e-4)

    # define learning scheduler
    if config.scheduler == 'cosine':
        scheduler = CosineAnnealingScheduler(optimizer, 'lr', 1e-3, 1e-8, config.epochs)
    elif config.scheduler == 'step_lr':
        scheduler = LRScheduler(StepLR(optimizer, step_size=5, gamma=0.5))

    def prepare_data(sample, device, non_blocking):
        return convert_tensor(sample['image'], device, non_blocking), \
            convert_tensor(sample['label'].long(), device, non_blocking)

    # define loss function
    def loss_fun(p, y):
        return F.cross_entropy(p[0], y) + dice_loss(p[0], y, softmax=True) + lgem_loss((p[1], p[0]), y)

    train_engine = create_supervised_trainer(net, optimizer, loss_fn=loss_fun, device=device,
                                             prepare_batch=prepare_data)
    train_engine.add_event_handler(Events.EPOCH_STARTED, scheduler)

    cm = ConfusionMatrix(config.num_classes, output_transform=lambda x: (x[0][0], x[1]))
    train_metric = {
        "loss_bce": Loss(F.cross_entropy, output_transform=lambda x: (x[0][0], x[1])),
        "loss_dice": Loss(dice_loss, output_transform=lambda x: (x[0][0], x[1])),
        "loss_lgem": Loss(lgem_loss, output_transform=lambda x: ((x[0][1], x[0][0]), x[1])),
        "Dice": DiceCoefficient(cm, ignore_index=0),
        "HD95": HD95Metric(output_transform=lambda x: (x[0][0], x[1]))
    }
    evaluate_metric = {
        "Dice": DiceCoefficient(cm, ignore_index=0),
        "HD95": HD95Metric(output_transform=lambda x: (x[0][0], x[1]))
    }

    train_evaluator = create_supervised_evaluator(net, train_metric, device=device, prepare_batch=prepare_data)
    valid_evaluator = create_supervised_evaluator(net, evaluate_metric, device=device, prepare_batch=prepare_data)
    test_evaluator = create_supervised_evaluator(net, evaluate_metric, device=device, prepare_batch=prepare_data)

    for name, metric in train_metric.items():
        metric.attach(train_evaluator, name)
    for name, metric in evaluate_metric.items():
        metric.attach(valid_evaluator, name)
        metric.attach(test_evaluator, name)

    GpuInfo().attach(train_engine, name='gpu')
    progressBar = ProgressBar()
    progressBar.attach(train_engine, metric_names='all')

    # log setting
    @train_engine.on(Events.EPOCH_COMPLETED(every=1))
    def log_training_result(engine):
        train_evaluator.run(train_loader)
        metrics = train_evaluator.state.metrics
        tb_logger.writer.add_scalar('training/lr', optimizer.state_dict()['param_groups'][0]['lr'], engine.state.epoch)
        logValidAndTest(engine, "training", train_evaluator, train_loader, tb_logger, progressBar)

    @train_engine.on(Events.EPOCH_COMPLETED(every=1))
    def log_valid_result(engine):
        logValidAndTest(engine, "Valid", valid_evaluator, valid_loader, tb_logger, progressBar)

    @train_engine.on(Events.EPOCH_COMPLETED(every=1))
    def log_test_result(engine):
        logValidAndTest(engine, "Test", test_evaluator, test_loader, tb_logger, progressBar)

    # save best model critical
    def score_function(engine):
        return engine.state.metrics['Dice'].item()

    early_stop = EarlyStopping(patience=config.early_stop_patience, score_function=score_function, trainer=train_engine)
    valid_evaluator.add_event_handler(Events.COMPLETED, early_stop)

    for tag, evaluator in [('training', train_evaluator), ('validation', valid_evaluator), ("test", test_evaluator)]:
        tb_logger.attach_output_handler(
            evaluator,
            event_name=Events.EPOCH_COMPLETED,
            tag=tag,
            metric_names="all",
            global_step_transform=global_step_from_engine(train_engine)
        )

    model_checkpoint = ModelCheckpoint(
        'checkpoint',
        n_saved=config.checkpoint_epoch,
        filename_prefix='best',
        score_function=score_function,
        score_name='Dice',
        global_step_transform=global_step_from_engine(train_engine)
    )

    valid_evaluator.add_event_handler(Events.EPOCH_COMPLETED, model_checkpoint, {'model': net})
    train_engine.run(train_loader, config.epochs)
    torch.save('{}/final.pt'.format(logger_path), net.state_dict())
    progressBar.log_message("===================Save final model and finished!!!=======================")


if __name__ == '__main__':
    run()
