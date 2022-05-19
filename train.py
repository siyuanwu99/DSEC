import argparse
import collections
import torch
import numpy as np
import model.loss as module_loss
import model.metric as module_metric
import model.unet as module_arch
from parse_config import ConfigParser
from trainer import Trainer
from utils import prepare_device
from torch.utils.tensorboard import SummaryWriter

from dataset.provider import DatasetProvider
from dataset.dataloader import BaseDataLoader
from pathlib import Path

# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)

def main(config,writer_tensbd):
    logger = config.get_logger('train')

    # setup data_loader instances
    dataset_provider = DatasetProvider(Path(config['dsec_dir']))
    train_dataset = dataset_provider.get_train_dataset()

    data_loader = BaseDataLoader(  # could have bugs
        dataset=train_dataset,
        batch_size=config["data_loader"]["args"]["batch_size"],
        shuffle=config["data_loader"]["args"]["shuffle"],
        validation_split=config["data_loader"]["args"]["validation_split"],
        num_workers=config["data_loader"]["args"]["num_workers"],
        drop_last=False,
    )
    valid_data_loader = data_loader.split_validation()

    # build model architecture, then print to console
    model = config.init_obj('arch', module_arch)
    logger.info(model)

    # prepare for (multi-device) GPU training
    device, device_ids = prepare_device(config['n_gpu'])
    # print(device, device_ids)
    model = model.to(device)
    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids)

    # get function handles of loss and metrics
    criterion = getattr(module_loss, config['loss'])
    metrics = [getattr(module_metric, met) for met in config['metrics']]

    # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = config.init_obj('optimizer', torch.optim, trainable_params)
    lr_scheduler = config.init_obj('lr_scheduler', torch.optim.lr_scheduler, optimizer)

    trainer = Trainer(model, criterion, metrics, optimizer,
                      config=config,
                      device=device,
                      data_loader=data_loader,
                      valid_data_loader=valid_data_loader,
                      lr_scheduler=lr_scheduler,
                      writer_tensbd=writer_tensbd)

    trainer.train()


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default="./config.json", type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--lr', '--learning_rate'], type=float, target='optimizer;args;lr'),
        CustomArgs(['--bs', '--batch_size'], type=int, target='data_loader;args;batch_size')
    ]
    tb_writer = SummaryWriter()
    config = ConfigParser.from_args(args, options)
    main(config,tb_writer)
