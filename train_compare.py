import warnings
warnings.filterwarnings("ignore")

import multiprocessing
import argparse
from SAM.build_sam import sam_model_registry
import random
import datetime
import torch
import numpy as np
import os
import yaml
from utils_compare import trainer_synapse, Tee
from torch.utils.data import DataLoader
from compare_model import *

torch.backends.cudnn.enable =True
torch.backends.cudnn.benchmark = True


with open("./config.yaml", 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)


dataset_name = config['dataset_name']
save_dir = config['save_dir']

class_dict = {"BJL": ['no_landslide', 'landslide'],
              "YYL": ['no_landslide', 'landslide'],
              "uavid": ['Building', 'Road', 'Tree', 'LowVeg', 'Moving_Car', 'Static_Car', 'Human', 'Clutter'],
              "vaihingen": ['ImSurf', 'Building', 'LowVeg', 'Tree', 'Car', 'Clutter'],
              "potsdam": ['ImSurf', 'Building', 'LowVeg', 'Tree', 'Car', 'Clutter']} 

data_dict = {
    'uavid': config['uavid_dir'],
    'vaihingen': config['vaihingen_dir'],
    'potsdam': config['potsdam_dir'],
    "BJL" : config['bjl_dir'],
    "YYL" : config['yyl_dir'],
}

CLASSES = class_dict[dataset_name]


def main(args, snapshot_path):
    args.is_pretrain = True
    if args.module == "UNetFormer":
        net = UNetFormer(num_classes=args.num_classes).cuda()
    elif args.module == "SegFormer":
        net = SegFormer(num_classes=args.num_classes).cuda()
        net_weight = torch.load('./cp/segformer_b2_weights_voc.pth')
        for k, v in list(net_weight.items()):
            if "decode_head.linear_pred" in k:
                del net_weight[k]
        net.load_state_dict(net_weight, strict=False)
    elif args.module == "SamLST":
        net = SAM_LST(num_classes=args.num_classes).cuda()
    elif args.module == "EfficientViTSAM":
        net = create_efficientvit_sam_model("efficientvit-sam-l0", num_cls=args.num_classes).cuda()
    elif args.module == "DCSwin":
        net = dcswin_base(num_classes=args.num_classes).cuda()
    # elif args.module == "RS3Mamba":
    #     net = RS3Mamba(num_classes=args.num_classes).cuda()
    #     net = load_pretrained_ckpt(net)
    #     pass
    else:
        raise NotImplementedError

    config_file = os.path.join(snapshot_path, 'config.txt')
    config_items = []
    for key, value in args.__dict__.items():
        config_items.append(f'{key}: {value}\n')

    with open(config_file, 'w') as f:
        f.writelines(config_items)

    if dataset_name in ['YYL', 'BJL']:
        from dataset.landslide_dataset import LandslideDataset

        train_dataset = LandslideDataset(voc_root=data_dict[args.dataset], txt_name="train.txt")
        val_dataset = LandslideDataset(voc_root=data_dict[args.dataset], txt_name="val.txt")
    else:
        raise ValueError("Dataset not supported")

    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=args.batch_size,
                              num_workers=8,
                              pin_memory=True,
                              shuffle=True,
                              drop_last=True)

    val_loader = DataLoader(dataset=val_dataset, batch_size=2, shuffle=False)

    trainer_synapse(args, net, snapshot_path, train_loader, val_loader, CLASSES)


def parameter(network):
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=str, default=save_dir)
    parser.add_argument('--dataset', type=str,
                        default=dataset_name, help='dataset_name')
    parser.add_argument('--experiment', type=str,
                        default='MySAM', help='experiment_name')

    parser.add_argument('--num_classes', type=int,
                        default=len(CLASSES), help='output channel of network')
    parser.add_argument('--max_iterations', type=int,
                        default=30000, help='maximum epoch number to train')
    parser.add_argument('--max_epochs', type=int,
                        default=200, help='maximum epoch number to train')
    parser.add_argument('--batch_size', type=int,
                        default=2, help='batch_size per gpu')
    parser.add_argument('--deterministic', type=int, default=1,
                        help='whether use deterministic training')
    parser.add_argument('--base_lr', type=float, default=0.0004,
                        help='segmentation network learning rate')
    parser.add_argument('--backbone_lr', type=float, default=6e-5,
                        help='segmentation backbone network learning rate')
    parser.add_argument('--backbone_weight_decay', type=float, default=2.5e-4,
                        help='backbone weight decay')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                        help='weight decay')
    parser.add_argument('--img_size', type=int,
                        default=512, help='input patch size of network input')
    parser.add_argument('--input_size', type=int, default=512, help='The input size for training SAM efficientvitsam')
    parser.add_argument('--seed', type=int,
                        default=1234, help='random seed')
    parser.add_argument('--vit_name', type=str,
                        default='vit_b', help='select one vit efficientvitsam')
    parser.add_argument('--warmup', action='store_true', default=True,
                        help='If activated, warp up the learning from a lower lr to the base_lr')
    parser.add_argument('--warmup_period', type=int, default=100,
                        help='Warp up iterations, only valid whrn warmup is activated')
    parser.add_argument('--AdamW', action='store_true', default=True,
                        help='If activated, use AdamW to finetune SAM efficientvitsam')
    parser.add_argument('--module', type=str, default=network)
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    for net in config['model_list']:
        args = parameter(net)
        now = datetime.datetime.now()
        now = now.strftime("%m%d_%H%M%S")
        snapshot_path = os.path.join(args.output, "{}".format(args.dataset), now)
        if not os.path.exists(snapshot_path):
            os.makedirs(snapshot_path)

        with Tee(os.path.join(snapshot_path, "running.txt"), 'w'):
            main(args, snapshot_path)
