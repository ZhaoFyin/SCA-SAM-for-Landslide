from __future__ import annotations

import os
import time

from tools.metric import Evaluator
import numpy as np
import torch.optim as optim
from tensorboardX import SummaryWriter
from MySamModel import ScaSAM
from tqdm import tqdm
from losses import *
import sys
from catalyst import utils
from catalyst.contrib.nn import Lookahead
import pynvml


def get_temp(handle) -> int:
    return int(pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU))


def cooling_down(pbar, stop_temp_c: int = 80, resume_temp_c: int = 60, check_interval_s: float = 1.0) -> None:
    pynvml.nvmlInit()
    h = pynvml.nvmlDeviceGetHandleByIndex(0)
    try:
        t = get_temp(h)
        if t <= stop_temp_c:
            return

        while True:
            pbar.set_postfix_str(f"COOLING GPU0 {t}°C", refresh=True)
            time.sleep(check_interval_s)
            t = get_temp(h)
            if t <= resume_temp_c:
                return
    finally:
        pynvml.nvmlShutdown()


class Tee:
    def __init__(self, filename, mode='w'):
        self.filename = filename
        self.mode = mode
        self.stdout = sys.stdout
        self.file = None

    def __enter__(self):
        self.file = open(self.filename, self.mode)
        sys.stdout = self
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        sys.stdout = self.stdout
        if self.file:
            self.file.close()

    def write(self, data):
        if self.file:
            self.file.write(data)
        self.stdout.write(data)

    def flush(self):
        if self.file:
            self.file.flush()


def val(args, model, dataloader, metrics_val, img_size, CLASSES):
    model.eval()
    t = tqdm(dataloader, desc=f"val...", leave=False, dynamic_ncols=True)
    with torch.no_grad():
        for sampled_batch in t:
            image_batch, label_batch = sampled_batch['img'], sampled_batch['gt_semantic_seg']
            image_batch, label_batch = image_batch.cuda(), label_batch.cuda()

            with torch.no_grad():
                if args.module == "SamLST":
                    outputs = model(image_batch, None, args.input_size)["masks"]
                elif args.module == "SamAdapter":
                    outputs = model(image_batch, args.input_size)
                elif args.module == "EfficientViTSAM":
                    outputs = model([{"image": image_batch[i]} for i in range(image_batch.shape[0])], True)[0].squeeze(1)
                    outputs = F.interpolate(outputs, size=image_batch.shape[-2:], mode='bilinear', align_corners=True)
                else:
                    outputs = model(image_batch)
            pre_mask = nn.Softmax(dim=1)(outputs)
            pre_mask = pre_mask.argmax(dim=1)

            for i in range(label_batch.shape[0]):
                metrics_val.add_batch(label_batch[i].squeeze(0).cpu().numpy(), pre_mask[i].cpu().numpy())

    if args.dataset == "vaihingen":
        val_miou = np.nanmean(metrics_val.Intersection_over_Union()[:-1])
        val_f1 = np.nanmean(metrics_val.F1()[:-1])
        val_oa = np.nanmean(metrics_val.OA())
    elif args.dataset == 'potsdam':
        val_miou = np.nanmean(metrics_val.Intersection_over_Union()[:-1])
        val_f1 = np.nanmean(metrics_val.F1()[:-1])
        val_oa = np.nanmean(metrics_val.OA())
    else:
        val_miou = np.nanmean(metrics_val.Intersection_over_Union())
        val_f1 = np.nanmean(metrics_val.F1())
        val_oa = np.nanmean(metrics_val.OA())
    t.clear()
    t.close()
    val_iou_per_class = metrics_val.Intersection_over_Union()
    eval_value = {'mIoU': val_miou,
                  'F1': val_f1,
                  'OA': val_oa}
    print('\t val:', eval_value)
    iou_value = {}
    for class_name, iou in zip(CLASSES, val_iou_per_class):
        iou_value[class_name] = iou
    print('\t ' + str(iou_value))
    metrics_val.reset()
    return val_miou, val_f1, val_oa


def trainer_synapse(args, model, snapshot_path, trainloader, valloader, CLASSES):
    model.train()

    if args.AdamW:
        # layerwise_params = {"backbone.*": dict(lr=args.backbone_lr, weight_decay=args.backbone_weight_decay)}
        # net_params = utils.process_model_params(efficientvitsam, layerwise_params=layerwise_params)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.base_lr, weight_decay=args.weight_decay)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_epochs, eta_min=1e-6)
    else:
        # RS3Mamba
        optimizer = optim.SGD(model.parameters(), lr=args.base_lr, momentum=0.9, weight_decay=0.0005)
        lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [25, 35, 45], gamma=0.1)

    iter_num = 0
    max_epoch = args.max_epochs
    max_iterations = args.max_epochs * len(trainloader)  # max_epoch = max_iterations // len(trainloader) + 1
    print("{} iterations per epoch. {} max iterations ".format(len(trainloader), max_iterations))
    best_miou = 0.0
    ignore_index = args.num_classes if args.dataset != "uavid" else 255
    criterion = UnetFormerLoss(num_c=args.num_classes, ignore_index=ignore_index)
    #
    # class_pixel_counts = {"uavid": [482, 213, 431, 232, 16, 19, 1, 300],
    #                       "vaihingen": [1681, 1588, 1210, 1340, 65, 35],
    #                       "potsdam": [1111, 1212, 1048, 578, 76, 293]}
    #
    # criterion = SAMAwareLoss(class_pixel_counts=class_pixel_counts[args.dataset],
    #                          gamma=2.0,
    #                          alpha=0.3, beta=0.7, ft_gamma=1.5,
    #                          k_top=0.2,
    #                          ignore_index=ignore_index,
    #                          w_ce=0.5, w_tversky=0.3, w_topk=0.2).cuda()
    metrics_train = Evaluator(num_class=args.num_classes)
    metrics_val = Evaluator(num_class=args.num_classes)
    # model.load_state_dict(torch.load("compare_result/uavid/0112_095801/epoch_64_6312.pth"))
    for epoch_num in range(max_epoch):
        model.train()
        epoch_loss = 0.0

        # if epoch_num <= 68:
        #     for _ in range(len(trainloader)):
        #         iter_num = iter_num + 1
        #     print(f"Epoch {epoch_num + 1}/{max_epoch} Skip!")
        #     lr_scheduler.step()
        #     continue

        pbar = tqdm(enumerate(trainloader), total=len(trainloader), ncols=100,
                    desc=f"Epoch [{epoch_num + 1}/{max_epoch}]", leave=False)

        for iter_idx, sampled_batch in pbar:

            image_batch, label_batch = sampled_batch['img'], sampled_batch['gt_semantic_seg']

            image_batch, label_batch = image_batch.cuda(), label_batch.cuda()

            if args.module == "SamLST":
                outputs = model(image_batch, None, args.img_size)
            elif args.module == "EfficientViTSAM":
                outputs = model([{"image": image_batch[i]} for i in range(image_batch.shape[0])], True)[0].squeeze()
                outputs = F.interpolate(outputs, size=image_batch.shape[-2:], mode='bilinear', align_corners=True)
            elif args.module == "SamAdapter":
                outputs = model(image_batch, args.img_size)
            else:
                outputs = model(image_batch)

            if args.module in ["UNetFormer"]:
                loss = criterion(outputs, label_batch)
                pre_mask = nn.Softmax(dim=1)(outputs[0])
                pre_mask = pre_mask.argmax(dim=1)
            elif args.module in ["SegFormer", "SamAdapter", "DCSwin", "HQSAM", "EfficientViTSAM"]:
                loss = criterion(outputs, label_batch)
                pre_mask = nn.Softmax(dim=1)(outputs)
                pre_mask = pre_mask.argmax(dim=1)
            elif args.module == "SamLST":
                loss = criterion(outputs["masks"], label_batch)
                pre_mask = nn.Softmax(dim=1)(outputs["masks"])
                pre_mask = pre_mask.argmax(dim=1)

            else:
                raise NotImplementedError

            epoch_loss += loss.item()
            for i in range(label_batch.shape[0]):
                metrics_train.add_batch(label_batch[i].squeeze(0).cpu().numpy(), pre_mask[i].cpu().numpy())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pbar.set_postfix({
                "it": f"{iter_idx + 1}/{len(trainloader)}",
                "ls": f"{loss.item():.4f}",
            })

            iter_num = iter_num + 1

        lr_scheduler.step()
        pbar.clear()
        pbar.close()

        if args.dataset == "vaihingen":
            train_miou = np.nanmean(metrics_train.Intersection_over_Union()[:-1])
            train_f1 = np.nanmean(metrics_train.F1()[:-1])
            train_oa = np.nanmean(metrics_train.OA())
        elif args.dataset == 'potsdam':
            train_miou = np.nanmean(metrics_train.Intersection_over_Union()[:-1])
            train_f1 = np.nanmean(metrics_train.F1()[:-1])
            train_oa = np.nanmean(metrics_train.OA())
        else:
            train_miou = np.nanmean(metrics_train.Intersection_over_Union())
            train_f1 = np.nanmean(metrics_train.F1())
            train_oa = np.nanmean(metrics_train.OA())

        epoch_loss /= len(trainloader)
        print(f"Epoch {epoch_num + 1}/{max_epoch} finished — Avg Loss: {epoch_loss:.4f}")
        train_iou_per_class = metrics_train.Intersection_over_Union()
        eval_value = {'mIoU': train_miou,
                      'F1': train_f1,
                      'OA': train_oa}
        print(f"\t Train metrics: {eval_value}")
        iou_value = {}
        for class_name, iou in zip(CLASSES, train_iou_per_class):
            iou_value[class_name] = iou
        print('\t ' + str(iou_value))
        metrics_train.reset()

        val_miou, val_f1, val_oa = val(args, model, valloader, metrics_val, args.input_size, CLASSES)
        metrics_val.reset()

        # val
        if val_miou > best_miou:
            best_miou = val_miou
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num+1) + "_" + str(int(10000*val_miou)) + '.pth')
            torch.save(model.state_dict(), save_mode_path)
            print("\t save efficientvitsam to {}".format(save_mode_path))

    return "Training Finished!"
