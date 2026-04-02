import os
from tools.metric import Evaluator
import gc
import numpy as np
import torch.optim as optim
from tensorboardX import SummaryWriter
from MySamModel import ScaSAM
from tqdm import tqdm
from losses import *
import sys


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


def val(args, model, dataloader, metrics_val, multimask_output, img_size, CLASSES):
    # train_model_w = efficientvitsam.state_dict()
    # test_model_w = lora_merge(train_model_w)
    # test_model = ScaSAM(lora_cfg={"enable": False}, num_classes=args.num_classes, sca=args.sca)
    # test_model.load_state_dict(test_model_w, strict=True)

    # efficientvitsam.cpu()
    # test_model = test_model.cuda()
    model.eval()
    t = tqdm(dataloader, desc=f"val...", leave=False, dynamic_ncols=True)
    with torch.no_grad():
        for sampled_batch in t:
            image_batch, label_batch = sampled_batch['img'], sampled_batch['gt_semantic_seg']
            image_batch, label_batch = image_batch.cuda(), label_batch.cuda()

            with torch.no_grad():
                outputs = model(image_batch, multimask_output, img_size)
            pre_mask = nn.Softmax(dim=1)(outputs['masks'])
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

    # del test_model
    # efficientvitsam.cuda()

    return val_miou, val_f1, val_oa


def trainer_synapse(args, model, snapshot_path, trainloader, valloader, multimask_output, CLASSES):
    model.train()
    base_lr = args.base_lr
    if args.warmup:
        b_lr = base_lr / args.warmup_period
    else:
        b_lr = base_lr

    # ======== 参数分组：LoRA 参数单独设学习率 ========
    lora_params, base_params = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if "A" in name or "B" in name:
            lora_params.append(p)
        else:
            base_params.append(p)

    param_groups = [
        {"params": base_params, "lr": b_lr},  # 正常参数
        {"params": lora_params, "lr": b_lr * args.lora_cfg["lr_rate"]},
    ]

    # ======== 选择优化器 ========
    if args.AdamW:
        optimizer = torch.optim.AdamW(
            param_groups,
            betas=(0.9, 0.999),
            weight_decay=args.weight_decay
        )
    else:
        optimizer = torch.optim.SGD(
            param_groups,
            momentum=0.9,
            weight_decay=0.0001
        )

    iter_num = 0
    max_epoch = args.max_epochs
    max_iterations = args.max_epochs * len(trainloader)  # max_epoch = max_iterations // len(trainloader) + 1
    print("{} iterations per epoch. {} max iterations ".format(len(trainloader), max_iterations))
    best_miou = 0.0
    ignore_index = args.num_classes if args.dataset not in ["uavid", "loveda"] else 255
    if args.dataset not in ["uavid", "loveda"]:
        criterion = UnetFormerLoss(num_c=args.num_classes, ignore_index=ignore_index)
    else:
        class_pixel_counts = {"uavid": [5700, 3681, 5442, 2614, 320, 395, 62, 4011],
                              "vaihingen": [1681, 1588, 1210, 1340, 65, 35],
                              "potsdam": [1111, 1212, 1048, 578, 76, 293],
                              "loveda": [358, 111, 53, 64, 52, 161, 202]}

        criterion = SAMAwareLoss(class_pixel_counts=class_pixel_counts[args.dataset],
                                 gamma=2.0,
                                 alpha=0.3, beta=0.7, ft_gamma=1.5,
                                 k_top=0.2,
                                 ignore_index=ignore_index,
                                 w_ce=0.5, w_tversky=0.3, w_topk=0.2).cuda()
    metrics_train = Evaluator(num_class=args.num_classes)
    metrics_val = Evaluator(num_class=args.num_classes)
    lora_rate = float(args.lora_cfg["lr_rate"])

    # model.load_state_dict(torch.load("results/potsdam/0116_173718/epoch_116_8539.pth"), strict=True)
    for epoch_num in range(max_epoch):
        model.train()
        epoch_loss = 0.0
        # if epoch_num <= 114:
        #     iter_num += len(trainloader)
        #     best_miou = 0.8539
        #     print(f"Epoch {epoch_num + 1}/{max_epoch} Skipped!")
        #     continue

        pbar = tqdm(enumerate(trainloader), total=len(trainloader), ncols=100,
                    desc=f"Epoch [{epoch_num + 1}/{max_epoch}]", leave=False)

        for iter_idx, sampled_batch in pbar:

            image_batch, label_batch = sampled_batch['img'], sampled_batch['gt_semantic_seg']

            image_batch, label_batch = image_batch.cuda(), label_batch.cuda()
            outputs = model(image_batch, multimask_output, args.img_size)

            pre_mask = nn.Softmax(dim=1)(outputs['masks'])
            pre_mask = pre_mask.argmax(dim=1)
            loss = criterion(outputs['masks'], label_batch)
            epoch_loss += loss.item()
            for i in range(label_batch.shape[0]):
                metrics_train.add_batch(label_batch[i].squeeze(0).cpu().numpy(), pre_mask[i].cpu().numpy())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            iter_num = iter_num + 1

            if args.warmup and iter_num < args.warmup_period:
                # 线性 warmup：随迭代从 0 → base_lr
                base_lr_t = args.base_lr * ((iter_num + 1) / args.warmup_period)
            else:
                # Poly（或 Cosine）衰减，这里沿用你原来的 poly^0.9
                if args.warmup:
                    shift_iter = iter_num - args.warmup_period
                    assert shift_iter >= 0, f"Shift iter {shift_iter} < 0"
                    total_after = max(1, max_iterations - args.warmup_period)
                    decay_pos = min(shift_iter, total_after)
                    frac = 1.0 - decay_pos / total_after
                else:
                    decay_pos = min(iter_num, max_iterations)
                    frac = 1.0 - decay_pos / max_iterations
                base_lr_t = args.base_lr * (frac ** 0.9)

            if len(optimizer.param_groups) >= 1:
                optimizer.param_groups[0]["lr"] = base_lr_t

            if len(optimizer.param_groups) >= 2:
                optimizer.param_groups[1]["lr"] = base_lr_t * lora_rate
            lr_base_show = optimizer.param_groups[0]["lr"]
            lr_lora_show = optimizer.param_groups[1]["lr"] if len(optimizer.param_groups) > 1 else -1.0

            pbar.set_postfix({
                "ls": f"{loss.item():.4f}",
                "lr": f"{lr_base_show:.6f}/{lr_lora_show:.6f}"
            })

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

        val_miou, val_f1, val_oa = val(args, model, valloader, metrics_val, multimask_output, args.input_size, CLASSES)
        metrics_val.reset()
        del image_batch, label_batch, outputs, loss, pre_mask

        # val
        if val_miou > best_miou:
            best_miou = val_miou
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num+1) + "_" + str(int(10000*val_miou)) + '.pth')
            torch.save(model.state_dict(), save_mode_path)
            print("\t save efficientvitsam to {}".format(save_mode_path))

    return "Training Finished!"
