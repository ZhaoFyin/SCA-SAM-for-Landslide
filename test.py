import warnings

warnings.filterwarnings("ignore")
import os
from compare_model import *
from MySamModel import ScaSAM
from torch.utils.data import ConcatDataset, DataLoader
import glob
import torch
import re
from tqdm import tqdm
import torch.nn as nn
from tools.metric import Evaluator
import numpy as np
import cv2
from matplotlib.patches import Patch
import matplotlib.pyplot as plt
from PIL import Image
import torch.nn.functional as F


def get_epoch(p):
    m = re.search(r'epoch_(\d+)_', p)
    return int(m.group(1)) if m else -1


def visualize_prediction(img_tensor, pred_tensor, gt_tensor, output_dir, file_name, CLASSES):
    """
    img_tensor: (3, H, W)
    pred_tensor: (H, W) - 预测结果
    gt_tensor: (H, W) - 真实标签（用于提取255边界）
    """
    PALETTE = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
        (255, 0, 255), (0, 255, 255), (128, 0, 0), (0, 128, 0),
        (0, 0, 128), (128, 128, 0), (128, 0, 128), (0, 128, 128)
    ]

    # 1. 反归一化 (根据实际使用的 transform 修改 mean/std)
    mean = np.array([0.485, 0.456, 0.406]).reshape(1, 1, 3)
    std = np.array([0.229, 0.224, 0.225]).reshape(1, 1, 3)

    img = img_tensor.transpose(1, 2, 0)  # (H, W, 3)
    img = img * std + mean
    img = np.clip(img * 255, 0, 255).astype(np.uint8)
    img = np.ascontiguousarray(img)  # 防止 OpenCV 报错

    pred = pred_tensor.astype(np.uint8)
    gt = gt_tensor.astype(np.uint8)

    overlay = img.copy()
    legend_elements = []

    # 获取当前预测中包含的类别
    unique_classes = np.unique(pred)

    # --- A. 绘制预测类别 (Mask + 边界) ---
    for cls_idx in unique_classes:
        if cls_idx >= len(CLASSES): continue  # 忽略非法索引

        color = PALETTE[int(cls_idx)]
        class_name = CLASSES[int(cls_idx)]

        # 掩码
        mask = (pred == cls_idx).astype(np.uint8)

        # 半透明填充
        # 创建彩色层
        color_layer = np.zeros_like(img, dtype=np.uint8)
        color_layer[mask == 1] = color

        # 仅在mask区域混合颜色
        # 逻辑：img[mask] = 0.6*img[mask] + 0.4*color
        roi = overlay[mask == 1]
        blended = cv2.addWeighted(roi, 0.6, color_layer[mask == 1], 0.4, 0)
        overlay[mask == 1] = blended

        # 加粗边界
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, color, thickness=2)

        # 添加到 Legend
        legend_elements.append(Patch(facecolor=np.array(color) / 255,
                                     edgecolor=np.array(color) / 255,
                                     label=f"{class_name}"))

    # --- B. 绘制 Ignore/Boundary (255) ---
    # 通常预测结果里没有255，但GT里有。我们将GT的255覆盖在预测之上，显示为黑色
    if 255 in gt:
        mask_255 = (gt == 255).astype(np.uint8)

        # 将255区域涂黑 (完全覆盖)
        overlay[mask_255 == 1] = [0, 0, 0]

        # 画白色边框区分黑色区域 (可选)
        contours_255, _ = cv2.findContours(mask_255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours_255, -1, (255, 255, 255), thickness=1)

        # 添加 Legend
        legend_elements.append(Patch(facecolor='black', edgecolor='white', label='Boundary (255)'))

    # --- C. 使用 Matplotlib 保存 ---
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(overlay)

    # Legend 放在右上角外部
    if len(legend_elements) > 0:
        ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.01, 1), borderaxespad=0.)

    ax.axis('off')
    plt.tight_layout()

    save_path = os.path.join(output_dir, f"{file_name}.png")
    plt.savefig(save_path, bbox_inches='tight', dpi=100)
    plt.close(fig)


def save_result(rgb, pred, gt, result_dir, name, mode, CLASSES):
    if mode == "mask":
        if 255 in gt:
            mask_255 = (gt == 255).astype(np.uint8)
            pred[mask_255 == 1] = 255

        img = Image.fromarray(pred)
        img.save(os.path.join(result_dir, f"{name}.png"))
    elif mode == "overlap":
        visualize_prediction(rgb, pred, gt, result_dir, name, CLASSES)
    else:
        return


def return_model(model_name, num_classes):
    sca_cfg = {
        "enable": True,
        "sca_enable": True,
        "r": 8, "alpha": 16, "dropout": 0.0,
        "target_modules": ["qkv", "proj"],
        "target_blocks": "indices",
        "indices": [8, 9, 10, 11],
    }

    frz_cfg = {
        "enable": False,
    }
    inp_size = 512
    model_dict = {
        "UNetFormer": UNetFormer(num_classes=num_classes),
        "SegFormer": SegFormer(num_classes=num_classes),
        "DCSwin": dcswin_base(num_classes=num_classes),
        "SAM_Frozen": ScaSAM(frz_cfg, num_classes=num_classes, sca=False, img_s=inp_size),
        "EfficientViTSAM": create_efficientvit_sam_model("efficientvit-sam-l0", num_cls=num_classes),
        "SAM_LST": SAM_LST(num_classes=num_classes),
        "SCA_SAM": ScaSAM(sca_cfg, num_classes=num_classes, sca=True, img_s=inp_size)}
    return model_dict[model_name]


def main(dataset_name, model_name, save_mode):
    assert save_mode in [None, "mask", "overlap"]
    class_dict = {"BJL": ["no_landslide", "landslide"],
                  "YYL": ["no_landslide", "landslide"],}

    data_dict = {'YYL': r"C:\Users\48188\Data\VOCdevkit_YYL",
                 'BJL': r"C:\Users\48188\Data\VOCdevkit_BJL"}

    CLASSES = class_dict[dataset_name]
    num_classes = len(CLASSES)

    trained_dir_dict = {
        "UNetFormer": f"./compare_result/{dataset_name}/{model_name}",
        "SegFormer": f"./compare_result/{dataset_name}/{model_name}",
        "DCSwin": f"./compare_result/{dataset_name}/{model_name}",
        "SAM_Frozen": f"./results/{dataset_name}/{model_name}",
        "EfficientViTSAM": f"./compare_result/{dataset_name}/{model_name}",
        "SAM_LST": f"./compare_result/{dataset_name}/{model_name}",
        "SCA_SAM": f"./results/{dataset_name}/{model_name}"}


    if dataset_name in ['YYL', 'BJL']:
        from dataset.landslide_dataset import LandslideDataset
        test_dataset = LandslideDataset(voc_root=data_dict[dataset_name], txt_name="test.txt")
        # train_dataset = LandslideDataset(voc_root=data_dict[dataset_name], txt_name="train.txt")
        # val_dataset = LandslideDataset(voc_root=data_dict[dataset_name], txt_name="val.txt")
        # test_dataset = ConcatDataset([train_dataset, val_dataset, test_dataset])
    else:
        raise ValueError("Dataset not supported")

    val_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)

    model = return_model(model_name, num_classes).cuda()
    weight_list = glob.glob(os.path.join(trained_dir_dict[model_name], "*.pth"))
    best = max(weight_list, key=get_epoch)
    model.load_state_dict(torch.load(best), strict=True)

    save_dir = f"./vis_results/{dataset_name}/{model_name}"
    # save_dir = r"G:\KGE-SCA-SAM\ScaPred/"

    os.makedirs(save_dir, exist_ok=True)

    model.eval()
    t = tqdm(val_loader, desc=f"val...", leave=False, dynamic_ncols=True)
    metrics_val = Evaluator(num_class=num_classes)

    with torch.no_grad():
        for sampled_batch in t:
            image_batch, label_batch = sampled_batch['img'], sampled_batch['gt_semantic_seg']
            image_batch, label_batch = image_batch.cuda(), label_batch.cuda()
            if model_name in ["SAM_LST", "SCA_SAM", "SAM_Frozen"]:
                outputs = model(image_batch, True, image_batch.shape[-1])
                pre_mask = nn.Softmax(dim=1)(outputs["masks"])
            elif model_name == "EfficientViTSAM":
                outputs = model([{"image": image_batch[i]} for i in range(image_batch.shape[0])], True)[0].squeeze(0)
                outputs = F.interpolate(outputs, size=image_batch.shape[-2:], mode='bilinear', align_corners=True)
                pre_mask = nn.Softmax(dim=1)(outputs)
            elif model_name == "SamAdapter":
                outputs = model(image_batch, image_batch.shape[-1])
                pre_mask = nn.Softmax(dim=1)(outputs)
            else:
                outputs = model(image_batch)
                pre_mask = nn.Softmax(dim=1)(outputs)

            pre_mask = pre_mask.argmax(dim=1)

            for i in range(label_batch.shape[0]):
                l = label_batch[i].squeeze(0).cpu().numpy().astype(np.uint8)
                p = pre_mask[i].cpu().numpy().astype(np.uint8)
                metrics_val.add_batch(l, p)
                save_result(rgb=image_batch[i].cpu().numpy(), pred=p, gt=l,
                            result_dir=save_dir, name=sampled_batch['img_id'][i], mode=save_mode, CLASSES=CLASSES)

    if dataset_name in ["vaihingen", "potsdam"]:
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
    for class_name, iou in zip(CLASSES, val_iou_per_class):
        eval_value[class_name] = iou
    print(f"{model_name} on {dataset_name} dataset")
    print("\t".join(map(str, eval_value.keys())))
    print("\t".join(f"{v * 100:.2f}" for v in eval_value.values()))


if __name__ == '__main__':
    for dataset_name in ["YYL"]:
        for model_name in ["SCA_SAM"]:
            save_mode = "mask"
            main(dataset_name, model_name, save_mode)


r"""
D:\Anaconda\envs\mss\python.exe G:\My_SAM_forL\test.py 
UNetFormer on YYL dataset
mIoU	F1	OA	no_landslide	landslide
81.32	88.91	96.79	96.57	66.07
SegFormer on YYL dataset
mIoU	F1	OA	no_landslide	landslide
83.73	90.57	97.24	97.04	70.41
DCSwin on YYL dataset
mIoU	F1	OA	no_landslide	landslide
80.09	88.03	96.61	96.40	63.78
SAM_Frozen on YYL dataset
mIoU	F1	OA	no_landslide	landslide
81.73	89.19	96.91	96.71	66.75
EfficientViTSAM on YYL dataset
mIoU	F1	OA	no_landslide	landslide
78.74	87.09	95.95	95.67	61.82
SAM_LST on YYL dataset
mIoU	F1	OA	no_landslide	landslide
81.28	88.91	96.54	96.28	66.28
SCA_SAM on YYL dataset
mIoU	F1	OA	no_landslide	landslide
87.35	92.92	97.90	97.74	76.96
UNetFormer on BJL dataset
mIoU	F1	OA	no_landslide	landslide
85.48	91.82	96.51	96.11	74.85
SegFormer on BJL dataset
mIoU	F1	OA	no_landslide	landslide
86.01	92.14	96.81	96.46	75.55
DCSwin on BJL dataset
mIoU	F1	OA	no_landslide	landslide
84.97	91.49	96.45	96.05	73.89
SAM_Frozen on BJL dataset
mIoU	F1	OA	no_landslide	landslide
84.84	91.39	96.47	96.08	73.59
EfficientViTSAM on BJL dataset
mIoU	F1	OA	no_landslide	landslide
81.59	89.24	95.64	95.19	67.98
SAM_LST on BJL dataset
mIoU	F1	OA	no_landslide	landslide
82.25	89.69	95.79	95.35	69.15
SCA_SAM on BJL dataset
mIoU	F1	OA	no_landslide	landslide
88.60	93.74	97.38	97.07	80.13

进程已结束，退出代码为 0
"""