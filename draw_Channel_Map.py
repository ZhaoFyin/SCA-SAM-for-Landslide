from MySamModel import ScaSAM
import torch
import os
from torch.utils.data import DataLoader
from dataset.landslide_dataset import LandslideDataset
from matplotlib import pyplot as plt
import cv2
import numpy as np
import torch.nn.functional as F
import itertools
from pathlib import Path
from PIL import Image
import matplotlib.patches as patches
import re
from matplotlib.font_manager import FontProperties

pic_id = 27
dataset = "YYL"
zoom = [50, 450, 150, 350]
data_dir = {'YYL': r"C:\Users\48188\Data\VOCdevkit_YYL",
            'BJL': r"C:\Users\48188\Data\VOCdevkit_BJL"}


# 字体：中文宋体 + 英文 Times New Roman
fp_cn = FontProperties(family='SimSun', size=12)
fp_en = FontProperties(family='Times New Roman', size=12)


# 判断：是否包含中文
def has_cn(s: str) -> bool:
    return re.search(r'[\u4e00-\u9fff]', s) is not None


def split_cn_en(s: str):
    """
    简单拆分：把字符串按“是否中文字符”切成若干段
    每段返回 (text, is_cn)
    """
    parts = []
    if not s:
        return parts
    buf = s[0]
    cur_is_cn = has_cn(s[0])
    for ch in s[1:]:
        is_cn = has_cn(ch)
        if is_cn == cur_is_cn:
            buf += ch
        else:
            parts.append((buf, cur_is_cn))
            buf = ch
            cur_is_cn = is_cn
    parts.append((buf, cur_is_cn))
    return parts


def set_xlabel_mixed(ax, text, fontsize=12, y=-0.06):
    """
    在 ax 底部居中位置绘制混合字体 xlabel（中文宋体，英文新罗马）
    y 为相对坐标（transform=ax.transAxes），可按你排版微调
    """
    # 先清掉默认 xlabel
    ax.set_xlabel("")

    # 先用 invisible text 测每段宽度（以像素为单位）
    fig = ax.figure
    fig.canvas.draw()  # 确保 renderer 可用
    renderer = fig.canvas.get_renderer()

    parts = split_cn_en(text)

    # 计算总宽度
    widths = []
    for seg, is_cn in parts:
        t = ax.text(0, 0, seg,
                    fontproperties=(fp_cn if is_cn else fp_en),
                    fontsize=fontsize, alpha=0.0,
                    transform=ax.transAxes)
        bb = t.get_window_extent(renderer=renderer)
        widths.append(bb.width)
        t.remove()

    total_w = sum(widths)

    # 轴坐标系 0~1 对应的像素宽度
    ax_bb = ax.get_window_extent(renderer=renderer)
    ax_w = ax_bb.width

    # 起始 x，使整体居中
    x = 0.5 - (total_w / ax_w) / 2.0

    # 逐段绘制
    for (seg, is_cn), w in zip(parts, widths):
        ax.text(x, y, seg,
                fontproperties=(fp_cn if is_cn else fp_en),
                fontsize=fontsize,
                transform=ax.transAxes,
                ha='left', va='top')
        x += (w / ax_w)


def overlap(img, pred, gt, PALETTE, classes):

    pred = pred.astype(np.uint8)
    gt = gt.astype(np.uint8)

    overlay = img.copy()

    unique_classes = np.unique(pred)

    for cls_idx in unique_classes:
        if cls_idx >= len(classes): continue  # 忽略非法索引
        color = PALETTE[int(cls_idx)]
        mask = (pred == cls_idx).astype(np.uint8)
        color_layer = np.zeros_like(img, dtype=np.uint8)
        color_layer[mask == 1] = color

        # 仅在mask区域混合颜色
        # 逻辑：img[mask] = 0.6*img[mask] + 0.4*color
        roi = overlay[mask == 1]
        blended = cv2.addWeighted(roi, 0.3, color_layer[mask == 1], 0.7, 0)
        overlay[mask == 1] = blended

        # 加粗边界
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, color, thickness=2)

    if 255 in gt:
        mask_255 = (gt == 255).astype(np.uint8)

        # 将255区域涂黑 (完全覆盖)
        overlay[mask_255 == 1] = [0, 0, 0]

        # 画白色边框区分黑色区域 (可选)
        contours_255, _ = cv2.findContours(mask_255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours_255, -1, (255, 255, 255), thickness=1)
    return overlay


def main():
    lora_cfg = {
        "enable": True,
        "sca_enable": True,
        "r": 8, "alpha": 16, "dropout": 0.0,
        "target_modules": ["qkv", "proj"],
        "target_blocks": "indices",
        "indices": [8, 9, 10, 11],
        "lr_rate": 1
    }
    sca_net = ScaSAM(lora_cfg, num_classes=2, sca=True, img_s=512)
    sca_net.load_state_dict(torch.load("results/YYL/SCA_SAM/epoch_140_8444.pth", map_location="cpu", weights_only=True))
    # sca_net.load_state_dict(torch.load("results/BJL/SCA_SAM/epoch_13_8996.pth", map_location="cpu", weights_only=True))
    sca_net.cuda()
    sca_net.eval()

    lora_cfg = {
        "enable": False,
        "sca_enable": True,
        "r": 8, "alpha": 16, "dropout": 0.0,
        "target_modules": ["qkv", "proj"],
        "target_blocks": "indices",
        "indices": [8, 9, 10, 11],
        "lr_rate": 1
    }
    non_net = ScaSAM(lora_cfg, num_classes=2, sca=False, img_s=512)
    # non_net.load_state_dict(torch.load("results/uavid/SAM_Frozen/epoch_189_6510.pth", map_location="cpu", weights_only=True))
    # non_net.load_state_dict(torch.load("results/YYL/WO_SCA/epoch_52_8229.pth"))
    # non_net.load_state_dict(torch.load("results/BJL/WO_SCA/epoch_91_8955.pth"))
    non_net.load_state_dict(torch.load("results/YYL/SAM_Frozen/epoch_25_7739.pth"))

    non_net.cuda()
    non_net.eval()
    val_dataset = LandslideDataset(voc_root=data_dir[dataset], txt_name="test.txt")

    val_loader = DataLoader(dataset=val_dataset, batch_size=1, shuffle=False)

    PALETTE = [(0,  0,  0), (255, 0,  0)]
    classes = ['非滑坡', '滑坡']
    with torch.no_grad():
        sampled_batch = next(itertools.islice(val_loader, pic_id, None))
        image_batch = sampled_batch['img'].cuda()

        _, sca_outputs = sca_net.net.image_encoder(image_batch)
        _, non_outputs = non_net.net.image_encoder(image_batch)

    sca_outputs = [F.interpolate(sca_output.permute(0, 3, 1, 2).contiguous(), (512, 512), mode="bilinear") for sca_output in sca_outputs]
    non_outputs = [F.interpolate(non_output.permute(0, 3, 1, 2).contiguous(), (512, 512), mode="bilinear") for non_output in non_outputs]

    with open(os.path.join(data_dir[dataset], r"VOC_landslide\ImageSets\Segmentation\test.txt"), "r") as f:
        draw_names = [x.strip() for x in f.readlines() if len(x.strip()) > 0]
    file_name = draw_names[pic_id]
    rgb_img = Image.open(os.path.join(data_dir[dataset], "VOC_landslide\JPEGImages", file_name + ".jpg"))
    msk_img = Image.open(os.path.join(data_dir[dataset], "VOC_landslide\SegmentationObject", file_name + ".png"))

    img_resized = rgb_img.resize((512, 512), Image.Resampling.BILINEAR)
    msk_resized = msk_img.resize((512, 512), Image.Resampling.NEAREST)

    rgb_img, msk_img = np.array(img_resized), np.array(msk_resized)
    msk_img[msk_img != 0] = 1

    norm_data = np.array([
        [
            sca_outputs[0][0, 0].detach().cpu().numpy(),
            sca_outputs[1][0, 0].detach().cpu().numpy(),
            sca_outputs[2][0, 0].detach().cpu().numpy(),
            sca_outputs[3][0, 0].detach().cpu().numpy()
        ],
        [
            non_outputs[0][0, 0].detach().cpu().numpy(),
            non_outputs[1][0, 0].detach().cpu().numpy(),
            non_outputs[2][0, 0].detach().cpu().numpy(),
            non_outputs[3][0, 0].detach().cpu().numpy()
        ]
    ])
    # mu = norm_data.mean()
    # std = norm_data.std()
    # lo, hi = mu - 3*std, mu + 3*std
    # norm_data = np.clip(norm_data, lo, hi)

    subplot_data = [
        [
            rgb_img,
            norm_data[0, 0] - norm_data.mean(),
            norm_data[0, 1] - norm_data.mean(),
            norm_data[0, 2] - norm_data.mean(),
            norm_data[0, 3] - norm_data.mean()
        ],
        [
            overlap(rgb_img, msk_img, msk_img, PALETTE, classes),
            norm_data[1, 0] - norm_data.mean(),
            norm_data[1, 1] - norm_data.mean(),
            norm_data[1, 2] - norm_data.mean(),
            norm_data[1, 3] - norm_data.mean()
        ]
    ]

    win_size = 800
    h_gap = 100
    w_gap = 100
    dpi = 500
    h_c = 2
    w_c = 5
    w_px = w_c * (win_size + w_gap) + w_gap
    h_px = h_c * (win_size + 2 * h_gap) + 1 * h_gap

    fig = plt.figure(figsize=(w_px / dpi, h_px / dpi), dpi=dpi)
    title_list = ["(a) RGB Image",
                  "(b) SCA-SAM阶段1特征",
                  "(c) SCA-SAM阶段2特征",
                  "(d) SCA-SAM阶段3特征",
                  "(e) SCA-SAM阶段4特征",
                  "(f) Landslide Mask",
                  "(g) w/o SCA阶段1特征",
                  "(h) w/o SCA阶段2特征",
                  "(i) w/o SCA阶段3特征",
                  "(j) w/o SCA阶段4特征",]
    idx = 0
    for h_index in range(h_c):
        for w_index in range(w_c):
            left = w_index * (win_size + w_gap) + w_gap
            bottom = h_px - (h_index + 1) * (win_size + 2 * h_gap) + h_gap
            ax = fig.add_axes([left / w_px, bottom / h_px, win_size / w_px, win_size / h_px])
            if w_index == 0:
                ax.imshow(subplot_data[h_index][w_index])
            else:
                ax.imshow(-subplot_data[h_index][w_index], vmin=-3*norm_data.std(), vmax = 3*norm_data.std())

            ax.set_xticks([])  # 移除x轴刻度
            ax.set_yticks([])  # 移除y轴刻度
            ax.set_aspect('equal')  # 设置每个子图为正方形
            for spine in ax.spines.values():
                spine.set_visible(False)
            if w_index != 0:
                rect = patches.Rectangle((zoom[0], zoom[2]),
                                         zoom[1] - zoom[0],
                                         zoom[3] - zoom[2],
                                         linewidth=1, edgecolor=[255/255, 111/255, 97/255], facecolor='none')
                ax.add_patch(rect)
            set_xlabel_mixed(ax, title_list[idx], fontsize=12)
            idx += 1

    plt.savefig(f'fig/Channel_Map.png')
    plt.close()

if __name__ == '__main__':
    main()
