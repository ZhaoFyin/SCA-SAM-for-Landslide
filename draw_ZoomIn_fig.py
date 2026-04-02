import os
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
import cv2


fig_data = [{'data': "BJL",
             'id': 15,
             'zoom': [150, 356, 150, 356]},  # 左右上下
            {'data': "YYL",
             'id': 20,
             'zoom': [100, 306, 100, 306]},  # 左右上下
            {'data': "YYL",
             'id': 25,
             'zoom': [200, 406, 200, 406]}]
data_dir = {'YYL': r"C:\Users\48188\Data\VOCdevkit_YYL",
            'BJL': r"C:\Users\48188\Data\VOCdevkit_BJL"}

classes = ['非滑坡', '滑坡']
PALETTE = [(0,  0,  0), (255, 0,  0)]


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
        blended = cv2.addWeighted(roi, 0.6, color_layer[mask == 1], 0.4, 0)
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


win_size = 800
win_gap = 100
dpi = 500
h_c = 3
w_c = 4

w_px = w_c * (win_size + win_gap) + win_gap
h_px = h_c * (win_size + win_gap) + 2 * win_gap

fig = plt.figure(figsize=(w_px / dpi, h_px / dpi), dpi=dpi)
title_name = ["(a) Image", "(b) Zoom-in Image ", "(c) Zoom-in Mask", "(d) Zoom-in Pred"]
for h_i, data in enumerate(fig_data):
    with open(os.path.join(data_dir[data["data"]], r"VOC_landslide\ImageSets\Segmentation\test.txt"), "r") as f:
        draw_names = [x.strip() for x in f.readlines() if len(x.strip()) > 0]

    file_name = draw_names[data["id"]]
    rgb_img = Image.open(os.path.join(data_dir[data["data"]], "VOC_landslide\JPEGImages", file_name + ".jpg"))
    msk_img = Image.open(os.path.join(data_dir[data["data"]], "VOC_landslide\SegmentationObject", file_name + ".png"))

    prd_img = Image.open(os.path.join("vis_results/{}/SCA_SAM".format(data["data"]), file_name + ".png"))
    img_resized = rgb_img.resize((512, 512), Image.Resampling.BILINEAR)
    msk_resized = msk_img.resize((512, 512), Image.Resampling.NEAREST)

    rgb_img, msk_img, prd_img = np.array(img_resized), np.array(msk_resized), np.array(prd_img)
    msk_img[msk_img != 0] = 1
    PALETTE = [(0,  0,  0), (255, 0,  0)]
    classes = ['非滑坡', '滑坡']

    msk_img, prd_img = overlap(rgb_img, msk_img, msk_img, PALETTE, classes), overlap(rgb_img, prd_img, msk_img, PALETTE, classes)
    rgb_img_0 = rgb_img[data["zoom"][2]:data["zoom"][3], data["zoom"][0]:data["zoom"][1]]
    msk_img = msk_img[data["zoom"][2]:data["zoom"][3], data["zoom"][0]:data["zoom"][1]]
    prd_img = prd_img[data["zoom"][2]:data["zoom"][3], data["zoom"][0]:data["zoom"][1]]

    subplot_data = [rgb_img, rgb_img_0, msk_img, prd_img]

    for w_i, subplot in enumerate(subplot_data):
        left = w_i * (win_size + win_gap) + win_gap
        bottom = h_px - (h_i + 1) * (win_size + win_gap)
        ax = fig.add_axes([left / w_px, bottom / h_px, win_size / w_px, win_size / h_px])
        ax.imshow(subplot)
        ax.set_xticks([])  # 移除x轴刻度
        ax.set_yticks([])  # 移除y轴刻度
        ax.set_aspect('equal')  # 设置每个子图为正方形
        for spine in ax.spines.values():
            spine.set_visible(False)
        if h_i == h_c - 1:
            ax.set_xlabel(title_name[w_i], fontsize=16, fontname="times new roman")
        if w_i == 0:
            import matplotlib.patches as patches
            rect = patches.Rectangle((data["zoom"][0], data["zoom"][2]),
                                     data["zoom"][3] - data["zoom"][2],
                                     data["zoom"][1] - data["zoom"][0],
                                     linewidth=2, edgecolor=[0, 1, 1], facecolor='none')
            ax.add_patch(rect)

plt.savefig(f'fig/Zoom_in.png')
plt.close()




