import os
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
import cv2

abl_dir = "vis_results_abl"
classes = ['非滑坡', '滑坡']
PALETTE = [(0,  0,  0), (255, 0,  0)]
data_dir = {'YYL': r"C:\Users\48188\Data\VOCdevkit_YYL",
            'BJL': r"C:\Users\48188\Data\VOCdevkit_BJL"}


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


def main(fig_data):
    win_size = 800
    win_gap = 100
    dpi = 500
    h_c = len(fig_data)
    w_c = 6

    w_px = w_c * (win_size + win_gap) + win_gap
    h_px = h_c * (win_size + win_gap) + 2 * win_gap

    fig = plt.figure(figsize=(w_px / dpi, h_px / dpi), dpi=dpi)

    title_name = ["(a) Image", "(b) Mask", "(c) SCA-SAM", "(d) w/o SCA", "(e) Full Train", "(f) Full Freeze"]
    dir_name = ["SCA_SAM", "WO_SCA", "Full_Train", "Full_Frozen"]

    for h_i, data in enumerate(fig_data):
        with open(os.path.join(data_dir[data["data"]], r"VOC_landslide\ImageSets\Segmentation\test.txt"), "r") as f:
            draw_names = [x.strip() for x in f.readlines() if len(x.strip()) > 0]

        file_name = draw_names[data["id"]]
        rgb_img = Image.open(os.path.join(data_dir[data["data"]], "VOC_landslide\JPEGImages", file_name + ".jpg"))
        msk_img = Image.open(os.path.join(data_dir[data["data"]], "VOC_landslide\SegmentationObject", file_name + ".png"))

        img_resized = rgb_img.resize((512, 512), Image.Resampling.BILINEAR)
        msk_resized = msk_img.resize((512, 512), Image.Resampling.NEAREST)

        rgb_img, msk_img = np.array(img_resized), np.array(msk_resized)
        msk_img[msk_img != 0] = 1
        msk_img_ol = overlap(rgb_img, msk_img, msk_img, PALETTE, classes)
        rgb_img_zoom = rgb_img[data["zoom"][2]:data["zoom"][3], data["zoom"][0]:data["zoom"][1]]
        msk_img_zoom = msk_img_ol[data["zoom"][2]:data["zoom"][3], data["zoom"][0]:data["zoom"][1]]
        subplot_data = [rgb_img, msk_img_zoom]

        for name in dir_name:
            pred_img = os.path.join(abl_dir, data["data"], name, file_name + ".png")
            pred_img = Image.open(pred_img)

            pred_img = np.array(pred_img)
            pred_img_ol = overlap(rgb_img, pred_img, msk_img, PALETTE, classes)
            pred_img_zoom = pred_img_ol[data["zoom"][2]:data["zoom"][3], data["zoom"][0]:data["zoom"][1]]
            subplot_data.append(pred_img_zoom)

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
            # if w_i == 0:
            #     import matplotlib.patches as patches
            #     rect = patches.Rectangle((data["zoom"][0], data["zoom"][2]),
            #                              data["zoom"][3] - data["zoom"][2],
            #                              data["zoom"][1] - data["zoom"][0],
            #                              linewidth=2, edgecolor=[0, 1, 1], facecolor='none')
            #     ax.add_patch(rect)

    plt.savefig(f'fig/Ablation_Zoom_in.png')
    plt.close()


if __name__ == '__main__':

    fig_data = [
                {'data': "BJL",
                 'id': 16,
                 'zoom': [0, 512, 0, 512]},  # 左右上下
                {'data': "BJL",
                 'id': 75,
                 'zoom': [0, 512, 0, 512]},
                {'data': "YYL",
                 'id': 25,
                 'zoom': [0, 512, 0, 512]},
                {'data': "YYL",
                 'id': 54,
                 'zoom': [0, 512, 0, 512]}
                ]
    main(fig_data)
