import os
import PIL.Image as Image
import numpy as np
import matplotlib.pyplot as plt
import cv2
from matplotlib.patches import Patch

data_name = 'BJL'

draw_ids = {'BJL': [3, 11, 15, 21, 24, 25],
            'YYL': [7, 12, 17, 20, 22, 25]}

draw_ids = draw_ids[data_name]

data_dir = {'YYL': r"C:\Users\48188\Data\VOCdevkit_YYL",
            'BJL': r"C:\Users\48188\Data\VOCdevkit_BJL"}

classes = ['非滑坡', '滑坡']
PALETTE = [(0,  0,  0), (255, 0,  0)]


def get_draw_data():
    with open(os.path.join(data_dir[data_name], r"VOC_landslide\ImageSets\Segmentation\test.txt"), "r") as f:
        draw_names = [x.strip() for x in f.readlines() if len(x.strip()) > 0]
    draw_names = [draw_names[draw_id] for draw_id in draw_ids]
    model_list = ["UNetFormer", "SegFormer", "DCSwin", "SAM_Frozen", "EfficientViTSAM", "SAM_LST", "SCA_SAM"]

    draw_dict_list = []
    for draw_name in draw_names:
        rgb_img = Image.open(os.path.join(data_dir[data_name], "VOC_landslide\JPEGImages", draw_name + ".jpg"))
        msk_img = Image.open(os.path.join(data_dir[data_name], "VOC_landslide\SegmentationObject", draw_name + ".png"))
        img_resized = rgb_img.resize((512, 512), Image.Resampling.BILINEAR)
        msk_resized = msk_img.resize((512, 512), Image.Resampling.NEAREST)
        rgb_img, msk_img = np.array(img_resized), np.array(msk_resized)
        msk_img[msk_img != 0] = 1
        tmp_dict = {"image": rgb_img, "mask": msk_img}

        for model_name in model_list:
            pred_dir = os.path.join("vis_results", data_name, model_name)
            pred_img = Image.open(os.path.join(pred_dir, draw_name + ".png"))
            pred_img = np.array(pred_img)
            tmp_dict[model_name] = pred_img

        draw_dict_list.append(tmp_dict)
    return draw_dict_list


def crop_square(inp_data: np.ndarray, gt: np.ndarray, ignore_index) -> np.ndarray:
    mask = (gt != ignore_index)
    ys, xs = np.where(mask)  # ys: 行索引, xs: 列索引
    top = int(ys.min())
    bottom = int(ys.max())
    left = int(xs.min())
    right = int(xs.max())

    sq = min(right - left, bottom - top)

    cropped = inp_data[top:top + sq, left:left + sq]
    out = cv2.resize(cropped, (1024, 1024), interpolation=cv2.INTER_NEAREST)
    return out


def overlap(img, pred, gt):
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


def plot_fig(inp_list: list[dict]):
    win_size = 800
    win_gap = 100
    dpi = 500
    h_c = len(inp_list)
    w_c = len(inp_list[0])

    w_px = w_c * (win_size + win_gap) + win_gap
    h_px = h_c * (win_size + win_gap) + 2 * win_gap

    fig = plt.figure(figsize=(w_px / dpi, h_px / dpi), dpi=dpi)

    title_name = ["(a) Image", "(b) Mask", "(c) UNetFormer", "(d) SegFormer", "(e) DCSwin", "(f) SAM_Frozen",
                  "(g) EfficientViTSAM", "(h) SAM_LST", "(i) SCA_SAM"]
    title_to = ["image", "mask", "UNetFormer", "SegFormer", "DCSwin", "SAM_Frozen", "EfficientViTSAM", "SAM_LST",
                "SCA_SAM"]
    assert w_c == len(title_name) == len(title_to)
    for h_i in range(h_c):
        for w_i in range(w_c):
            left = w_i * (win_size + win_gap) + win_gap
            bottom = h_px - (h_i + 1) * (win_size + win_gap)
            ax = fig.add_axes([left / w_px, bottom / h_px, win_size / w_px, win_size / h_px])
            subplot_data = inp_list[h_i][title_to[w_i]].astype("int")
            if w_i != 0:
                subplot_data = overlap(img=inp_list[h_i]["image"], pred=subplot_data, gt=inp_list[h_i]["mask"])
            ignore_index = 255 if data_name == "uavid" else len(classes)
            if np.any(inp_list[h_i]["mask"] == ignore_index):
                subplot_data = crop_square(subplot_data, inp_list[h_i]["mask"], ignore_index)

            if data_name in ["vaihingen"]:
                subplot_data = subplot_data[256:768, 256:768, :]
            ax.imshow(subplot_data)
            ax.set_xticks([])  # 移除x轴刻度
            ax.set_yticks([])  # 移除y轴刻度
            ax.set_aspect('equal')  # 设置每个子图为正方形
            for spine in ax.spines.values():
                spine.set_visible(False)
            if h_i == h_c - 1:
                ax.set_xlabel(title_name[w_i], fontsize=16, fontname="times new roman")

    handles = []
    for i in range(min(len(classes), len(PALETTE))):
        rgb = (np.array(PALETTE[i], dtype=np.float32) / 255.0).tolist()  # [r,g,b]
        handles.append(
            Patch(
                facecolor=(rgb[0], rgb[1], rgb[2], 0.40),
                edgecolor=(rgb[0], rgb[1], rgb[2], 1.00),
                linewidth=2.0,
                label=classes[i]
            )
        )

    # fig.legend(
    #     handles=handles,
    #     loc='lower center',
    #     ncol=len(handles),  # uavid=8 类一行；想两行可改成 4
    #     bbox_to_anchor=(0.5, 0.02),
    #     frameon=False,
    #     prop={'family': 'SimSun', 'size': 16},
    #     handlelength=3,
    #     handleheight=1.2,
    #     columnspacing=3.0,
    #     borderaxespad=0.0
    # )

    plt.savefig(f'fig/FIG_{data_name}.png')
    plt.close()


if __name__ == "__main__":
    plot_fig(get_draw_data())
