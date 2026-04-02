import os
import PIL.Image as Image
import numpy as np
import matplotlib.pyplot as plt
import cv2
import matplotlib.patches as patches
from matplotlib.patches import Patch

data_name = 'uavid'
draw_ids = [{"id":  10, "sign": "遮挡", "zoom_in": [336, 40, 464, 168], "pmt": [40, 10, 110, 50]},
            {"id":  98, "sign": "过曝", "zoom_in": [464, 64, 592, 192], "pmt": [24, 64, 56, 96]},
            {"id": 247, "sign": "欠曝", "zoom_in": [562, 562, 818, 818], "pmt": [64, 32, 192, 192]},
            {"id": 305, "sign": "阴影", "zoom_in": [512, 256, 768, 512], "pmt": [60, 80, 192, 200]},]


data_dir = {'uavid': {"rgb": r"data/uavid_255/val/images",
                      "msk": r"data/uavid_255/val/masks"},
            'vaihingen': {"rgb": r"data/vaihingen/test/part_images_1024",
                          "msk": r"data/vaihingen/test/part_masks_1024"},
            'potsdam': {"rgb": r"data/potsdam/test/part_images_1024",
                        "msk": r"data/potsdam/test/part_masks_1024"}}

class_dict = {"uavid": ['建筑', '道路', '树木', '灌丛', '移动车辆', '静止车辆', '人', '杂物'],
              "vaihingen": ['不透水表面', '建筑', '灌丛', '树木', '车辆', '杂物'],
              "potsdam": ['不透水表面', '建筑', '灌丛', '树木', '车辆', '杂物']}
classes = class_dict[data_name]


def color_map(data_name=data_name):
    color_ = [
        (230, 25,  75), (60, 180,  75), (  0, 130, 200), (245, 130,  48),
        (70, 240, 240), (145, 30, 180), (240,  50, 230), (210, 245,  60)
    ]
    return color_[:8] if data_name == 'uavid' else color_[:6]


PALETTE = color_map(data_name)


def get_draw_data():
    model_name = "SCA_SAM"
    draw_dict_list = []
    for draw_id in draw_ids:
        draw_name = os.listdir(data_dir[data_name]["rgb"])[draw_id["id"]].split(".")[0]
        rgb_img = os.path.join(data_dir[data_name]["rgb"], draw_name + ".tif")
        if os.path.isfile(rgb_img):
            pass
        else:
            rgb_img = os.path.join(data_dir[data_name]["rgb"], draw_name + ".png")
        rgb_img = Image.open(rgb_img)

        msk_img = Image.open(os.path.join(data_dir[data_name]["msk"], draw_name + ".png"))

        rgb_img, msk_img = np.array(rgb_img), np.array(msk_img)
        zi = draw_id["zoom_in"]
        zoom_in_img = rgb_img[zi[0]:zi[2], zi[1]:zi[3]]
        zoom_in_msk = msk_img[zi[0]:zi[2], zi[1]:zi[3]]
        tmp_dict = {"image": rgb_img, "mask": msk_img, "zoom_in_image": zoom_in_img, "zoom_in_mask": zoom_in_msk}

        pred_dir = os.path.join("vis_results", data_name, model_name)
        pred_img = Image.open(os.path.join(pred_dir, draw_name + ".png"))
        pred_img = np.array(pred_img)
        tmp_dict["zoom_in_pred"] = pred_img[zi[0]:zi[2], zi[1]:zi[3]]

        draw_dict_list.append(tmp_dict)
    return draw_dict_list


def crop_square(inp_data: np.ndarray,
                gt: np.ndarray,
                ignore_index: int,
                box: list,
                out_size: int = 1024):
    """
    inp_data: HxW 或 HxWxC
    gt: HxW
    box: [y0, x0, y1, x1]，对应 inp_data[y0:y1, x0:x1]
    return:
        out: resize后的正方形图 (out_size, out_size, ...)
        new_box: resize后的新框 [y0, x0, y1, x1]（在 out 上的坐标）
    """
    mask = (gt != ignore_index)
    ys, xs = np.where(mask)
    if ys.size == 0 or xs.size == 0:
        raise ValueError("gt 中全是 ignore_index，无法裁剪。")

    top = int(ys.min())
    bottom = int(ys.max())
    left = int(xs.min())
    right = int(xs.max())

    # 你原来的 sq 定义（注意：right/left 是 index，是否 +1 取决于你想覆盖的范围）
    sq = int(min(right - left, bottom - top))
    if sq <= 0:
        raise ValueError(f"裁剪边长 sq={sq} 非法，请检查 gt 的有效区域大小。")

    crop_top, crop_left, crop_size = top, left, sq

    # --- 1) 裁剪图像 ---
    out = inp_data[crop_top:crop_top + crop_size, crop_left:crop_left + crop_size]
    out = cv2.resize(out, (out_size, out_size), interpolation=cv2.INTER_NEAREST)

    # --- 2) box 映射到裁剪后的正方形，并 resize ---
    y0, x0, y1, x1 = map(int, box)

    # 平移到 crop 局部坐标
    y0 -= crop_top
    y1 -= crop_top
    x0 -= crop_left
    x1 -= crop_left

    # clip 到 [0, crop_size]（注意 y1/x1 作为切片上界允许等于 crop_size）
    y0 = int(np.clip(y0, 0, crop_size))
    y1 = int(np.clip(y1, 0, crop_size))
    x0 = int(np.clip(x0, 0, crop_size))
    x1 = int(np.clip(x1, 0, crop_size))

    # 缩放到 out_size
    scale = out_size / float(crop_size)
    new_box = [
        int(round(y0 * scale)),
        int(round(x0 * scale)),
        int(round(y1 * scale)),
        int(round(x1 * scale)),
    ]

    # 再 clip 一次到输出范围
    new_box[0] = int(np.clip(new_box[0], 0, out_size))
    new_box[2] = int(np.clip(new_box[2], 0, out_size))
    new_box[1] = int(np.clip(new_box[1], 0, out_size))
    new_box[3] = int(np.clip(new_box[3], 0, out_size))

    return out, new_box


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

    title_name = ["(a) Image", "(b) Zoom-in Image", "(c) Zoom-in Mask", "(d) Zoom-in Pred"]
    title_to = ['image', 'zoom_in_image', 'zoom_in_mask', 'zoom_in_pred']
    h_c = len(inp_list)
    w_c = len(title_name)

    w_px = w_c * (win_size + win_gap) + 2 * win_gap
    h_px = h_c * (win_size + win_gap) + 5 * win_gap

    fig = plt.figure(figsize=(w_px / dpi, h_px / dpi), dpi=dpi)

    assert w_c == len(title_name) == len(title_to)
    for h_i in range(h_c):
        for w_i in range(w_c):
            left = w_i * (win_size + win_gap) + 3 / 2 * win_gap
            bottom = h_px - (h_i + 1) * (win_size + win_gap)
            ax = fig.add_axes([left / w_px, bottom / h_px, win_size / w_px, win_size / h_px])

            subplot_data = inp_list[h_i][title_to[w_i]].astype("int")
            if w_i > 1:
                subplot_data = overlap(img=inp_list[h_i]['zoom_in_image'], pred=subplot_data, gt=inp_list[h_i]["zoom_in_mask"])

            ignore_index = 255
            box = draw_ids[h_i]['zoom_in']
            if np.any(inp_list[h_i]["mask"] == ignore_index) and w_i == 0:
                subplot_data, box = crop_square(subplot_data, inp_list[h_i]["mask"], ignore_index, box)

            ax.imshow(subplot_data)
            ax.set_xticks([])  # 移除x轴刻度
            ax.set_yticks([])  # 移除y轴刻度
            ax.set_aspect('equal')  # 设置每个子图为正方形
            if w_i == 0:
                rect = patches.Rectangle((box[1], box[0]),
                                         box[3] - box[1],
                                         box[2] - box[0],
                                         linewidth=2, edgecolor=[255/255, 111/255, 97/255], facecolor='none')
                ax.add_patch(rect)
            else:
                box = draw_ids[h_i]["pmt"]
                rect = patches.Rectangle((box[1], box[0]),
                                         box[3] - box[1],
                                         box[2] - box[0],
                                         linewidth=2, edgecolor=[255/255, 111/255, 97/255], facecolor='none')
                ax.add_patch(rect)
            for spine in ax.spines.values():
                spine.set_visible(False)
            if h_i == h_c - 1:
                ax.set_xlabel(title_name[w_i], fontsize=16, fontname="times new roman")
            if w_i == 0:
                ax.set_ylabel(draw_ids[h_i]["sign"], fontsize=16, fontname="SimSun")

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

    fig.legend(
        handles=handles,
        loc='lower center',
        ncol=len(handles)//2,  # uavid=8 类一行；想两行可改成 4
        bbox_to_anchor=(0.5, 0.02),
        frameon=False,
        prop={'family': 'SimSun', 'size': 12},
        handlelength=3,
        handleheight=1.2,
        columnspacing=3.0,
        borderaxespad=0.0
    )

    plt.savefig(f'fig/Error_Cls.png')
    plt.close()


if __name__ == "__main__":
    data = get_draw_data()
    plot_fig(data)
