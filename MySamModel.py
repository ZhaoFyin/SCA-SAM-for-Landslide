import torch
import torch.nn as nn
from SAM.build_sam import sam_model_registry


def remap_for_lora(w):
    new_w = {}
    for k, v in list(w.items()):
        if "qkv" in k:
            new_k = k.replace(".weight", ".base.weight")
            new_w[new_k] = v
        elif "proj" and ".weight" in k:
            new_k = k.replace(".weight", ".base.weight")
            new_w[new_k] = v
        elif "proj" and ".bias" in k:
            new_k = k.replace(".bias", ".base.bias")
            new_w[new_k] = v
        else:
            new_w[k] = v
    return new_w


class ScaSAM(nn.Module):
    def __init__(self, lora_cfg, num_classes, sca, img_s=512):
        super(ScaSAM, self).__init__()

        self.net, _ = sam_model_registry["vit_b"](image_size=img_s,
                                                  num_classes=num_classes,
                                                  checkpoint=r"./cp/sam_vit_b_01ec64.pth",
                                                  pixel_mean=[0.3394, 0.3598, 0.3226],
                                                  pixel_std=[0.2037, 0.1899, 0.1922],
                                                  use_sca=sca,
                                                  lora_cfg=lora_cfg,
                                                  process_cp=lora_cfg["enable"],
                                                  report=False)
        if sca:
            sca_w = torch.load(r"cp/sca.pth")
            if lora_cfg["sca_enable"]:
                sca_w = remap_for_lora(sca_w)

            self.net.load_state_dict(sca_w, strict=False)


        for p in self.net.parameters():
            p.requires_grad = True

        for p in self.net.image_encoder.parameters():
            p.requires_grad = False
        for n, p in self.net.image_encoder.named_parameters():
            if hasattr(p, "is_lora_param"):
                p.requires_grad = True
            if ".sca." in n:
                p.requires_grad = True
            if "base" in n:
                p.requires_grad = False
            if "neck" in n:
                p.requires_grad = False

    def forward(self, x, multi_output, img_size):
        return self.net(x, multi_output, img_size)
