from typing import Callable, Optional
import torch.nn.functional as F

from .efficientvitsam.efficientvit import (
    EfficientViTSam,
    efficientvit_sam_l0,
    efficientvit_sam_l1,
    efficientvit_sam_l2,
    efficientvit_sam_xl0,
    efficientvit_sam_xl1,
)
from .efficientvitsam.nn.norm import set_norm_eps
from .efficientvitsam.utils import load_state_dict_from_file

__all__ = ["create_efficientvit_sam_model"]


REGISTERED_EFFICIENTVIT_SAM_MODEL: dict[str, tuple[Callable, float, str]] = {
    "efficientvit-sam-l0": (efficientvit_sam_l0, 1e-6, "./cp/efficientvit_sam_l0.pt"),
}


def load_from(sam, state_dict, image_size, vit_patch_size):
    sam_dict = sam.state_dict()
    except_keys = ['mask_tokens', 'output_hypernetworks_mlps', 'iou_prediction_head']
    new_state_dict = {k: v for k, v in state_dict.items() if
                      k in sam_dict.keys() and except_keys[0] not in k and except_keys[1] not in k and except_keys[2] not in k}

    sam_dict.update(new_state_dict)
    return sam_dict


def create_efficientvit_sam_model(
    name: str, num_cls: int, pretrained=True, weight_url: Optional[str] = None, **kwargs
) -> EfficientViTSam:
    if name not in REGISTERED_EFFICIENTVIT_SAM_MODEL:
        raise ValueError(
            f"Cannot find {name} in the model zoo. List of models: {list(REGISTERED_EFFICIENTVIT_SAM_MODEL.keys())}"
        )
    else:
        model_cls, norm_eps, default_pt = REGISTERED_EFFICIENTVIT_SAM_MODEL[name]
        model = model_cls(num_cls=num_cls, **kwargs)
        set_norm_eps(model, norm_eps)
        weight_url = default_pt if weight_url is None else weight_url

    if pretrained:
        if weight_url is None:
            raise ValueError(f"Cannot find the pretrained weight of {name}.")
        else:
            weight = load_state_dict_from_file(weight_url)
            weight = load_from(model, weight, image_size=512, vit_patch_size=16)
            model.load_state_dict(weight)
    return model