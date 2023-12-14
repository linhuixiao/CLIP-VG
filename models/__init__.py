from .clip_vg import CLIP_VG
from .clip_vg import ML_CLIP_VG
from .clip_vg import ML_CLIP_VG_PROMPT


def build_model(args):
    return ML_CLIP_VG(args)

    # TODO: We provide the last-layer feature version benefit for researchers conducting ablation studies.
    #       Additionally, we offer the implementation of learnable prompts for ML_CLIP_VG, benefiting researchers
    #       in their respective research.
    # return CLIP_VG(args)
    # return ML_CLIP_VG_PROMPT(args)


