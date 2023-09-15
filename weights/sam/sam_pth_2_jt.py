import torch 
import jittor as jt
import os

with open('checkpoint/sam_vit_b_01ec64.pth', "rb") as f:
    state_dict = torch.load(f)
    save_dict = {
                "state_dict": state_dict,
            }
    jt.save(
        save_dict,
        os.path.join("./weights/sam/sam_vit_b_01ec64.pth.tar"),
    )
