import os
import torch


if __name__ == "__main__":
    in_path = "./data/pretrained_models/fcos3d.pth"
    out_path = "./data/pretrained_models/fcos3d_converted.pth"
    ckpt = torch.load(in_path, map_location="cpu")

    new_ckpt = dict(state_dict=dict())
    for k, v in ckpt["state_dict"].items():
        if k.startswith("img_backbone."):
            print(k)
            new_ckpt["state_dict"][k.replace("img_backbone.", "")] = v
    torch.save(new_ckpt, out_path)