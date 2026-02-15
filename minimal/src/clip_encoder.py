# -----------------------------
# CLIP encoder
# -----------------------------
import open_clip
import torch
import cv2
from typing import List
import numpy as np

class Clip:
    def __init__(self, model="ViT-B-32", pretrained="openai", device="cuda"):
        self.device = device if (torch.cuda.is_available() and device.startswith("cuda")) else "cpu"
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(model, pretrained=pretrained)
        self.model = self.model.to(self.device).eval()
        self.tokenizer = open_clip.get_tokenizer(model)

    @torch.inference_mode()
    def encode_text(self, text: str) -> torch.Tensor:
        toks = self.tokenizer([text]).to(self.device)
        emb = self.model.encode_text(toks)
        return l2(emb).squeeze(0).cpu()

    @torch.inference_mode()
    def encode_images_bgr(self, frames_bgr: List[np.ndarray], batch_size=32) -> torch.Tensor:
        import PIL.Image
        outs = []
        for i in range(0, len(frames_bgr), batch_size):
            batch = frames_bgr[i:i+batch_size]
            imgs = []
            for fr in batch:
                rgb = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
                imgs.append(self.preprocess(PIL.Image.fromarray(rgb)))
            x = torch.stack(imgs, dim=0).to(self.device)
            emb = self.model.encode_image(x)
            outs.append(l2(emb).cpu())
        return torch.cat(outs, dim=0) if outs else torch.empty((0, 512))
    

def l2(x: torch.Tensor, eps=1e-8) -> torch.Tensor:
    return x / (x.norm(dim=-1, keepdim=True) + eps)
