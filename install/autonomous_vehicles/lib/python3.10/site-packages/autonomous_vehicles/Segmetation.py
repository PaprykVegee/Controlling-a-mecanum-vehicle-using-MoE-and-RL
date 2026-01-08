import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
from torchvision import transforms
from PIL import Image as PILImage
import numpy as np

class EvalModel:
    def __init__(self, path_to_checkpoint, n_classes=8, device='cuda'):
        self.device = device
        self.model = smp.Unet(
            encoder_name="resnet18",
            encoder_weights="imagenet",
            in_channels=3,
            classes=n_classes
        )
        checkpoint = torch.load(path_to_checkpoint, map_location=device)
        state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('model.'):
                new_state_dict[k[len('model.'):]] = v
            else:
                new_state_dict[k] = v
        self.model.load_state_dict(new_state_dict)
        self.model.to(device)
        self.model.eval()
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
        ])

    @torch.no_grad()
    def predict(self, img):
        if isinstance(img, PILImage.Image):
            img_pil = img
        else:
            img_pil = PILImage.fromarray(img)
        
        input_tensor = self.transform(img_pil).unsqueeze(0).to(self.device)  # shape [1, C, H, W]
        
        with torch.no_grad():
            logits = self.model(input_tensor)  # shape [1, num_classes, H, W]
            probs = torch.softmax(logits, dim=1)  # opcjonalnie, jeśli chcesz prawdopodobieństwa
            preds = torch.argmax(logits, dim=1).squeeze(0)  # shape [H, W]
        
        num_classes = logits.shape[1]
        H, W = preds.shape
        
        # Tworzymy maskę bool dla każdej klasy
        mask_bool = np.zeros((H, W, num_classes), dtype=bool)
        for c in range(num_classes):
            mask_bool[:, :, c] = (preds.cpu().numpy() == c)
        
        return mask_bool



