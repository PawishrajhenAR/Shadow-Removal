import os
import cv2
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import torchvision.utils as vutils
from diffusers import StableDiffusionInpaintPipeline

# ========== U-Net Models ==========

class UNetShadowDetection(torch.nn.Module):
    def __init__(self):
        super(UNetShadowDetection, self).__init__()
        self.enc1 = torch.nn.Conv2d(3, 64, 3, padding=1)
        self.enc2 = torch.nn.Conv2d(64, 128, 3, padding=1)
        self.dec1 = torch.nn.Conv2d(128, 64, 3, padding=1)
        self.dec2 = torch.nn.Conv2d(64, 1, 3, padding=1)

    def forward(self, x):
        e1 = torch.relu(self.enc1(x))
        e2 = torch.relu(self.enc2(e1))
        d1 = torch.relu(self.dec1(e2))
        d2 = torch.sigmoid(self.dec2(d1))
        return d2

class UNetShadowRemoval(torch.nn.Module):
    def __init__(self):
        super(UNetShadowRemoval, self).__init__()
        self.enc1 = torch.nn.Conv2d(4, 64, 3, padding=1)
        self.enc2 = torch.nn.Conv2d(64, 128, 3, padding=1)
        self.dec1 = torch.nn.Conv2d(128, 64, 3, padding=1)
        self.dec2 = torch.nn.Conv2d(64, 3, 3, padding=1)

    def forward(self, x):
        e1 = torch.relu(self.enc1(x))
        e2 = torch.relu(self.enc2(e1))
        d1 = torch.relu(self.dec1(e2))
        d2 = torch.tanh(self.dec2(d1))
        return d2

# ========== Utility Functions ==========

def create_shadow_mask(image_path, threshold=100):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=1)
    mask_pil = Image.fromarray(mask).convert("L")
    return mask_pil, mask  # Return both PIL and NumPy formats

def inpaint_with_sd(image_path, mask_image_pil, output_path):
    print("[*] Running Stable Diffusion Inpainting...")
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        "stabilityai/stable-diffusion-2-inpainting",
        torch_dtype=torch.float32
    ).to("cpu")

    input_image = Image.open(image_path).convert("RGB").resize((512, 512))
    mask_image_pil = mask_image_pil.resize((512, 512))

    prompt = "A clean, well-lit scene without shadows, matching the surrounding area"
    result = pipe(
        prompt=prompt,
        image=input_image,
        mask_image=mask_image_pil,
        num_inference_steps=50,
        guidance_scale=7.5
    ).images[0]

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    result.save(output_path)
    print(f"[✓] Saved Stable Diffusion result: {output_path}")

def inpaint_with_opencv(image_path, mask_np, output_path):
    print("[*] Running OpenCV (Telea/DeepFill-style) Inpainting...")
    image = cv2.imread(image_path)
    inpainted = cv2.inpaint(image, mask_np, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, inpainted)
    print(f"[✓] Saved OpenCV result: {output_path}")

def inpaint_with_unet(image_path, detection_model_path, removal_model_path, output_path):
    print("[*] Running custom U-Net shadow removal...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    detection_model = UNetShadowDetection().to(device)
    removal_model = UNetShadowRemoval().to(device)
    detection_model.load_state_dict(torch.load(detection_model_path, map_location=device))
    removal_model.load_state_dict(torch.load(removal_model_path, map_location=device))

    transform_rgb = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
    ])

    image = Image.open(image_path).convert("RGB")
    input_tensor = transform_rgb(image).unsqueeze(0).to(device)

    detection_model.eval()
    removal_model.eval()

    with torch.no_grad():
        predicted_mask = detection_model(input_tensor)
        predicted_mask_bin = (predicted_mask > 0.5).float()
        input_removal = torch.cat([input_tensor, predicted_mask_bin], dim=1)
        shadow_free = removal_model(input_removal)

    shadow_free = (shadow_free + 1) / 2
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    vutils.save_image(shadow_free, output_path)
    print(f"[✓] Saved UNet result: {output_path}")

# ========== Unified Entry Point ==========

def remove_shadow(
    image_path,
    method="unet",
    output_path="outputs/result.png",
    detection_model_path=None,
    removal_model_path=None
):
    mask_pil, mask_np = create_shadow_mask(image_path)

    if method == "sd":
        inpaint_with_sd(image_path, mask_pil, output_path)

    elif method == "deepfill":
        inpaint_with_opencv(image_path, mask_np, output_path)

    elif method == "unet":
        assert detection_model_path and removal_model_path, "Please provide UNet model paths."
        inpaint_with_unet(image_path, detection_model_path, removal_model_path, output_path)

    else:
        raise ValueError(f"Unknown method: {method}")

# ========== Example Usage ==========

if __name__ == "__main__":
    # Choose one of: 'unet', 'sd', 'deepfill'
    remove_shadow(
        image_path="2.png",
        method="sd",  # change to 'sd' or 'deepfill' as needed
        output_path="outputs/unet_result.png",
        detection_model_path="detection_model.pth",
        removal_model_path="removal_model.pth"
    )
