import os
import cv2
import torch
import numpy as np
import streamlit as st
from PIL import Image
import torchvision.transforms as transforms
import torchvision.utils as vutils
from diffusers import StableDiffusionInpaintPipeline
from pathlib import Path

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
    return mask_pil, mask

def inpaint_with_sd(image_path, mask_image_pil, output_path):
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
    return result

def inpaint_with_opencv(image_path, mask_np, output_path):
    image = cv2.imread(image_path)
    inpainted = cv2.inpaint(image, mask_np, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, inpainted)
    return Image.open(output_path)

def inpaint_with_unet(image_path, detection_model_path, removal_model_path, output_path):
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
    return Image.open(output_path)

# ========== Streamlit Interface ==========

st.title("Shadow Removal Application")
st.markdown("Upload an image and select a method to remove shadows.")

# Create temp directory if it doesn't exist
temp_dir = Path("temp")
temp_dir.mkdir(exist_ok=True)

uploaded_file = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg"])
method = st.selectbox("Select Shadow Removal Method", ["U-Net", "Stable Diffusion", "OpenCV (DeepFill)"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Original Image", use_column_width=True)
    temp_input_path = str(temp_dir / "input_image.png")
    image.save(temp_input_path)

    if st.button("Run Shadow Removal"):
        mask_pil, mask_np = create_shadow_mask(temp_input_path)
        st.image(mask_pil, caption="Generated Mask", use_column_width=True)

        output_path = "temp/result.png"

        if method == "U-Net":
            detection_model_path = "detection_model.pth"
            removal_model_path = "removal_model.pth"
            result_image = inpaint_with_unet(temp_input_path, detection_model_path, removal_model_path, output_path)

        elif method == "Stable Diffusion":
            result_image = inpaint_with_sd(temp_input_path, mask_pil, output_path)

        elif method == "OpenCV (DeepFill)":
            result_image = inpaint_with_opencv(temp_input_path, mask_np, output_path)

        st.image(result_image, caption="Shadow Removed Image", use_column_width=True)
        with open(output_path, "rb") as f:
            st.download_button("Download Result", f, file_name="shadow_free.png")
