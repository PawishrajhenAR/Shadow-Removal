import torch
import os
from PIL import Image
import torchvision.transforms as transforms
import torchvision.utils as vutils
import torch.nn as nn

# Models
class UNetShadowDetection(nn.Module):
    def __init__(self):
        super(UNetShadowDetection, self).__init__()
        self.enc1 = nn.Conv2d(3, 64, 3, padding=1)
        self.enc2 = nn.Conv2d(64, 128, 3, padding=1)
        self.dec1 = nn.Conv2d(128, 64, 3, padding=1)
        self.dec2 = nn.Conv2d(64, 1, 3, padding=1)

    def forward(self, x):
        e1 = torch.relu(self.enc1(x))
        e2 = torch.relu(self.enc2(e1))
        d1 = torch.relu(self.dec1(e2))
        d2 = torch.sigmoid(self.dec2(d1))
        return d2

class UNetShadowRemoval(nn.Module):
    def __init__(self):
        super(UNetShadowRemoval, self).__init__()
        self.enc1 = nn.Conv2d(4, 64, 3, padding=1)
        self.enc2 = nn.Conv2d(64, 128, 3, padding=1)
        self.dec1 = nn.Conv2d(128, 64, 3, padding=1)
        self.dec2 = nn.Conv2d(64, 3, 3, padding=1)

    def forward(self, x):
        e1 = torch.relu(self.enc1(x))
        e2 = torch.relu(self.enc2(e1))
        d1 = torch.relu(self.dec1(e2))
        d2 = torch.tanh(self.dec2(d1))
        return d2

# Function to remove shadow from a single image
def remove_shadow(image_tensor, detection_model, removal_model, device):
    detection_model.eval()
    removal_model.eval()
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        predicted_mask = detection_model(image_tensor)
        predicted_mask_bin = (predicted_mask > 0.5).float()
        input_removal = torch.cat([image_tensor, predicted_mask_bin], dim=1)
        shadow_free = removal_model(input_removal)
    return shadow_free, predicted_mask_bin

# Main inference function
def infer_single_image(image_path, detection_model_path, removal_model_path, output_dir="output"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load models
    detection_model = UNetShadowDetection().to(device)
    removal_model = UNetShadowRemoval().to(device)
    detection_model.load_state_dict(torch.load(detection_model_path, map_location=device))
    removal_model.load_state_dict(torch.load(removal_model_path, map_location=device))

    # Transform image
    transform_rgb = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
    ])

    image = Image.open(image_path).convert("RGB")
    input_tensor = transform_rgb(image).unsqueeze(0)

    shadow_free, predicted_mask = remove_shadow(input_tensor, detection_model, removal_model, device)

    # Create output folder
    os.makedirs(output_dir, exist_ok=True)

    # Save results
    shadow_free = (shadow_free + 1) / 2  # Denormalize
    vutils.save_image(shadow_free, os.path.join(output_dir, "shadow_free.png"))
    vutils.save_image(predicted_mask, os.path.join(output_dir, "predicted_mask.png"))
    print("Inference complete. Results saved in:", output_dir)

# Example usage
if __name__ == "__main__":
    infer_single_image(
        image_path="2.png",  # Replace with your test image path
        detection_model_path="detection_model.pth",
        removal_model_path="removal_model.pth",
        output_dir="outputs"
    )
