import torch
from diffusers import StableDiffusionInpaintPipeline
from PIL import Image
import cv2
import numpy as np

# Load the inpainting model
model_id = "stabilityai/stable-diffusion-2-inpainting"
pipe = StableDiffusionInpaintPipeline.from_pretrained(
    model_id, torch_dtype=torch.float32  # Use float32 for CPU
)
pipe = pipe.to(torch.device("cpu"))  # Explicitly set to CPU


# Function to create a mask for shadows
def create_shadow_mask(image_path, threshold=100):
    # Load the image
    image = cv2.imread(image_path)
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Threshold to detect dark areas (shadows)
    _, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)
    # Refine mask with morphological operations
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=1)
    # Convert mask to PIL Image
    mask_image = Image.fromarray(mask).convert("L")
    return mask_image


# Main function to remove shadows
def remove_shadows(input_image_path, output_image_path):
    # Load the input image
    input_image = Image.open(input_image_path).convert("RGB")

    # Create shadow mask
    mask_image = create_shadow_mask(input_image_path)

    # Resize images to 512x512 for Stable Diffusion
    input_image = input_image.resize((512, 512))
    mask_image = mask_image.resize((512, 512))

    # Prompt to guide inpainting
    prompt = "A clean, well-lit scene without shadows, matching the surrounding area"

    # Perform inpainting
    output = pipe(
        prompt=prompt,
        image=input_image,
        mask_image=mask_image,
        num_inference_steps=50,
        guidance_scale=7.5,
    ).images[0]

    # Save the result
    output.save(output_image_path)


# Example usage
input_path = "2.png"  # Replace with your image path
output_path = "shadow_removed_image.png"
remove_shadows(input_path, output_path)