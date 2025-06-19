import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from diffusers import StableDiffusionXLPipeline
from torchvision import transforms
import os

# ---------- Config ----------
image_path = "reconstructed_from_latent_sdxl.png"       # Input image (512×512)
output_dir = "latent_zoom_images"   # Where to save frames
latent_crop_ratio = 0.4                 # Final zoom = 40% crop
frames = 10                             # Number of zoom frames

os.makedirs(output_dir, exist_ok=True)

# ---------- Load SDXL ----------
pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
    variant="fp16",
    use_safetensors=True,
).to("cuda")
pipe.vae.to(dtype=torch.float32)

# ---------- Preprocess image ----------
image = Image.open(image_path).convert("RGB").resize((512, 512))
preprocess = transforms.Compose([
    transforms.ToTensor(),                                        # [0,1]
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),        # [-1,1]
])
image_tensor = preprocess(image).unsqueeze(0).to("cuda", dtype=torch.float32)

# ---------- Encode to latent (noise-free) ----------
with torch.no_grad(), torch.cuda.amp.autocast(False):
    latent = pipe.vae.encode(image_tensor).latent_dist.mode()     # Clean latent
    latent = latent.to(dtype=torch.float32)                       # Ensure FP32

# ---------- Latent zoom helper ----------
def crop_and_resize_latent(latent, zoom_factor, target_shape=(64, 64)):
    _, c, h, w = latent.shape
    crop_h = int(h / zoom_factor)
    crop_w = int(w / zoom_factor)
    top = (h - crop_h) // 2
    left = (w - crop_w) // 2
    cropped = latent[:, :, top:top + crop_h, left:left + crop_w]
    resized = F.interpolate(cropped, size=target_shape, mode="bilinear", align_corners=False)
    return resized

# ---------- Generate zoom frames ----------
for i, alpha in enumerate(np.linspace(0, 1, frames)):
    zoom = 1.0 + alpha * ((1 / latent_crop_ratio) - 1.0)
    zoomed_latent = crop_and_resize_latent(latent, zoom)

    with torch.no_grad(), torch.cuda.amp.autocast(False):
        decoded = pipe.vae.decode(zoomed_latent).sample

    decoded_image = (decoded[0].cpu() * 0.5 + 0.5).clamp(0, 1)
    img = Image.fromarray((decoded_image.permute(1, 2, 0).numpy() * 255).astype(np.uint8))
    path = os.path.join(output_dir, f"zoom_{i:02d}.png")
    img.save(path)
    print(f"Saved: {path}")
