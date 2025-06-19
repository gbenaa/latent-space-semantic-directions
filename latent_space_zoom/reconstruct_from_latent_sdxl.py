import torch
from PIL import Image
from diffusers import StableDiffusionXLPipeline
from torchvision.transforms import ToPILImage

# --- configuration ----------------------------------------------------------
model_id      = "stabilityai/stable-diffusion-xl-base-1.0"
image_path    = "image_base_1024.png"            # must be 1024×1024 RGB
output_path   = "reconstructed_from_latent_sdxl.png"

# scaling is only needed when you plan to feed latents to the UNet;                 #
# for a round-trip (encode ➜ decode) you can skip it.  kept here for completeness. #
latent_scale  = 0.13025

# --- load pipeline in fp16, then convert ONLY the VAE to fp32 --------------------
pipe = StableDiffusionXLPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.float16,     # keep UNet / text encoders in fp16
    variant="fp16",
    use_safetensors=True,
).to("cuda")

pipe.vae.to(dtype=torch.float32)   # <<< critical line

# --- preprocess ---------------------------------------------------------------
img = Image.open(image_path).convert("RGB").resize((1024, 1024))
img_tensor = pipe.image_processor.preprocess(img).to("cuda", dtype=torch.float32)  # fp32 to match VAE

# --- encode → latent ----------------------------------------------------------
with torch.no_grad():
    latents = pipe.vae.encode(img_tensor).latent_dist.mean * latent_scale  # use mean to avoid extra noise

# --- (optionally) convert to fp16 if you need to save VRAM --------------------
latents = latents.to(dtype=torch.float32)        # keep fp32 for decode; fp16 also works now

# --- decode -------------------------------------------------------------------
with torch.no_grad():
    recon = pipe.vae.decode(latents / latent_scale).sample

# --- post-process & save ------------------------------------------------------
recon = (recon[0] * 0.5 + 0.5).clamp(0, 1).cpu()  # [-1,1] ➜ [0,1]
ToPILImage()(recon).save(output_path)
print("saved:", output_path)
