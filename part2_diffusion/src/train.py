"""2.2 LoRA fine-tuning of the Stable Diffusion 1.5 UNet."""
import json
import time

import torch
import torch.nn.functional as F
from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionPipeline, UNet2DConditionModel
from diffusers.utils import convert_state_dict_to_diffusers
from peft import LoraConfig
from peft.utils import get_peft_model_state_dict
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from transformers import CLIPTextModel, CLIPTokenizer

from .common import BIG, CONFIG, DATA, METRICS, SEED, load_json, model_path, save_json


class CaptionedImages(Dataset):
    def __init__(self, folder, tokenizer):
        self.folder = folder
        self.rows = [json.loads(line) for line in open(folder / "metadata.jsonl")]
        self.tokenizer = tokenizer
        self.transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),        # pixels to [-1, 1]
        ])

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        row = self.rows[i]
        image = self.transform(Image.open(self.folder / row["file_name"]).convert("RGB"))
        ids = self.tokenizer(row["text"], padding="max_length", truncation=True,
                             max_length=self.tokenizer.model_max_length,
                             return_tensors="pt").input_ids[0]
        return image, ids


def train(run, max_steps=None):
    out_dir = BIG / "loras" / run
    stats_file = METRICS / f"train_{run}.json"
    if (out_dir / "pytorch_lora_weights.safetensors").exists() and stats_file.exists():
        print(run, "already trained")
        return load_json(stats_file)

    settings = CONFIG["train"]
    rank = CONFIG["runs"][run]["rank"]
    steps = max_steps or settings["steps"]
    base = model_path(CONFIG["base_model"])
    torch.manual_seed(SEED)

    tokenizer = CLIPTokenizer.from_pretrained(base, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(base, subfolder="text_encoder", variant="fp16")
    vae = AutoencoderKL.from_pretrained(base, subfolder="vae", variant="fp16")
    unet = UNet2DConditionModel.from_pretrained(base, subfolder="unet", variant="fp16")
    noise_scheduler = DDPMScheduler.from_pretrained(base, subfolder="scheduler")

    # Everything frozen and in fp16; only the LoRA layers added below are trained (in fp32).
    for model in (text_encoder, vae, unet):
        model.requires_grad_(False)
        model.to("cuda", dtype=torch.float16)
    unet.add_adapter(LoraConfig(r=rank, lora_alpha=rank, init_lora_weights="gaussian",
                                target_modules=settings["target_modules"]))
    lora_params = [p for p in unet.parameters() if p.requires_grad]
    for p in lora_params:
        p.data = p.data.float()
    unet.enable_gradient_checkpointing()          # saves memory on the 16 GB T4
    unet.train()

    n_trainable = sum(p.numel() for p in lora_params)
    n_total = sum(p.numel() for p in unet.parameters())
    print(f"{run}: rank {rank}, trainable parameters {n_trainable:,} "
          f"({100 * n_trainable / n_total:.2f}% of the UNet)")

    data = CaptionedImages(DATA / "train", tokenizer)
    loader = DataLoader(data, batch_size=settings["batch_size"], shuffle=True, drop_last=True,
                        num_workers=2, generator=torch.Generator().manual_seed(SEED))
    optimizer = torch.optim.AdamW(lora_params, lr=settings["learning_rate"])
    scaler = torch.amp.GradScaler("cuda")

    torch.cuda.reset_peak_memory_stats()
    start = time.time()
    losses = []
    step = 0
    while step < steps:
        for images, ids in loader:
            with torch.no_grad():
                latents = vae.encode(images.to("cuda", torch.float16)).latent_dist.sample()
                latents = latents * vae.config.scaling_factor
                text = text_encoder(ids.to("cuda"))[0]

            # add noise at a random timestep; the UNet learns to predict that noise
            noise = torch.randn_like(latents)
            t = torch.randint(0, noise_scheduler.config.num_train_timesteps, (len(latents),),
                              device="cuda")
            noisy = noise_scheduler.add_noise(latents, noise, t)
            with torch.autocast("cuda", dtype=torch.float16):
                pred = unet(noisy, t, encoder_hidden_states=text).sample
            loss = F.mse_loss(pred.float(), noise.float())

            scaler.scale(loss).backward()
            if step == 0:
                assert any(p.grad is not None for p in lora_params), "no gradient reached LoRA"
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(lora_params, 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            step += 1
            losses.append(loss.item())
            if step % 100 == 0 or step == 20:
                minutes = (time.time() - start) / 60
                print(f"step {step}/{steps}  loss {sum(losses[-100:]) / len(losses[-100:]):.4f}  "
                      f"{minutes:.1f} min, about {minutes / step * (steps - step):.0f} min left")
            if step == steps:
                break

    train_seconds = time.time() - start
    peak_gb = torch.cuda.max_memory_allocated() / 1024**3

    lora_layers = convert_state_dict_to_diffusers(get_peft_model_state_dict(unet))
    StableDiffusionPipeline.save_lora_weights(str(out_dir), unet_lora_layers=lora_layers)

    stats = {"run": run, "rank": rank, "trainable_params": n_trainable,
             "learning_rate": settings["learning_rate"], "steps": steps,
             "batch_size": settings["batch_size"], "peak_gpu_memory_gb": round(peak_gb, 2),
             "training_minutes": round(train_seconds / 60, 1),
             "gpu_hours": round(train_seconds / 3600, 3), "gpu": torch.cuda.get_device_name(0),
             "losses": losses}
    save_json(stats, stats_file)

    del unet, vae, text_encoder, optimizer
    torch.cuda.empty_cache()
    return stats
