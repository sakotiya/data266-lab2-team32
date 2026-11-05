"""Task 2.2 - LoRA fine-tuning of the Stable Diffusion 1.5 UNet.

The VAE and text encoder stay frozen, so their outputs are computed once before training:
  - the VAE latent distribution (mean, std) of every training image, plain and mirrored,
  - the CLIP text embedding of every caption.
Each step then samples a latent from the cached distribution, adds noise at a random timestep,
and trains the LoRA layers in the UNet to predict that noise (the standard SD objective).
Same data, step budget and settings for both runs; only the LoRA rank differs.

    python -m part2_diffusion.src.train_lora run1_lora_r4
"""

from __future__ import annotations

import gc
import json
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from .common import (DATA, SEED, config, gpu_name, read_json, repo_dir, seed_everything,
                     work_dir, write_csv, write_json)


def load_split(name: str):
    with open(DATA / name / "metadata.jsonl") as fh:
        rows = [json.loads(line) for line in fh]
    return [(DATA / name / r["file_name"], r["text"]) for r in rows]


def to_tensor(path) -> torch.Tensor:
    x = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 127.5 - 1.0
    return torch.from_numpy(x).permute(2, 0, 1)


@torch.no_grad()
def cache_inputs(items, vae, tokenizer, text_encoder, device, dtype, bs: int = 8):
    """Latent mean/std for [plain, mirrored] images: [N, 2, 4, 64, 64]; text embeddings [N, 77, 768]."""
    means, stds, embs = [], [], []
    for i in range(0, len(items), bs):
        chunk = items[i:i + bs]
        x = torch.stack([to_tensor(p) for p, _ in chunk]).to(device, dtype)
        pair_m, pair_s = [], []
        for img in (x, torch.flip(x, dims=[3])):
            dist = vae.encode(img).latent_dist
            pair_m.append(dist.mean)
            pair_s.append(dist.std)
        means.append(torch.stack(pair_m, 1))
        stds.append(torch.stack(pair_s, 1))
        ids = tokenizer([t for _, t in chunk], padding="max_length", truncation=True,
                        max_length=tokenizer.model_max_length, return_tensors="pt").input_ids
        embs.append(text_encoder(ids.to(device))[0])
    return torch.cat(means), torch.cat(stds), torch.cat(embs)


def save_lora(unet, out_dir) -> None:
    from diffusers import StableDiffusionPipeline
    from diffusers.utils import convert_state_dict_to_diffusers
    from peft.utils import get_peft_model_state_dict

    out_dir.mkdir(parents=True, exist_ok=True)
    layers = convert_state_dict_to_diffusers(get_peft_model_state_dict(unet))
    StableDiffusionPipeline.save_lora_weights(save_directory=str(out_dir),
                                              unet_lora_layers=layers, safe_serialization=True)


def train(run: str, log=print) -> dict:
    from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
    from diffusers.optimization import get_scheduler
    from peft import LoraConfig
    from transformers import CLIPTextModel, CLIPTokenizer

    cfg = config(run)
    rc, base = cfg["run"], cfg["base_model"]
    out = work_dir("loras") / run
    stats_path = repo_dir("metrics") / f"train_{run}.json"
    if (out / "pytorch_lora_weights.safetensors").exists() and stats_path.exists():
        log(f"{run}: already trained, using {out}")
        return read_json(stats_path)

    seed_everything(SEED)
    device, wdtype = "cuda", torch.float16
    torch.cuda.reset_peak_memory_stats()
    t_wall = time.time()

    tokenizer = CLIPTokenizer.from_pretrained(base, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(base, subfolder="text_encoder").to(device, wdtype)
    vae = AutoencoderKL.from_pretrained(base, subfolder="vae").to(device, wdtype)
    unet = UNet2DConditionModel.from_pretrained(base, subfolder="unet").to(device, wdtype)
    noise_sched = DDPMScheduler.from_pretrained(base, subfolder="scheduler")
    for m in (text_encoder, vae, unet):
        m.requires_grad_(False)

    items = load_split("train")
    t0 = time.time()
    lat_mean, lat_std, text_emb = cache_inputs(items, vae, tokenizer, text_encoder, device, wdtype)
    scaling = vae.config.scaling_factor
    log(f"{run}: cached {len(items)} images x 2 flips in {time.time() - t0:.0f} s")
    del vae, text_encoder
    gc.collect()
    torch.cuda.empty_cache()

    unet.add_adapter(LoraConfig(r=rc["rank"], lora_alpha=rc["lora_alpha"],
                                init_lora_weights="gaussian", target_modules=rc["target_modules"]))
    params = [p for p in unet.parameters() if p.requires_grad]
    for p in params:                       # LoRA weights train in fp32; the frozen UNet stays fp16
        p.data = p.data.float()
    if rc["gradient_checkpointing"]:
        unet.enable_gradient_checkpointing()
    unet.train()
    n_train = sum(p.numel() for p in params)
    n_total = sum(p.numel() for p in unet.parameters())
    log(f"{run}: trainable params {n_train:,} of {n_total:,} ({100 * n_train / n_total:.3f}%)")

    opt = torch.optim.AdamW(params, lr=rc["learning_rate"], weight_decay=1e-2)
    steps, bs = rc["max_train_steps"], rc["train_batch_size"]
    lr_sched = get_scheduler(rc["lr_scheduler"], optimizer=opt,
                             num_warmup_steps=rc["warmup_steps"], num_training_steps=steps)
    scaler = torch.amp.GradScaler("cuda")
    g_gpu = torch.Generator(device).manual_seed(SEED)      # noise, timesteps, latent samples
    g_cpu = torch.Generator().manual_seed(SEED)            # batch order and flips
    n, order, losses, window = len(items), [], [], []
    T = noise_sched.config.num_train_timesteps

    torch.cuda.reset_peak_memory_stats()
    t_train = time.time()
    for step in range(1, steps + 1):
        if len(order) < bs:                                # a fresh shuffle every epoch
            order += torch.randperm(n, generator=g_cpu).tolist()
        idx = torch.tensor(order[:bs], device=device)
        order = order[bs:]
        flip = ((torch.rand(bs, generator=g_cpu) < 0.5) if rc["random_flip"]
                else torch.zeros(bs, dtype=torch.bool)).long().to(device)

        mean, std = lat_mean[idx, flip], lat_std[idx, flip]
        latents = (mean + std * torch.randn(mean.shape, generator=g_gpu, device=device,
                                            dtype=mean.dtype)) * scaling
        noise = torch.randn(latents.shape, generator=g_gpu, device=device, dtype=latents.dtype)
        t = torch.randint(0, T, (bs,), generator=g_gpu, device=device)
        noisy = noise_sched.add_noise(latents, noise, t)
        target = (noise if noise_sched.config.prediction_type == "epsilon"
                  else noise_sched.get_velocity(latents, noise, t))

        with torch.autocast("cuda", dtype=torch.float16):
            pred = unet(noisy, t, encoder_hidden_states=text_emb[idx]).sample
        loss = F.mse_loss(pred.float(), target.float())
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        if step == 1:                      # a silent no-op (no gradient reaching LoRA) would waste hours
            gnorm = sum(float(p.grad.abs().sum()) for p in params if p.grad is not None)
            assert gnorm > 0, "LoRA parameters received no gradient"
        torch.nn.utils.clip_grad_norm_(params, rc["max_grad_norm"])
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)
        lr_sched.step()

        window.append(loss.item())
        if step % 25 == 0 or step == steps:
            el = time.time() - t_train
            losses.append({"step": step, "loss": float(np.mean(window)),
                           "lr": lr_sched.get_last_lr()[0], "elapsed_s": round(el, 1)})
            window = []
            if step % 100 == 0 or step == 25 or step == steps:
                log(f"{run}: step {step}/{steps} loss {losses[-1]['loss']:.4f} "
                    f"{step / el:.2f} it/s, ETA {(steps - step) * el / step / 60:.0f} min")
        if rc["checkpoint_every"] and step % rc["checkpoint_every"] == 0 and step < steps:
            save_lora(unet, out / f"checkpoint-{step}")

    train_s = time.time() - t_train
    peak_alloc = torch.cuda.max_memory_allocated() / 2**30
    peak_res = torch.cuda.max_memory_reserved() / 2**30
    save_lora(unet, out)
    wall_s = time.time() - t_wall
    stats = {
        "run": run, "method": rc["method"], "rank": rc["rank"], "lora_alpha": rc["lora_alpha"],
        "target_modules": rc["target_modules"], "trainable_params": n_train,
        "unet_params": n_total, "trainable_pct": round(100 * n_train / n_total, 4),
        "learning_rate": rc["learning_rate"], "lr_scheduler": rc["lr_scheduler"],
        "steps": steps, "batch_size": bs, "train_images": n,
        "epochs": round(steps * bs / n, 2), "mixed_precision": rc["mixed_precision"],
        "gradient_checkpointing": rc["gradient_checkpointing"],
        "peak_gpu_mem_allocated_gb": round(peak_alloc, 2),
        "peak_gpu_mem_reserved_gb": round(peak_res, 2),
        "train_loop_s": round(train_s, 1), "wall_s": round(wall_s, 1),
        "gpu_hours": round(wall_s / 3600, 3), "gpu": gpu_name(), "seed": SEED,
        "final_loss_last100": float(np.mean([r["loss"] for r in losses[-4:]])),
        "lora_dir": str(out),
    }
    write_csv(repo_dir("metrics") / f"loss_{run}.csv", losses)
    write_json(stats_path, stats)
    del unet, opt, lat_mean, lat_std, text_emb
    gc.collect()
    torch.cuda.empty_cache()
    return stats


if __name__ == "__main__":
    print(json.dumps(train(sys.argv[1]), indent=2))
