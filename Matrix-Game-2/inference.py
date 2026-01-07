# inference.py
#
# Baseline Matrix-Game-2 inference runner with disciplined profiling / benchmarking controls.
# - Keeps model behavior the same by default.
# - Adds optional bench mode (skip mp4 writing) and returns p50/p95 block latency breakdown.
# - Optional torch.profiler artifacts (chrome trace + top-op table).
#
# NOTE: This expects your pipeline.py CausalInferencePipeline.inference(...) to accept:
#   warmup_blocks, max_blocks, torch_profiler, return_metrics, profile_focus, profile_timesteps
# as in the patch I provided.

import os
import json
import argparse
import time
from typing import Any, Dict, List

import torch
import numpy as np

from omegaconf import OmegaConf
from torchvision.transforms import v2
from diffusers.utils import load_image
from einops import rearrange

from pipeline import CausalInferencePipeline
from wan.vae.wanx_vae import get_wanx_vae_wrapper
from demo_utils.vae_block3 import VAEDecoderWrapper
from utils.visualize import process_video
from utils.misc import set_seed
from utils.conditions import Bench_actions_universal, Bench_actions_gta_drive, Bench_actions_templerun
from utils.wan_wrapper import WanDiffusionWrapper
from safetensors.torch import load_file


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, default="configs/inference_yaml/inference_universal.yaml",
                        help="Path to the config file")
    parser.add_argument("--checkpoint_path", type=str, default="", help="Path to the checkpoint")
    parser.add_argument("--img_path", type=str, default="demo_images/universal/0000.png", help="Path to the image")
    parser.add_argument("--output_folder", type=str, default="outputs/", help="Output folder")
    parser.add_argument("--num_output_frames", type=int, default=150, help="Number of output latent frames")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--pretrained_model_path", type=str, default="Matrix-Game-2.0", help="Path to model folder")

    # ---- benchmarking/profiling knobs ----
    parser.add_argument("--bench", action="store_true",
                        help="Benchmark run: skip mp4 writing and save metrics JSON.")
    parser.add_argument("--warmup_blocks", type=int, default=3,
                        help="Blocks to run but exclude from metrics.")
    parser.add_argument("--max_blocks", type=int, default=0,
                        help="Max blocks to run (0 => run all blocks).")
    parser.add_argument("--profile_focus", type=str, default="all",
                        choices=["all", "denoise", "ctx", "decode"],
                        help="Which parts to time/execute for metrics focus.")
    parser.add_argument("--profile_timesteps", type=int, default=0,
                        help="If >0, run only first N denoise steps per block (fast profiling).")

    # torch profiler artifacts
    parser.add_argument("--torch_profile", action="store_true",
                        help="Enable torch.profiler and write chrome trace + summary table.")
    parser.add_argument("--torch_profile_dir", type=str, default="prof_runs",
                        help="Where to write trace.json and profiler_summary.txt.")
    parser.add_argument("--prof_row_limit", type=int, default=60,
                        help="Rows in profiler summary table.")
    parser.add_argument("--metrics_path", type=str, default="",
                        help="Optional explicit metrics json path. Default: <output_folder>/bench_stats.json")

    return parser.parse_args()


def _percentile(xs: List[float], p: float) -> float:
    if not xs:
        return float("nan")
    return float(np.percentile(np.asarray(xs, dtype=np.float64), p))


def _summarize_metrics(metrics: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {"raw": {}}

    def add_summary(key: str):
        xs = metrics.get(key, [])
        out[key] = {
            "count": len(xs),
            "p50": _percentile(xs, 50),
            "p95": _percentile(xs, 95),
            "mean": float(np.mean(xs)) if xs else float("nan"),
        }

    add_summary("block_ms_total")
    add_summary("block_ms_denoise")
    add_summary("block_ms_ctx")
    add_summary("block_ms_decode")
    add_summary("fps")

    frames_per_block = int(metrics.get("frames_per_block", 1))
    out["frames_per_block"] = frames_per_block

    # FPS derived from total p50 if user ran focus=all|decode
    p50_total = out["block_ms_total"]["p50"]
    out["fps_p50_total_derived"] = float(frames_per_block * 1000.0 / p50_total) if p50_total and p50_total > 0 else float("nan")

    out["peak_mem_bytes"] = int(metrics.get("peak_mem_bytes", 0))
    out["peak_mem_gb"] = float(out["peak_mem_bytes"]) / (1024**3) if out["peak_mem_bytes"] else 0.0
    out["notes"] = metrics.get("notes", {})
    return out


class InteractiveGameInference:
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda")
        self.weight_dtype = torch.bfloat16

        self._init_config()
        self._init_models()

        self.frame_process = v2.Compose([
            v2.Resize(size=(352, 640), antialias=True),
            v2.ToTensor(),
            v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

    def _init_config(self):
        self.config = OmegaConf.load(self.args.config_path)

    def _init_models(self):
        # Initialize pipeline generator
        generator = WanDiffusionWrapper(**getattr(self.config, "model_kwargs", {}), is_causal=True)

        # VAE decoder (decode side)
        current_vae_decoder = VAEDecoderWrapper()
        vae_state_dict = torch.load(
            os.path.join(self.args.pretrained_model_path, "Wan2.1_VAE.pth"),
            map_location="cpu"
        )
        decoder_state_dict = {}
        for key, value in vae_state_dict.items():
            if "decoder." in key or "conv2" in key:
                decoder_state_dict[key] = value

        current_vae_decoder.load_state_dict(decoder_state_dict)
        current_vae_decoder.to(self.device, torch.float16)
        current_vae_decoder.requires_grad_(False)
        current_vae_decoder.eval()

        # keep baseline compile choice
        current_vae_decoder.compile(mode="max-autotune-no-cudagraphs")

        pipeline = CausalInferencePipeline(self.config, generator=generator, vae_decoder=current_vae_decoder)
        if self.args.checkpoint_path:
            print("Loading Pretrained Model...")
            state_dict = load_file(self.args.checkpoint_path)
            pipeline.generator.load_state_dict(state_dict)

        self.pipeline = pipeline.to(device=self.device, dtype=self.weight_dtype)
        self.pipeline.vae_decoder.to(torch.float16)

        # VAE encoder wrapper (conditioning side)
        vae = get_wanx_vae_wrapper(self.args.pretrained_model_path, torch.float16)
        vae.requires_grad_(False)
        vae.eval()
        self.vae = vae.to(self.device, self.weight_dtype)

    def _resizecrop(self, image, th, tw):
        w, h = image.size
        if h / w > th / tw:
            new_w = int(w)
            new_h = int(new_w * th / tw)
        else:
            new_h = int(h)
            new_w = int(new_h * tw / th)
        left = (w - new_w) / 2
        top = (h - new_h) / 2
        right = (w + new_w) / 2
        bottom = (h + new_h) / 2
        return image.crop((left, top, right, bottom))

    def generate_videos(self):
        # Use config's mode (and pop to match original behavior if it was a DictConfig)
        mode = self.config.pop("mode") if "mode" in self.config else getattr(self.config, "mode")
        assert mode in ["universal", "gta_drive", "templerun"]

        # --------------------------
        # Conditioning setup (NOT target for optimization per assignment)
        # --------------------------
        image_pil = load_image(self.args.img_path)
        image_pil = self._resizecrop(image_pil, 352, 640)
        image = self.frame_process(image_pil)[None, :, None, :, :].to(dtype=self.weight_dtype, device=self.device)

        padding_video = torch.zeros_like(image).repeat(1, 1, 4 * (self.args.num_output_frames - 1), 1, 1)
        img_cond = torch.concat([image, padding_video], dim=2)

        tiler_kwargs = {"tiled": True, "tile_size": [44, 80], "tile_stride": [23, 38]}
        img_cond = self.vae.encode(img_cond, device=self.device, **tiler_kwargs).to(self.device)

        mask_cond = torch.ones_like(img_cond)
        mask_cond[:, :, 1:] = 0
        cond_concat = torch.cat([mask_cond[:, :4], img_cond], dim=1)

        visual_context = self.vae.clip.encode_video(image)

        sampled_noise = torch.randn(
            [1, 16, self.args.num_output_frames, 44, 80],
            device=self.device,
            dtype=self.weight_dtype,
        )

        num_frames = (self.args.num_output_frames - 1) * 4 + 1
        conditional_dict: Dict[str, torch.Tensor] = {
            "cond_concat": cond_concat.to(device=self.device, dtype=self.weight_dtype),
            "visual_context": visual_context.to(device=self.device, dtype=self.weight_dtype),
        }

        mouse_condition = None
        if mode == "universal":
            cond_data = Bench_actions_universal(num_frames)
            mouse_condition = cond_data["mouse_condition"].unsqueeze(0).to(device=self.device, dtype=self.weight_dtype)
            conditional_dict["mouse_cond"] = mouse_condition
        elif mode == "gta_drive":
            cond_data = Bench_actions_gta_drive(num_frames)
            mouse_condition = cond_data["mouse_condition"].unsqueeze(0).to(device=self.device, dtype=self.weight_dtype)
            conditional_dict["mouse_cond"] = mouse_condition
        else:
            cond_data = Bench_actions_templerun(num_frames)

        keyboard_condition = cond_data["keyboard_condition"].unsqueeze(0).to(device=self.device, dtype=self.weight_dtype)
        conditional_dict["keyboard_cond"] = keyboard_condition

        # --------------------------
        # Timed inference
        # --------------------------
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

        metrics = None
        videos = None

        t0 = time.perf_counter()
        if self.args.torch_profile:
            from torch.profiler import profile as torch_profile
            from torch.profiler import ProfilerActivity, schedule

            os.makedirs(self.args.torch_profile_dir, exist_ok=True)
            trace_path = os.path.join(self.args.torch_profile_dir, "trace.json")
            summary_path = os.path.join(self.args.torch_profile_dir, "profiler_summary.txt")

            # --- Phase 0: warm compile / autotune OUTSIDE the profiler ---
            # This minimizes compile noise in the exported trace.
            warm_compile_blocks = int(self.args.warmup_blocks) if int(self.args.warmup_blocks) > 0 else 2
            warm_compile_total = warm_compile_blocks + 1  # a little extra
            with torch.no_grad():
                _ = self.pipeline.inference(
                    noise=sampled_noise,
                    conditional_dict=conditional_dict,
                    return_latents=False,
                    mode=mode,
                    profile=False,
                    warmup_blocks=0,
                    max_blocks=warm_compile_total,
                    torch_profiler=None,
                    return_metrics=False,
                    profile_focus=self.args.profile_focus,
                    profile_timesteps=int(self.args.profile_timesteps),
                )
            torch.cuda.synchronize()

            # --- Phase 1: capture ONLY steady blocks using a schedule ---
            wait = int(self.args.warmup_blocks)
            active = int(self.args.max_blocks) if int(self.args.max_blocks) > 0 else 4
            total_blocks = wait + active

            with torch_profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                record_shapes=True,
                profile_memory=True,
                with_stack=False,
                schedule=schedule(wait=wait, warmup=0, active=active, repeat=1),
            ) as prof:
                with torch.no_grad():
                    videos, metrics = self.pipeline.inference(
                        noise=sampled_noise,
                        conditional_dict=conditional_dict,
                        return_latents=False,
                        mode=mode,
                        profile=False,  # avoid per-block prints during profiler capture
                        warmup_blocks=wait,          # schedule handles "warmup exclusion"
                        max_blocks=total_blocks,  # run enough blocks to cover wait+active
                        torch_profiler=prof,
                        return_metrics=True,
                        profile_focus=self.args.profile_focus,
                        profile_timesteps=int(self.args.profile_timesteps),
                    )
                torch.cuda.synchronize()

            prof.export_chrome_trace(trace_path)
            with open(summary_path, "w") as f:
                f.write(
                    prof.key_averages().table(
                        sort_by="self_cuda_time_total",
                        row_limit=int(self.args.prof_row_limit),
                    )
                )

            print(f"[torch.profiler] wrote:\n  {trace_path}\n  {summary_path}")
        else:
            with torch.no_grad():
                videos, metrics = self.pipeline.inference(
                    noise=sampled_noise,
                    conditional_dict=conditional_dict,
                    return_latents=False,
                    mode=mode,
                    profile=True,  # prints per-block breakdown after warmup
                    warmup_blocks=int(self.args.warmup_blocks),
                    max_blocks=int(self.args.max_blocks),
                    torch_profiler=None,
                    return_metrics=True,
                    profile_focus=self.args.profile_focus,
                    profile_timesteps=int(self.args.profile_timesteps),
                )
            torch.cuda.synchronize()

        t1 = time.perf_counter()
        wall_s = t1 - t0
        if metrics is None:
            metrics = {}
        metrics["notes"] = dict(metrics.get("notes", {}))
        metrics["notes"]["wall_seconds"] = float(wall_s)

        summary = _summarize_metrics(metrics)

        metrics_path = self.args.metrics_path or os.path.join(self.args.output_folder, "bench_stats.json")
        os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
        with open(metrics_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"[metrics] wrote {metrics_path}")
        print(json.dumps(summary, indent=2))

        if self.args.bench:
            print("[bench] done (skipped video writing).")
            return

        # --------------------------
        # Original mp4 writing (kept; not used for perf)
        # --------------------------
        assert videos is not None and len(videos) > 0, "No videos returned; did you run profile_focus=denoise/ctx?"
        videos_tensor = torch.cat(videos, dim=1)
        videos_np = rearrange(videos_tensor, "B T C H W -> B T H W C")
        videos_np = ((videos_np.float() + 1) * 127.5).clip(0, 255).cpu().numpy().astype(np.uint8)[0]
        video = np.ascontiguousarray(videos_np)

        mouse_icon = "assets/images/mouse.png"
        if mode != "templerun":
            assert mouse_condition is not None
            config = (
                keyboard_condition[0].float().cpu().numpy(),
                mouse_condition[0].float().cpu().numpy(),
            )
        else:
            config = (keyboard_condition[0].float().cpu().numpy(),)

        out_demo = os.path.join(self.args.output_folder, "demo.mp4")
        out_demo_icon = os.path.join(self.args.output_folder, "demo_icon.mp4")

        process_video(video.astype(np.uint8), out_demo, config, mouse_icon, mouse_scale=0.1, process_icon=False, mode=mode)
        process_video(video.astype(np.uint8), out_demo_icon, config, mouse_icon, mouse_scale=0.1, process_icon=True, mode=mode)
        print("Done")


def main():
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.output_folder, exist_ok=True)
    runner = InteractiveGameInference(args)
    runner.generate_videos()


if __name__ == "__main__":
    main()


