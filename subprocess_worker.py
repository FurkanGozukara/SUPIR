#!/usr/bin/env python
"""
Subprocess Worker for SUPIR Processing

This script runs as a separate process to handle LLaVA and SUPIR processing.
When the process exits, all GPU memory is automatically freed.

Usage:
    python subprocess_worker.py --params <params.json> --progress <progress.json> --results <results.json>
"""

import argparse
import gc
import json
import os
import sys
import time
import traceback
from typing import List, Dict, Any

# Add the current directory to the path for imports
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)


def write_progress(progress_file: str, data: Dict[str, Any]):
    """Write progress data to file atomically."""
    temp_file = progress_file + ".tmp"
    try:
        with open(temp_file, 'w', encoding='utf-8') as f:
            json.dump(data, f)
        # Atomic rename
        if os.path.exists(progress_file):
            os.remove(progress_file)
        os.rename(temp_file, progress_file)
    except Exception as e:
        print(f"Error writing progress: {e}")


def write_results(results_file: str, data: Dict[str, Any]):
    """Write results data to file."""
    try:
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(data, f)
    except Exception as e:
        print(f"Error writing results: {e}")


class SubprocessProgressTracker:
    """Tracks progress and writes to file for main process to read."""
    
    def __init__(self, progress_file: str):
        self.progress_file = progress_file
        self.current_step = 0
        self.total_steps = 1
        self.description = "Initializing..."
        self.batch_processed = 0
        self.batch_total = 0
        self.is_cancelled = False
        
    def update(self, step: int = None, total: int = None, desc: str = None, 
               batch_processed: int = None, batch_total: int = None):
        """Update progress and write to file."""
        if step is not None:
            self.current_step = step
        if total is not None:
            self.total_steps = total
        if desc is not None:
            self.description = desc
        if batch_processed is not None:
            self.batch_processed = batch_processed
        if batch_total is not None:
            self.batch_total = batch_total
            
        progress_data = {
            "step": self.current_step,
            "total": self.total_steps,
            "description": self.description,
            "batch_processed": self.batch_processed,
            "batch_total": self.batch_total,
            "progress_percent": (self.current_step / self.total_steps * 100) if self.total_steps > 0 else 0,
            "timestamp": time.time(),
            "completed": False,
            "error": None
        }
        write_progress(self.progress_file, progress_data)
        
    def complete(self, error: str = None):
        """Mark processing as complete."""
        progress_data = {
            "step": self.total_steps,
            "total": self.total_steps,
            "description": "Complete" if error is None else f"Error: {error}",
            "batch_processed": self.batch_processed,
            "batch_total": self.batch_total,
            "progress_percent": 100 if error is None else self.current_step / self.total_steps * 100,
            "timestamp": time.time(),
            "completed": True,
            "error": error
        }
        write_progress(self.progress_file, progress_data)


def run_llava_processing(params: Dict[str, Any], image_paths: List[str], 
                         progress_tracker: SubprocessProgressTracker) -> Dict[str, str]:
    """
    Run LLaVA captioning on images.
    Returns a dict mapping image_path -> caption
    """
    from llava.llava_agent import LLavaAgent
    from SUPIR.utils.model_fetch import get_model
    from SUPIR.util import HWC3
    from PIL import Image
    import numpy as np
    import torch
    
    captions = {}
    
    # Determine device
    if torch.cuda.device_count() >= 2:
        llava_device = 'cuda:1'
    elif torch.cuda.device_count() == 1:
        llava_device = 'cuda:0'
    else:
        llava_device = 'cpu'
    
    progress_tracker.update(desc="Loading LLaVA model...")
    
    # Load LLaVA
    llava_path = get_model('liuhaotian/llava-v1.5-7b')
    load_8bit = params.get('load_8bit_llava', False)
    load_4bit = params.get('load_4bit_llava', True)
    llava_agent = LLavaAgent(llava_path, device=llava_device, 
                             load_8bit=load_8bit, load_4bit=load_4bit)
    
    temperature = float(params.get('temperature', 0.2))
    top_p = float(params.get('top_p', 0.7))
    question = params.get('qs', 'Describe this image and its style in a very detailed manner.')
    save_captions = params.get('save_captions', False)
    skip_if_txt_exists = params.get('skip_llava_if_txt_exists', True)
    
    total_images = len(image_paths)
    
    for idx, image_path in enumerate(image_paths):
        progress_tracker.update(
            step=idx + 1, 
            total=total_images,
            desc=f"LLaVA: Processing image {idx + 1}/{total_images}",
            batch_processed=idx,
            batch_total=total_images
        )
        
        # Check if txt file exists and should skip
        txt_path = os.path.splitext(image_path)[0] + ".txt"
        if skip_if_txt_exists and os.path.exists(txt_path):
            print(f"Found {txt_path}, skipping LLaVA")
            with open(txt_path, 'r', encoding='utf-8') as f:
                caption = f.read().strip()
            captions[image_path] = caption
            continue
        
        try:
            img = Image.open(image_path)
            img_array = np.array(img)
            lq = HWC3(img_array)
            lq_img = Image.fromarray(lq.astype('uint8'))
            
            caption = llava_agent.gen_image_caption([lq_img], temperature=temperature, 
                                                     top_p=top_p, qs=question)
            caption = caption[0]
            captions[image_path] = caption
            
            if save_captions:
                with open(txt_path, 'w', encoding='utf-8') as f:
                    f.write(caption)
                    
        except Exception as e:
            print(f"Error processing {image_path} with LLaVA: {e}")
            captions[image_path] = params.get('main_prompt', '')
    
    # Cleanup LLaVA
    del llava_agent
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return captions


def run_supir_processing(params: Dict[str, Any], image_paths: List[str], 
                         captions: Dict[str, str], progress_tracker: SubprocessProgressTracker,
                         outputs_folder: str) -> List[Dict[str, Any]]:
    """
    Run SUPIR upscaling on images.
    Returns list of result info dicts.
    """
    import torch
    import numpy as np
    import einops
    from PIL import Image, PngImagePlugin
    
    from SUPIR.util import HWC3, upscale_image, convert_dtype, create_SUPIR_model
    from SUPIR.utils.face_restoration_helper import FaceRestoreHelper
    try:
        from SUPIR.utils.rename_meta import rename_meta_key
    except ImportError:
        # Fallback if rename_meta_key is not available
        def rename_meta_key(key):
            return key
    from SUPIR.utils import shared
    
    # Set shared options
    shared.opts.half_mode = params.get('loading_half_params', False)
    shared.opts.fast_load_sd = params.get('fast_load_sd', False)
    if params.get('fp8', False):
        shared.opts.half_mode = True
        shared.opts.fp8_storage = True
    
    results = []
    
    # Determine device
    if torch.cuda.device_count() >= 1:
        supir_device = 'cuda:0'
    else:
        supir_device = 'cpu'
    
    bf16_supported = torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False
    
    # Get processing parameters
    model_select = params.get('model_select', 'v0-Q')
    ckpt_select = params.get('ckpt_select', '')
    sampler = params.get('sampler', 'EDM')
    diff_dtype = params.get('diff_dtype', 'bf16')
    ae_dtype = params.get('ae_dtype', 'bf16')
    use_tile_vae = params.get('use_tile_vae', False)
    
    # Build sampler class name
    tiled = "TiledRestore" if use_tile_vae else "Restore"
    sampler_cls = f"sgm.modules.diffusionmodules.sampling.{tiled}{sampler}Sampler"
    
    # Prepare LoRA configs
    lora_configs = []
    lora_dir = params.get('lora_dir', 'models/Lora')
    for i in range(1, 5):
        lora_name = params.get(f'lora_{i}')
        lora_weight = params.get(f'lora_{i}_weight', 1.0)
        if lora_name and lora_name != "None":
            lora_path = os.path.join(lora_dir, lora_name)
            if os.path.exists(lora_path):
                lora_configs.append((lora_path, float(lora_weight)))
    
    progress_tracker.update(desc="Loading SUPIR model...")
    
    # Resolve checkpoint path
    ckpt_dir = params.get('ckpt_dir', 'models/checkpoints')
    if not os.path.isabs(ckpt_select):
        ckpt_path = os.path.join(ckpt_dir, ckpt_select)
        if not os.path.exists(ckpt_path):
            ckpt_path = ckpt_select
    else:
        ckpt_path = ckpt_select
    
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    
    # Load model
    model_cfg = "options/SUPIR_v0_tiled.yaml" if use_tile_vae else "options/SUPIR_v0.yaml"
    weight_dtype = 'fp16' if not bf16_supported else diff_dtype
    
    model = create_SUPIR_model(model_cfg, weight_dtype, supir_sign=model_select[-1], 
                               device=supir_device, ckpt=ckpt_path,
                               sampler=sampler_cls, 
                               lora_configs=lora_configs if lora_configs else None)
    
    if use_tile_vae:
        encoder_tile_size = params.get('encoder_tile_size', 512)
        decoder_tile_size = params.get('decoder_tile_size', 64)
        use_fast_tile = params.get('use_fast_tile', False)
        model.init_tile_vae(encoder_tile_size=encoder_tile_size, 
                           decoder_tile_size=decoder_tile_size, 
                           use_fast=use_fast_tile)
    
    model.ae_dtype = convert_dtype('fp32' if not bf16_supported else ae_dtype)
    model.model.dtype = convert_dtype('fp16' if not bf16_supported else diff_dtype)
    
    # Face helper (lazy load)
    face_helper = None
    apply_face = params.get('apply_face', False)
    apply_face_only = params.get('apply_face_only', False)
    apply_bg = params.get('apply_bg', False)
    
    if apply_face or apply_face_only:
        face_helper = FaceRestoreHelper(
            device='cpu',
            upscale_factor=1,
            face_size=1024,
            use_parse=True,
            det_model='retinaface_resnet50'
        )
    
    # Processing parameters
    upscale = float(params.get('upscale', 1))
    max_megapixels = float(params.get('max_megapixels', 0))
    max_resolution = float(params.get('max_resolution', 0))
    first_downscale = params.get('first_downscale', False)
    
    num_images = int(params.get('num_images', 1))
    num_samples = int(params.get('num_samples', 1))
    edm_steps = int(params.get('edm_steps', 50))
    s_stage1 = float(params.get('s_stage1', -1.0))
    s_stage2 = float(params.get('s_stage2', 1.0))
    s_cfg = float(params.get('s_cfg', 3.0))
    seed = int(params.get('seed', 0))
    s_churn = float(params.get('s_churn', 5))
    s_noise = float(params.get('s_noise', 1.003))
    color_fix_type = params.get('color_fix_type', 'Wavelet')
    linear_cfg = params.get('linear_CFG', True)
    linear_s_stage2 = params.get('linear_s_stage2', False)
    spt_linear_cfg = float(params.get('spt_linear_CFG', 4.0))
    spt_linear_s_stage2 = float(params.get('spt_linear_s_stage2', 0))
    
    a_prompt = params.get('a_prompt', '')
    n_prompt = params.get('n_prompt', '')
    random_seed = params.get('random_seed', True)
    face_resolution = int(params.get('face_resolution', 1024))
    face_prompt = params.get('face_prompt', '')
    
    filename_prefix = params.get('filename_prefix', '')
    filename_suffix = params.get('filename_suffix', '')
    
    # Sanitize filename parts
    def sanitize_filename_part(text):
        if not text:
            return ""
        invalid_chars = '<>:"/\\|?*\x00\r\n\t'
        sanitized = text
        for char in invalid_chars:
            sanitized = sanitized.replace(char, '_')
        sanitized = ''.join(char if ord(char) > 31 and ord(char) != 127 else '_' for char in sanitized)
        sanitized = sanitized.strip(' .')
        return sanitized[:50] if len(sanitized) > 50 else sanitized
    
    filename_prefix = sanitize_filename_part(filename_prefix)
    filename_suffix = sanitize_filename_part(filename_suffix)
    
    # Create output directory
    os.makedirs(outputs_folder, exist_ok=True)
    metadata_dir = os.path.join(outputs_folder, "images_meta_data")
    os.makedirs(metadata_dir, exist_ok=True)
    
    total_images = len(image_paths)
    
    for img_idx, image_path in enumerate(image_paths):
        progress_tracker.update(
            step=img_idx + 1,
            total=total_images,
            desc=f"SUPIR: Processing image {img_idx + 1}/{total_images}",
            batch_processed=img_idx,
            batch_total=total_images
        )
        
        try:
            img = Image.open(image_path)
            img_array = np.array(img)
            img_prompt = captions.get(image_path, params.get('main_prompt', ''))
            
            # Prepare image
            img_processed = HWC3(img_array)
            h, w, _ = img_processed.shape
            
            # Calculate target dimensions
            target_h = float(h) * upscale
            target_w = float(w) * upscale
            
            if min(target_h, target_w) < 1024:
                min_scale = 1024 / min(target_h, target_w)
                target_h *= min_scale
                target_w *= min_scale
            
            if max_megapixels > 0:
                target_mp = (target_h * target_w) / 1_000_000
                if target_mp > max_megapixels:
                    mp_scale = (max_megapixels * 1_000_000 / (target_h * target_w)) ** 0.5
                    target_h *= mp_scale
                    target_w *= mp_scale
            
            if max_resolution > 0:
                if max(target_h, target_w) > max_resolution:
                    if target_w > target_h:
                        res_scale = max_resolution / target_w
                    else:
                        res_scale = max_resolution / target_h
                    target_h *= res_scale
                    target_w *= res_scale
            
            final_upscale = min(target_h / h, target_w / w)
            
            if first_downscale:
                final_target_h = int(target_h)
                final_target_w = int(target_w)
                downscale_target_h = int(final_target_h / upscale)
                downscale_target_w = int(final_target_w / upscale)
                img_pil = Image.fromarray(img_processed.astype('uint8'))
                img_pil_downscaled = img_pil.resize((downscale_target_w, downscale_target_h), Image.LANCZOS)
                img_processed = np.array(img_pil_downscaled)
                final_upscale = float(upscale)
            
            img_upscaled = upscale_image(img_processed, final_upscale, unit_resolution=32, min_size=1024)
            
            lq = np.array(img_upscaled)
            lq_tensor = lq / 255 * 2 - 1
            lq_tensor = torch.tensor(lq_tensor, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(supir_device)[:, :3, :, :]
            
            # Process faces if needed
            face_gallery = []
            if apply_face or apply_face_only:
                if face_helper is not None:
                    if apply_face_only:
                        face_helper.upscale_factor = 1
                    else:
                        face_helper.upscale_factor = final_upscale
                    
                    face_helper.clean_all()
                    face_helper.read_image(lq)
                    face_helper.get_face_landmarks_5(only_center_face=False, resize=640, eye_dist_threshold=5)
                    face_helper.align_warp_face()
                    
                    face_helper.to(supir_device)
                    
                    face_caption = face_prompt if len(face_prompt) > 1 else img_prompt
                    
                    for face in face_helper.cropped_faces:
                        face_tensor = np.array(face) / 255 * 2 - 1
                        face_tensor = torch.tensor(face_tensor, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(supir_device)[:, :3, :, :]
                        
                        face_samples = model.batchify_sample(
                            face_tensor, [face_caption], num_steps=edm_steps, 
                            restoration_scale=s_stage1, s_churn=s_churn, s_noise=s_noise,
                            cfg_scale=s_cfg, control_scale=s_stage2, seed=seed,
                            num_samples=num_samples, p_p=a_prompt, n_p=n_prompt,
                            color_fix_type=color_fix_type, use_linear_cfg=linear_cfg,
                            use_linear_control_scale=linear_s_stage2,
                            cfg_scale_start=spt_linear_cfg, control_scale_start=spt_linear_s_stage2
                        )
                        
                        if face_samples is not None:
                            if face_resolution < 1024:
                                face_samples = face_samples[:, :, 512 - face_resolution // 2:512 + face_resolution // 2,
                                              512 - face_resolution // 2:512 + face_resolution // 2]
                            
                            face_samples = torch.nn.functional.interpolate(
                                face_samples, size=face_helper.face_size, 
                                mode='bilinear', align_corners=False
                            )
                            x_face = (einops.rearrange(face_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().round().clip(0, 255).astype(np.uint8)
                            face_helper.add_restored_face(x_face[0])
                            face_gallery.append(x_face[0])
            
            # Process background or full image
            result = None
            for img_num in range(num_images):
                current_seed = seed
                if random_seed or num_images > 1:
                    current_seed = np.random.randint(0, 2147483647)
                
                if apply_bg:
                    samples = model.batchify_sample(
                        lq_tensor, [img_prompt], num_steps=edm_steps,
                        restoration_scale=s_stage1, s_churn=s_churn, s_noise=s_noise,
                        cfg_scale=s_cfg, control_scale=s_stage2, seed=current_seed,
                        num_samples=num_samples, p_p=a_prompt, n_p=n_prompt,
                        color_fix_type=color_fix_type, use_linear_cfg=linear_cfg,
                        use_linear_control_scale=linear_s_stage2,
                        cfg_scale_start=spt_linear_cfg, control_scale_start=spt_linear_s_stage2
                    )
                    
                    if samples is not None:
                        bg = (einops.rearrange(samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().round().clip(0, 255).astype(np.uint8)
                        if apply_face and face_helper is not None and len(face_helper.restored_faces) > 0:
                            face_helper.get_inverse_affine(None)
                            result = face_helper.paste_faces_to_input_image(upsample_img=bg[0])
                        else:
                            result = bg[0]
                            
                elif apply_face_only and face_helper is not None and len(face_helper.restored_faces) > 0:
                    face_helper.get_inverse_affine(None)
                    result = face_helper.paste_faces_to_input_image()
                    
                elif apply_face and not apply_bg and face_helper is not None and len(face_helper.restored_faces) > 0:
                    face_helper.upscale_factor = final_upscale
                    face_helper.get_inverse_affine(None)
                    result = face_helper.paste_faces_to_input_image()
                    
                else:
                    # Standard SUPIR upscale
                    samples = model.batchify_sample(
                        lq_tensor, [img_prompt], num_steps=edm_steps,
                        restoration_scale=s_stage1, s_churn=s_churn, s_noise=s_noise,
                        cfg_scale=s_cfg, control_scale=s_stage2, seed=current_seed,
                        num_samples=num_samples, p_p=a_prompt, n_p=n_prompt,
                        color_fix_type=color_fix_type, use_linear_cfg=linear_cfg,
                        use_linear_control_scale=linear_s_stage2,
                        cfg_scale_start=spt_linear_cfg, control_scale_start=spt_linear_s_stage2
                    )
                    
                    if samples is not None:
                        x_samples = (einops.rearrange(samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().round().clip(0, 255).astype(np.uint8)
                        result = x_samples[0]
                
                if result is not None:
                    # Save result
                    base_filename = os.path.splitext(os.path.basename(image_path))[0]
                    if len(base_filename) > 250:
                        base_filename = base_filename[:250]
                    
                    final_filename = f'{filename_prefix}{base_filename}{filename_suffix}'
                    save_path = os.path.join(outputs_folder, f'{final_filename}.png')
                    
                    index = 1
                    while os.path.exists(save_path):
                        save_path = os.path.join(outputs_folder, f'{final_filename}_{str(index).zfill(4)}.png')
                        index += 1
                    
                    result_img = Image.fromarray(result)
                    
                    # Add metadata
                    meta = PngImagePlugin.PngInfo()
                    meta_dict = {
                        'seed': current_seed,
                        'upscale': upscale,
                        'edm_steps': edm_steps,
                        's_cfg': s_cfg,
                        'model_select': model_select,
                        'ckpt_select': os.path.basename(ckpt_select),
                        'caption': img_prompt
                    }
                    for key, value in meta_dict.items():
                        try:
                            meta.add_text(rename_meta_key(key), str(value))
                        except:
                            meta.add_text(key, str(value))
                    
                    result_img.save(save_path, "PNG", pnginfo=meta)
                    
                    results.append({
                        'input_path': image_path,
                        'output_path': save_path,
                        'seed': current_seed,
                        'caption': img_prompt
                    })
                    
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            traceback.print_exc()
    
    # Cleanup
    del model
    if face_helper is not None:
        del face_helper
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return results


def main():
    parser = argparse.ArgumentParser(description='SUPIR Subprocess Worker')
    parser.add_argument('--params', required=True, help='Path to parameters JSON file')
    parser.add_argument('--progress', required=True, help='Path to progress JSON file')
    parser.add_argument('--results', required=True, help='Path to results JSON file')
    args = parser.parse_args()
    
    print(f"Subprocess worker started with:")
    print(f"  Params: {args.params}")
    print(f"  Progress: {args.progress}")
    print(f"  Results: {args.results}")
    
    progress_tracker = SubprocessProgressTracker(args.progress)
    
    try:
        # Read parameters
        progress_tracker.update(desc="Reading parameters...")
        with open(args.params, 'r', encoding='utf-8') as f:
            params = json.load(f)
        
        # Get image paths
        image_paths = params.get('image_paths', [])
        if not image_paths:
            raise ValueError("No image paths provided")
        
        outputs_folder = params.get('outputs_folder', 'outputs')
        apply_llava = params.get('apply_llava', False)
        apply_supir = params.get('apply_supir', True)
        
        # Initialize captions dict
        captions = {}
        main_prompt = params.get('main_prompt', '')
        for path in image_paths:
            captions[path] = main_prompt
        
        total_steps = 0
        if apply_llava:
            total_steps += len(image_paths) + 1  # +1 for loading
        if apply_supir:
            total_steps += len(image_paths) + 1  # +1 for loading
        
        progress_tracker.update(total=total_steps, batch_total=len(image_paths))
        
        # Run LLaVA if enabled
        if apply_llava:
            progress_tracker.update(desc="Starting LLaVA processing...")
            captions = run_llava_processing(params, image_paths, progress_tracker)
        
        # Run SUPIR if enabled
        results = []
        if apply_supir:
            progress_tracker.update(desc="Starting SUPIR processing...")
            results = run_supir_processing(params, image_paths, captions, 
                                          progress_tracker, outputs_folder)
        
        # Write results
        result_data = {
            'success': True,
            'results': results,
            'captions': captions,
            'total_processed': len(results),
            'error': None
        }
        write_results(args.results, result_data)
        
        progress_tracker.complete()
        print("Subprocess worker completed successfully")
        
    except Exception as e:
        error_msg = str(e)
        traceback.print_exc()
        
        result_data = {
            'success': False,
            'results': [],
            'captions': {},
            'total_processed': 0,
            'error': error_msg
        }
        write_results(args.results, result_data)
        progress_tracker.complete(error=error_msg)
        sys.exit(1)


if __name__ == '__main__':
    main()

