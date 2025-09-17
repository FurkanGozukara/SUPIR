import os

import cv2
import numpy as np
import torch
from PIL import Image, ImageFile
from omegaconf import OmegaConf

# Enable loading of truncated images automatically
ImageFile.LOAD_TRUNCATED_IMAGES = True
from torch.nn.functional import interpolate

from SUPIR.utils import models_utils
from SUPIR.utils.devices import torch_gc
from sgm.util import instantiate_from_config
from ui_helpers import printt
from SUPIR.utils import models_utils, sd_model_initialization, shared

def get_state_dict(d):
    return d.get('state_dict', d)


def load_state_dict(ckpt_path, location='cpu'):
    _, extension = os.path.splitext(ckpt_path)
    if extension.lower() == ".safetensors":
        import safetensors.torch
        state_dict = safetensors.torch.load_file(ckpt_path, device=location)
    else:
        state_dict = get_state_dict(torch.load(ckpt_path, map_location=torch.device(location)))
    state_dict = get_state_dict(state_dict)    
    return state_dict


def apply_lora_to_model(model, lora_path, alpha=1.0):
    """
    Apply a single LoRA to the model with the given alpha/weight.
    This modifies the model weights in-place by merging the LoRA.
    """
    if not lora_path or not os.path.exists(lora_path):
        return

    printt(f'Applying LoRA from [{lora_path}] with weight {alpha}')

    try:
        import safetensors.torch
        lora_sd = safetensors.torch.load_file(lora_path, device='cpu')
    except:
        printt(f'Failed to load LoRA from [{lora_path}]')
        return

    # Get all LoRA keys and find the unique layer names
    lora_layers = {}
    for key in lora_sd.keys():
        if 'lora_up' in key:
            layer_name = key.replace('.lora_up.weight', '').replace('lora_unet_', '').replace('lora_te_', '')
            up_key = key
            down_key = key.replace('lora_up', 'lora_down')

            if down_key in lora_sd:
                lora_layers[layer_name] = {
                    'up': lora_sd[up_key],
                    'down': lora_sd[down_key]
                }

    # Apply LoRA weights to the model
    applied_count = 0
    for name, param in model.model.named_parameters():
        # Convert parameter name to match LoRA naming convention
        lora_name = name.replace('.', '_')

        # Check various LoRA naming patterns
        for lora_layer_name, lora_weights in lora_layers.items():
            if lora_layer_name in lora_name or lora_name in lora_layer_name:
                try:
                    # Apply LoRA: weight = weight + alpha * (up @ down)
                    up_weight = lora_weights['up'].to(param.device).to(param.dtype)
                    down_weight = lora_weights['down'].to(param.device).to(param.dtype)

                    # Handle different layer types
                    if len(param.shape) == 4:  # Conv2d weights: [out_channels, in_channels, H, W]
                        if len(down_weight.shape) == 4:
                            # Conv2d LoRA weights
                            # Reshape for matrix multiplication
                            out_ch, in_ch = param.shape[0], param.shape[1]
                            kernel_size = param.shape[2:]

                            # Reshape down: [rank, in_ch * k * k] -> [rank, in_ch]
                            down_weight = down_weight.flatten(1)
                            # Reshape up: [out_ch * k * k, rank] -> [out_ch, rank]
                            up_weight = up_weight.flatten(1)

                            # Compute LoRA weight
                            lora_weight = alpha * torch.mm(up_weight, down_weight)

                            # Reshape back to conv shape
                            lora_weight = lora_weight.reshape(param.shape)
                        else:
                            # Linear LoRA applied to Conv - skip this combination
                            continue

                    elif len(param.shape) == 2:  # Linear weights: [out_features, in_features]
                        if len(down_weight.shape) == 2:
                            # Linear LoRA weights - standard matrix multiplication
                            lora_weight = alpha * torch.mm(up_weight, down_weight)
                        else:
                            # Conv LoRA applied to Linear - skip this combination
                            continue

                    elif len(param.shape) == 1:  # Bias or norm parameters
                        # LoRA typically doesn't apply to bias/norm params
                        continue
                    else:
                        # Unknown shape - skip
                        continue

                    # Ensure shapes match before adding
                    if lora_weight.shape != param.shape:
                        printt(f'Shape mismatch for {name}: param {param.shape} vs lora {lora_weight.shape}, skipping')
                        continue

                    param.data += lora_weight
                    applied_count += 1
                    break

                except Exception as e:
                    printt(f'Error applying LoRA to {name}: {e}, skipping')
                    continue

    printt(f'Applied LoRA to {applied_count} layers')


def apply_multiple_loras(model, lora_configs):
    """
    Apply multiple LoRAs to the model.
    lora_configs: list of tuples (lora_path, weight)
    """
    for lora_path, weight in lora_configs:
        if lora_path and weight != 0:
            apply_lora_to_model(model, lora_path, weight)


def create_SUPIR_model(config_path, weight_dtype='bf16', supir_sign=None, device='cpu', ckpt=None, sampler="DPMPP2M", lora_configs=None):
    # Load the model configuration
    config = OmegaConf.load(config_path)
    config.model.params.sampler_config.target = sampler
    if ckpt:
        config.SDXL_CKPT = ckpt

    weight_dtype_conversion = {
        'first_stage_model': None,
        'alphas_cumprod': None,
        '': convert_dtype(weight_dtype),
    }
    # Instantiate model from config
    printt(f'Loading model from [{config_path}]')
    if shared.opts.fast_load_sd:
        with sd_model_initialization.DisableInitialization(disable_clip=False):
            with sd_model_initialization.InitializeOnMeta():
                model = instantiate_from_config(config.model)
    else:
        model = instantiate_from_config(config.model)

    printt(f'Loaded model from [{config_path}]')

    # Function to load state dict to the chosen device
    def load_to_device(checkpoint_path):
        printt(f'Loading state_dict from [{checkpoint_path}]')
        if checkpoint_path and os.path.exists(checkpoint_path):
            if torch.cuda.is_available():
                tgt_device = 'cuda'
            else:
                tgt_device = 'cpu'
            state_dict = load_state_dict(checkpoint_path, tgt_device)
            with sd_model_initialization.LoadStateDictOnMeta(state_dict, device=model.device, weight_dtype_conversion=weight_dtype_conversion):
                models_utils.load_model_weights(model, state_dict)
            torch_gc()
            printt(f'Loaded state_dict from [{checkpoint_path}]')
        else:
            printt(f'No checkpoint found at [{checkpoint_path}]')

    # Load state dicts as needed
    load_to_device(config.get('SDXL_CKPT'))

    # Handling SUPIR checkpoints based on the sign
    if supir_sign:
        assert supir_sign in ['F', 'Q'], "supir_sign must be either 'F' or 'Q'"
        ckpt_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models", f"v0{supir_sign}.ckpt"))
        load_to_device(ckpt_path)

    # Apply LoRAs if provided
    if lora_configs:
        printt(f'Applying {len(lora_configs)} LoRA(s)')
        apply_multiple_loras(model, lora_configs)

    model.sampler = sampler
    printt(f'Loaded model config from [{config_path}] and moved to {device}')

    return model


def PIL2Tensor(img, upscale=1, min_size=1024, do_fix_resize=None):
    """
    PIL.Image -> Tensor[C, H, W], RGB, [-1, 1]
    """
    # size
    w, h = img.size
    w *= upscale
    h *= upscale
    w0, h0 = round(w), round(h)
    if min(w, h) < min_size:
        _upscale = min_size / min(w, h)
        w *= _upscale
        h *= _upscale
    if do_fix_resize is not None:
        _upscale = do_fix_resize / min(w, h)
        w *= _upscale
        h *= _upscale
        w0, h0 = round(w), round(h)
    w = int(np.round(w / 64.0)) * 64
    h = int(np.round(h / 64.0)) * 64
    x = img.resize((w, h), Image.BICUBIC)
    x = np.array(x).round().clip(0, 255).astype(np.uint8)
    x = x / 255 * 2 - 1
    x = torch.tensor(x, dtype=torch.float32).permute(2, 0, 1)
    return x, h0, w0


def Tensor2PIL(x, h0, w0):
    """
    Tensor[C, H, W], RGB, [-1, 1] -> PIL.Image
    """
    x = x.unsqueeze(0)
    x = interpolate(x, size=(h0, w0), mode='bicubic')
    x = (x.squeeze(0).permute(1, 2, 0) * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)
    return Image.fromarray(x)


def HWC3(x):
    assert x.dtype == np.uint8
    if x.ndim == 2:
        x = x[:, :, None]
    assert x.ndim == 3
    h, w, c = x.shape
    assert c == 1 or c == 3 or c == 4
    if c == 3:
        return x
    if c == 1:
        return np.concatenate([x, x, x], axis=2)
    if c == 4:
        color = x[:, :, 0:3].astype(np.float32)
        alpha = x[:, :, 3:4].astype(np.float32) / 255.0
        y = color * alpha + 255.0 * (1.0 - alpha)
        y = y.clip(0, 255).astype(np.uint8)
        return y


def upscale_image(input_image, upscale, min_size=None, unit_resolution=64):
    h, w, c = input_image.shape
    h = float(h)
    w = float(w)
    h *= upscale
    w *= upscale
    if min_size is not None:
        if min(h, w) < min_size:
            _upscale = min_size / min(w, h)
            w *= _upscale
            h *= _upscale
    h = int(np.round(h / unit_resolution)) * unit_resolution
    w = int(np.round(w / unit_resolution)) * unit_resolution
    img = cv2.resize(input_image, (w, h), interpolation=cv2.INTER_LANCZOS4 if upscale > 1 else cv2.INTER_AREA)
    img = img.round().clip(0, 255).astype(np.uint8)
    return img


def fix_resize(input_image, size=512, unit_resolution=64):
    h, w, c = input_image.shape
    h = float(h)
    w = float(w)
    upscale = size / min(h, w)
    h *= upscale
    w *= upscale
    h = int(np.round(h / unit_resolution)) * unit_resolution
    w = int(np.round(w / unit_resolution)) * unit_resolution
    img = cv2.resize(input_image, (w, h), interpolation=cv2.INTER_LANCZOS4 if upscale > 1 else cv2.INTER_AREA)
    img = img.round().clip(0, 255).astype(np.uint8)
    return img


def Numpy2Tensor(img):
    """
    np.array[H, w, C] [0, 255] -> Tensor[C, H, W], RGB, [-1, 1]
    """
    # size
    img = np.array(img) / 255 * 2 - 1
    img = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1)
    return img


def Tensor2Numpy(x, h0=None, w0=None):
    """
    Tensor[C, H, W], RGB, [-1, 1] -> PIL.Image
    """
    if h0 is not None and w0 is not None:
        x = x.unsqueeze(0)
        x = interpolate(x, size=(h0, w0), mode='bicubic')
        x = x.squeeze(0)
    x = (x.permute(1, 2, 0) * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)
    return x


def convert_dtype(dtype_str):
    if dtype_str == 'fp8':
        return torch.float8_e5m2
    if dtype_str == 'fp32':
        return torch.float32
    elif dtype_str == 'fp16':
        return torch.float16
    elif dtype_str == 'bf16':
        return torch.bfloat16
    else:
        raise NotImplementedError
