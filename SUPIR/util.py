import os
import torch
import numpy as np
import cv2
import platform
from PIL import Image
from torch.nn.functional import interpolate
from omegaconf import OmegaConf
from .utils import sd_disable_initialization, devices, shared, optimization, sd_models_xl
from sgm.util import instantiate_from_config
from SUPIR.utils.model_fetch import get_model
import CKPT_PTH 
checkpoint_dict_replacements_sd1 = {
    'cond_stage_model.transformer.embeddings.': 'cond_stage_model.transformer.text_model.embeddings.',
    'cond_stage_model.transformer.encoder.': 'cond_stage_model.transformer.text_model.encoder.',
    'cond_stage_model.transformer.final_layer_norm.': 'cond_stage_model.transformer.text_model.final_layer_norm.',
}

checkpoint_dict_replacements_sd2_turbo = { # Converts SD 2.1 Turbo from SGM to LDM format.
    'conditioner.embedders.0.': 'cond_stage_model.',
}
def transform_checkpoint_dict_key(k, replacements):
    for text, replacement in replacements.items():
        if k.startswith(text):
            k = replacement + k[len(text):]

    return k

def get_state_dict(d):

    pl_sd = d.pop("state_dict", d)
    pl_sd.pop("state_dict", None)

    is_sd2_turbo = 'conditioner.embedders.0.model.ln_final.weight' in pl_sd and pl_sd['conditioner.embedders.0.model.ln_final.weight'].size()[0] == 1024

    sd = {}
    for k, v in pl_sd.items():
        if is_sd2_turbo:
            new_key = transform_checkpoint_dict_key(k, checkpoint_dict_replacements_sd2_turbo)
        else:
            new_key = transform_checkpoint_dict_key(k, checkpoint_dict_replacements_sd1)

        if new_key is not None:
            sd[new_key] = v

    pl_sd.clear()
    pl_sd.update(sd)

    return pl_sd

    #return d.get('state_dict', d)

def load_state_dict(ckpt_path, location='cpu'):
    _, extension = os.path.splitext(ckpt_path)
    platform_name = platform.uname()
    isWSL2 = 'WSL2' in platform_name.release

    if extension.lower() == ".safetensors":
        import safetensors.torch        
        if not isWSL2:
            state_dict = safetensors.torch.load_file(ckpt_path, device=location)
        else:
            state_dict = safetensors.torch.load(open(ckpt_path, 'rb').read())
            state_dict = {k: v.to(location) for k, v in state_dict.items()}
    else:
        state_dict = get_state_dict(torch.load(ckpt_path, map_location=torch.device(location)))
    state_dict = get_state_dict(state_dict)
    print(f'Loaded state_dict from [{ckpt_path}]')
    return state_dict

def create_model(config_path):
    config = OmegaConf.load(config_path)
    model = instantiate_from_config(config.model).cpu()
    print(f'Loaded model config from [{config_path}]')
    return model

def create_SUPIR_model(config_path, supir_sign=None, device='cpu', ckpt_dir=None, ckpt=None):
    config = OmegaConf.load(config_path)
    if ckpt:
        config.SDXL_CKPT = ckpt
    
    weight_dtype_conversion = {
        'first_stage_model': None,
        'alphas_cumprod': None,
        '': torch.float16,
    }
    
    if config.SDXL_CKPT is not None:       
        state_dict = load_state_dict(config.SDXL_CKPT)    
        with sd_disable_initialization.DisableInitialization(disable_clip=False):
            with sd_disable_initialization.InitializeOnMeta():
                model = instantiate_from_config(config.model)
       
        print(f'Loaded model config from [{config_path}] and moved to {device}')

        with sd_disable_initialization.LoadStateDictOnMeta(state_dict, device=device, weight_dtype_conversion=weight_dtype_conversion):
            load_model_weights(model, state_dict)        

    if config.SUPIR_CKPT is not None:
        model_file = get_model(os.path.join(ckpt_dir, config.SUPIR_CKPT))
        state_dict = load_state_dict(model_file)
        with sd_disable_initialization.LoadStateDictOnMeta(state_dict, device=device, weight_dtype_conversion=weight_dtype_conversion):
            load_model_weights(model, state_dict) 
    if supir_sign is not None:
        assert supir_sign in ['F', 'Q']
        if supir_sign == 'F':            
            model_file = get_model(CKPT_PTH.SUPIR_CKPT_F_PTH)
            state_dict = load_state_dict(model_file)
            with sd_disable_initialization.LoadStateDictOnMeta(state_dict, device=device, weight_dtype_conversion=weight_dtype_conversion):
                load_model_weights(model, state_dict) 
        elif supir_sign == 'Q':
            model_file = get_model(CKPT_PTH.SUPIR_CKPT_Q_PTH)            
            state_dict = load_state_dict(model_file)
            with sd_disable_initialization.LoadStateDictOnMeta(state_dict, device=device, weight_dtype_conversion=weight_dtype_conversion):
                load_model_weights(model, state_dict) 
            #model.load_state_dict(load_state_dict(model_file), strict=False)
    
    #optimization.model_hijack.hijack(model)
    return model


def load_QF_ckpt(config_path, device='cpu'):
    config = OmegaConf.load(config_path)
    ckpt_F = torch.load(CKPT_PTH.SUPIR_CKPT_F_PTH, map_location=device)
    ckpt_Q = torch.load(CKPT_PTH.SUPIR_CKPT_Q_PTH, map_location=device)
    return ckpt_Q, ckpt_F


def PIL2Tensor(img, upsacle=1, min_size=1024):
    '''
    PIL.Image -> Tensor[C, H, W], RGB, [-1, 1]
    '''
    # size
    w, h = img.size
    w *= upsacle
    h *= upsacle
    w0, h0 = round(w), round(h)
    if min(w, h) < min_size:
        _upsacle = min_size / min(w, h)
        w *= _upsacle
        h *= _upsacle
    else:
        _upsacle = 1
    w = int(np.round(w / 64.0)) * 64
    h = int(np.round(h / 64.0)) * 64
    x = img.resize((w, h), Image.BICUBIC)
    x = np.array(x).round().clip(0, 255).astype(np.uint8)
    x = x / 255 * 2 - 1
    x = torch.tensor(x, dtype=torch.float32).permute(2, 0, 1)
    return x, h0, w0


def Tensor2PIL(x, h0, w0):
    '''
    Tensor[C, H, W], RGB, [-1, 1] -> PIL.Image
    '''
    x = x.unsqueeze(0)
    x = interpolate(x, size=(h0, w0), mode='bicubic')
    x = (x.squeeze(0).permute(1, 2, 0) * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)
    return Image.fromarray(x)


def HWC3(x):
    assert x.dtype == np.uint8
    if x.ndim == 2:
        x = x[:, :, None]
    assert x.ndim == 3
    H, W, C = x.shape
    assert C == 1 or C == 3 or C == 4
    if C == 3:
        return x
    if C == 1:
        return np.concatenate([x, x, x], axis=2)
    if C == 4:
        color = x[:, :, 0:3].astype(np.float32)
        alpha = x[:, :, 3:4].astype(np.float32) / 255.0
        y = color * alpha + 255.0 * (1.0 - alpha)
        y = y.clip(0, 255).astype(np.uint8)
        return y


def upscale_image(input_image, upscale, min_size=None, unit_resolution=64):
    H, W, C = input_image.shape
    H = float(H)
    W = float(W)
    H *= upscale
    W *= upscale
    if min_size is not None:
        if min(H, W) < min_size:
            _upsacle = min_size / min(W, H)
            W *= _upsacle
            H *= _upsacle
    H = int(np.round(H / unit_resolution)) * unit_resolution
    W = int(np.round(W / unit_resolution)) * unit_resolution
    img = cv2.resize(input_image, (W, H), interpolation=cv2.INTER_LANCZOS4 if upscale > 1 else cv2.INTER_AREA)
    img = img.round().clip(0, 255).astype(np.uint8)
    return img


def fix_resize(input_image, size=512, unit_resolution=64):
    H, W, C = input_image.shape
    H = float(H)
    W = float(W)
    upscale = size / min(H, W)
    H *= upscale
    W *= upscale
    H = int(np.round(H / unit_resolution)) * unit_resolution
    W = int(np.round(W / unit_resolution)) * unit_resolution
    img = cv2.resize(input_image, (W, H), interpolation=cv2.INTER_LANCZOS4 if upscale > 1 else cv2.INTER_AREA)
    img = img.round().clip(0, 255).astype(np.uint8)
    return img


def Numpy2Tensor(img):
    '''
    np.array[H, w, C] [0, 255] -> Tensor[C, H, W], RGB, [-1, 1]
    '''
    # size
    img = np.array(img) / 255 * 2 - 1
    img = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1)
    return img


def Tensor2Numpy(x, h0=None, w0=None):
    '''
    Tensor[C, H, W], RGB, [-1, 1] -> PIL.Image
    '''
    if h0 is not None and w0 is not None:
        x = x.unsqueeze(0)
        x = interpolate(x, size=(h0, w0), mode='bicubic')
        x = x.squeeze(0)
    x = (x.permute(1, 2, 0) * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)
    return x


def convert_dtype(dtype_str):
    if dtype_str == 'fp32':
        return torch.float32
    elif dtype_str == 'fp16':
        return torch.float16
    elif dtype_str == 'bf16':
        return torch.bfloat16
    else:
        raise NotImplementedError

def check_fp8(model):
    if model is None:
        return None
    if devices.get_optimal_device_name() == "mps":
        enable_fp8 = False
    elif shared.opts.fp8_storage == True:
        enable_fp8 = True    
    else:
        enable_fp8 = False
    return enable_fp8


def load_model_weights(model, state_dict):

    if devices.fp8:
        # prevent model to load state dict in fp8
        model.half()

    model.is_sdxl = hasattr(model, 'conditioner')
    model.is_sd2 = not model.is_sdxl and hasattr(model.cond_stage_model, 'model')
    model.is_sd1 = not model.is_sdxl and not model.is_sd2
    model.is_ssd = model.is_sdxl and 'model.diffusion_model.middle_block.1.transformer_blocks.0.attn1.to_q.weight' not in state_dict.keys()
    if model.is_sdxl:
        sd_models_xl.extend_sdxl(model)

    model.load_state_dict(state_dict, strict=False)    

    del state_dict

    if shared.opts.opt_channelslast:
        model.to(memory_format=torch.channels_last)        
        print('apply channels_last')

    if shared.opts.half_mode == False:
        model.float()        
        devices.dtype_unet = torch.float32        
        print('apply float')
    else:
        vae = model.first_stage_model
        depth_model = getattr(model, 'depth_model', None)

        if shared.opts.half_mode:
            model.half()
        
        model.first_stage_model = vae
        if depth_model:
            model.depth_model = depth_model
        print('apply half')

    for module in model.modules():
        if hasattr(module, 'fp16_weight'):
            del module.fp16_weight
        if hasattr(module, 'fp16_bias'):
            del module.fp16_bias

    if check_fp8(model):
        devices.fp8 = True
        first_stage = model.first_stage_model
        model.first_stage_model = None
        for module in model.modules():
            if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):               
                module.to(torch.float8_e4m3fn)
        model.first_stage_model = first_stage
        print("apply fp8")
    else:
        devices.fp8 = False

    devices.unet_needs_upcast = shared.opts.upcast_sampling  and devices.dtype == torch.float16 and devices.dtype_unet == torch.float16

    model.first_stage_model.to(devices.dtype_vae)
    

    # # clean up cache if limit is reached
    # while len(checkpoints_loaded) > shared.opts.sd_checkpoint_cache:
    #     checkpoints_loaded.popitem(last=False)

    # model.sd_model_hash = sd_model_hash
    # model.sd_model_checkpoint = checkpoint_info.filename
    # model.sd_checkpoint_info = checkpoint_info
    # shared.opts.data["sd_checkpoint_hash"] = checkpoint_info.sha256

    if hasattr(model, 'logvar'):
        model.logvar = model.logvar.to(devices.device)  # fix for training

    # sd_vae.delete_base_vae()
    # sd_vae.clear_loaded_vae()
    # vae_file, vae_source = sd_vae.resolve_vae(checkpoint_info.filename).tuple()
    # sd_vae.load_vae(model, vae_file, vae_source)
    # timer.record("load VAE")