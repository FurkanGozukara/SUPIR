import os
import torch
import numpy as np
import cv2

from PIL import Image
from torch.nn.functional import interpolate
from omegaconf import OmegaConf
from .utils import sd_disable_initialization, models_utils
from sgm.util import instantiate_from_config
from SUPIR.utils.model_fetch import get_model
import CKPT_PTH 


config = None


def create_model(config_path, device='cpu'):
    global config
    config = OmegaConf.load(config_path)
    with sd_disable_initialization.DisableInitialization(disable_clip=False):
        with sd_disable_initialization.InitializeOnMeta():
            model = instantiate_from_config(config.model)
    #model = model.to(device)
    print(f'Loaded model config from [{config_path}] and moved to {device}')
    return model


def load_supir_weights(model, supir_sign=None, reload_supir=False, ckpt_dir=None, ckpt=None, vae_file=None):
    global config
    
    weight_dtype_conversion = {
        'first_stage_model': None,
        'alphas_cumprod': None,
        '': torch.float16,
    }   

    if reload_supir == False:      
        if ckpt:
            config.SDXL_CKPT = ckpt

        if config.SDXL_CKPT is not None:       
            state_dict = models_utils.load_state_dict(config.SDXL_CKPT)    
            with sd_disable_initialization.LoadStateDictOnMeta(state_dict, device=model.device, weight_dtype_conversion=weight_dtype_conversion):
                models_utils.load_model_weights(model, state_dict, vae_file)        

        if config.SUPIR_CKPT is not None:
            model_file = get_model(os.path.join(ckpt_dir, config.SUPIR_CKPT))
            state_dict = models_utils.load_state_dict(model_file)
            with sd_disable_initialization.LoadStateDictOnMeta(state_dict, device=model.device, weight_dtype_conversion=weight_dtype_conversion):
                models_utils.load_model_weights(model, state_dict, vae_file) 

    if supir_sign is not None:
        assert supir_sign in ['F', 'Q']
        if supir_sign == 'F':            
            model_file = get_model(CKPT_PTH.SUPIR_CKPT_F_PTH)
            state_dict = models_utils.load_state_dict(model_file)
            with sd_disable_initialization.LoadStateDictOnMeta(state_dict, device=model.device, weight_dtype_conversion=weight_dtype_conversion):
                models_utils.load_model_weights(model, state_dict,vae_file) 
        elif supir_sign == 'Q':
            model_file = get_model(CKPT_PTH.SUPIR_CKPT_Q_PTH)            
            state_dict = models_utils.load_state_dict(model_file)
            with sd_disable_initialization.LoadStateDictOnMeta(state_dict, device=model.device, weight_dtype_conversion=weight_dtype_conversion):
                models_utils.load_model_weights(model, state_dict, vae_file)             
    
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

