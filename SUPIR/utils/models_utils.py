import platform
import os
import torch
from  SUPIR.utils import shared, devices

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

def load_state_dict(ckpt_path, map_location='cpu'):
    _, extension = os.path.splitext(ckpt_path)
    platform_name = platform.uname()
    isWSL2 = 'WSL2' in platform_name.release

    if extension.lower() == ".safetensors":
        import safetensors.torch        
        if not isWSL2:
            state_dict = safetensors.torch.load_file(ckpt_path, device=map_location)
        else:
            state_dict = safetensors.torch.load(open(ckpt_path, 'rb').read())
            state_dict = {k: v.to(map_location) for k, v in state_dict.items()}
    else:
        state_dict = get_state_dict(torch.load(ckpt_path, map_location=torch.device(map_location)))
    state_dict = get_state_dict(state_dict)
    print(f'Loaded state_dict from [{ckpt_path}]')
    return state_dict

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

def load_model_weights(model, state_dict, vae_file=None):

    if devices.fp8:
        model.half()
   
    model.load_state_dict(state_dict, strict=False)    

    del state_dict

    if shared.opts.opt_channelslast:
        model.to(memory_format=torch.channels_last)        
        #print('apply channels_last')

    if shared.opts.half_mode == False:
        model.float()        
        devices.dtype_unet = torch.float32        
        #print('apply float')
    else:
        vae = model.first_stage_model
        depth_model = getattr(model, 'depth_model', None)

        if shared.opts.half_mode:
            model.half()
        
        model.first_stage_model = vae
        if depth_model:
            model.depth_model = depth_model
        #print('apply half')

    for module in model.modules():
        if hasattr(module, 'fp16_weight'):
            del module.fp16_weight
        if hasattr(module, 'fp16_bias'):
            del module.fp16_bias

    if check_fp8(model):
        devices.fp8 = True
        #first_stage = model.first_stage_model
        #model.first_stage_model = None
        for module in model.modules():
            if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):               
                module.to(torch.float8_e4m3fn)
        #model.first_stage_model = first_stage
        #print("apply fp8")
    else:
        devices.fp8 = False

    devices.unet_needs_upcast = shared.opts.upcast_sampling  and devices.dtype == torch.float16 and devices.dtype_unet == torch.float16

    model.first_stage_model.to(devices.dtype_vae)

    if hasattr(model, 'logvar'):
        model.logvar = model.logvar.to(devices.device) 

   