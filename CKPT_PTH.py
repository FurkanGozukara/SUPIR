LLAVA_CLIP_PATH = None
LLAVA_MODEL_PATH = None
SDXL_CLIP1_PATH = None
SDXL_CLIP2_CKPT_PTH = None

def setModelPath(model_path):    
    global LLAVA_CLIP_PATH, LLAVA_MODEL_PATH, SDXL_CLIP1_PATH, SDXL_CLIP2_CKPT_PTH

    LLAVA_CLIP_PATH = fr'{model_path}/openai/clip-vit-large-patch14-336' if model_path is not None else 'openai/clip-vit-large-patch14-336'
    LLAVA_MODEL_PATH = fr'{model_path}/liuhaotian/llava-v1.5-7b' if model_path is not None else 'liuhaotian/llava-v1.5-7b'
    SDXL_CLIP1_PATH = fr'{model_path}/openai/clip-vit-large-patch14' if model_path is not None else 'openai/clip-vit-large-patch14'
    SDXL_CLIP2_CKPT_PTH = fr'{model_path}/laion/CLIP-ViT-bigG-14-laion2B-39B-b160k/open_clip_pytorch_model.bin' if model_path is not None else 'laion/CLIP-ViT-bigG-14-laion2B-39B-b160k/open_clip_pytorch_model.bin'
    

