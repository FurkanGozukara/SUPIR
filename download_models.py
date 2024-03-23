import sys
import os
import requests
from tqdm import tqdm
from huggingface_hub import snapshot_download
from pathlib import PureWindowsPath

MODEL_HOME = os.environ.get('MODEL_HOME', 'models')
CHECKPOINT_DIR= os.path.join(MODEL_HOME, 'checkpoints')

if sys.platform == 'win32':
    MODEL_HOME = '/'.join(PureWindowsPath(MODEL_HOME).parts)
    CHECKPOINT_DIR= '/'.join(PureWindowsPath(CHECKPOINT_DIR).parts)

def create_directory(path):
    """Create directory if it does not exist."""
    if not os.path.exists(path):
        os.makedirs(path)
        print(f'Directory created: {path}')
    else:
        print(f'Directory already exists: {path}')


def download_file(url, folder_path, file_name=None):
    """Download a file from a given URL to a specified folder with an optional file name."""
    local_filename = file_name if file_name else url.split('/')[-1]
    local_filepath = os.path.join(folder_path, local_filename)
    print(f'Downloading {url} to: {local_filepath}')
    if os.path.exists(local_filepath):
        print(f'File already exists: {local_filepath}')
        return
    # Stream download to handle large files
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        total_size_in_bytes = int(r.headers.get('content-length', 0))
        block_size = 1024  # 1 Kibibyte
        progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)
        with open(local_filepath, 'wb') as f:
            for data in r.iter_content(block_size):
                progress_bar.update(len(data))
                f.write(data)
        progress_bar.close()

    if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
        print('ERROR, something went wrong')
    else:
        print(f'Downloaded {local_filename} to {folder_path}')


# Define the folders and their corresponding file URLs with optional file names
checkpoint_files = {
    CHECKPOINT_DIR: [
        ('https://huggingface.co/RunDiffusion/Juggernaut-XL-v9/resolve/main/Juggernaut-XL_v9_RunDiffusionPhoto_v2.safetensors', 'Juggernaut-XL_v9_RunDiffusionPhoto_v2.safetensors'),
        ('https://huggingface.co/ashleykleynhans/SUPIR/resolve/main/SUPIR-v0F.ckpt', 'SUPIR-v0F.ckpt'),
        ('https://huggingface.co/ashleykleynhans/SUPIR/resolve/main/SUPIR-v0Q.ckpt', 'SUPIR-v0Q.ckpt')        
    ]
}

if __name__ == '__main__':
    for folder, files in checkpoint_files.items():
        create_directory(folder)
        for file_url, file_name in files:
            download_file(file_url, folder, file_name)

    llava_model = os.getenv('LLAVA_MODEL', 'liuhaotian/llava-v1.5-7b')
    llava_clip_model = 'openai/clip-vit-large-patch14-336'
    sdxl_clip_model = 'openai/clip-vit-large-patch14'
    sdxl_clip2_model ='laion/CLIP-ViT-bigG-14-laion2B-39B-b160k/open_clip_pytorch_model.bin' 

    print(f'Downloading LLaVA model: {llava_model}')
    model_folder = fr"{MODEL_HOME}/{llava_model}"
    if not os.path.exists(model_folder):
        snapshot_download(llava_model, local_dir=model_folder, local_dir_use_symlinks=False)
    else:        
        print(f'Model already exists: {llava_model}')

    print(f'Downloading LLaVA CLIP model: {llava_clip_model}')
    model_folder = fr"{MODEL_HOME}/{llava_clip_model}"
    if not os.path.exists(model_folder):
        snapshot_download(llava_clip_model, local_dir=model_folder, local_dir_use_symlinks=False)
    else:
        print(f'Model already exists: {llava_clip_model}')

    print(f'Downloading SDXL CLIP model: {sdxl_clip_model}')
    model_folder = fr"{MODEL_HOME}/{sdxl_clip_model}"
    if not os.path.exists(model_folder):
        snapshot_download(sdxl_clip_model, local_dir=model_folder, local_dir_use_symlinks=False)
    else:
        print(f'Model already exists: {sdxl_clip_model}')

    print(f'Downloading SDXL CLIP 2 model: {sdxl_clip2_model}')    
    model_id = fr"{sdxl_clip2_model.split('/')[-3]}/{sdxl_clip2_model.split('/')[-2]}"
    model_folder = fr"{MODEL_HOME}/{model_id}"
    if not os.path.exists(fr"{MODEL_HOME}/{sdxl_clip2_model}"):
        snapshot_download(model_id, allow_patterns=sdxl_clip2_model.split('/')[-1], local_dir=model_folder, local_dir_use_symlinks=False)
    else:
        print(f'Model already exists: {sdxl_clip2_model}')
