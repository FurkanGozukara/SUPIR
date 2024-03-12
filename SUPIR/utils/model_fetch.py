from huggingface_hub import snapshot_download, hf_hub_download
from pathlib import PureWindowsPath
import os
import sys

models_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", 'models'))


def get_model(model_repo: str):    
    if sys.platform == 'win32':
        model_repo = '/'.join(PureWindowsPath(model_repo).parts)

    extensions = ['ckpt', 'safetensors', 'bin']
    model_repo_splitted = model_repo.split('/')
    model_file = None
    if len(model_repo_splitted) > 2: #model folder was provided
        if any(x in model_repo for x in extensions):    #specific file was provided        
            if '.bin' in model_repo:
                model_path = '/'.join(model_repo_splitted[0:-1])
            else:
                model_path = '/'.join(model_repo_splitted[0:-3])

            model_file = model_repo_splitted[-1]
            model_id =  fr"{model_repo_splitted[-3]}/{model_repo_splitted[-2]}"  
            model = fr"{model_path}/{model_file}"

            if not os.path.exists(model):        
                snapshot_download(model_id, allow_patterns=model_file, local_dir=model_path, local_dir_use_symlinks=False) 
                #hf_hub_download(repo_id=model_id, filename=model_file, local_dir=model_path)                            
            return model        
        else: # entire repo
            model_path = model_repo
            model_id =  fr"{model_repo_splitted[-2]}/{model_repo_splitted[-1]}"
            
            if not os.path.exists(model_path):        
                snapshot_download(model_id, local_dir=model_path, local_dir_use_symlinks=False)            
            return model_path
    else: #use default model folder
        model_id = model_repo
        model_path = os.path.join(models_folder, model_id)
        
        if not os.path.exists(model_path):        
            snapshot_download(model_id, local_dir=model_path, local_dir_use_symlinks=False)            
        return model_path    

