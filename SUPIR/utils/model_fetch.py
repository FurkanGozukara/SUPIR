from huggingface_hub import snapshot_download, hf_hub_download
import os

models_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", 'models'))


def get_model(model_repo: str):    
    extensions = ['ckpt', 'safetensors', 'bin', 'json']
    model_repo_splitted = model_repo.split('/')
    model_file = None
    if len(model_repo_splitted) > 2: #model folder was provided
        if any(x in model_repo for x in extensions):    #specific file was provided        
            model_path = '/'.join(model_repo_splitted[0:-1])
            model_file = model_repo_splitted[-1]
            model_id =  os.path.join(model_repo_splitted[-3], model_repo_splitted[-2])  
        else: # entire repo
            model_path = model_repo
            model_id =  os.path.join(model_repo_splitted[-2], model_repo_splitted[-1])  
    else: #use default model folder
        model_id = model_repo
        model_path = os.path.join(models_folder, model_id)

    if not os.path.exists(model_path):
        if model_file:
            hf_hub_download(repo_id=model_id, filename=model_file, local_dir=model_path)            
        else:
            snapshot_download(model_id, local_dir=model_path, local_dir_use_symlinks=False)            
    
    if model_file:
        return os.path.join(model_path, model_file)
    else:
        return model_path
