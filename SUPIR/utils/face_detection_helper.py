import os
import torch
from copy import deepcopy
from facexlib.detection import RetinaFace
from facexlib.utils import load_file_from_url as facexlib_load_file


def init_detection_model_with_local_check(model_name, half=False, device='cuda', local_model_dir=None):
    """
    Initialize face detection model with local folder check first.

    Args:
        model_name: Name of the detection model ('retinaface_resnet50' or 'retinaface_mobile0.25')
        half: Whether to use half precision
        device: Device to load model on
        local_model_dir: Local directory to check for models first (e.g., 'SUPIR/models')

    Returns:
        Initialized detection model
    """
    # Define model configurations
    model_configs = {
        'retinaface_resnet50': {
            'network_name': 'resnet50',
            'filename': 'detection_Resnet50_Final.pth',
            'url': 'https://github.com/xinntao/facexlib/releases/download/v0.1.0/detection_Resnet50_Final.pth'
        },
        'retinaface_mobile0.25': {
            'network_name': 'mobile0.25',
            'filename': 'detection_mobilenet0.25_Final.pth',
            'url': 'https://github.com/xinntao/facexlib/releases/download/v0.1.0/detection_mobilenet0.25_Final.pth'
        }
    }

    if model_name not in model_configs:
        raise NotImplementedError(f'{model_name} is not implemented.')

    config = model_configs[model_name]

    # Initialize the model
    model = RetinaFace(network_name=config['network_name'], half=half, device=device)

    # Try to load from local directory first
    model_path = None
    if local_model_dir and os.path.exists(local_model_dir):
        local_path = os.path.join(local_model_dir, config['filename'])
        if os.path.exists(local_path):
            print(f"Loading face detection model from local path: {local_path}")
            model_path = local_path

    # If not found locally, download from URL
    if model_path is None:
        print(f"Local model not found. Downloading face detection model from: {config['url']}")
        model_path = facexlib_load_file(
            url=config['url'],
            model_dir='facexlib/weights',
            progress=True,
            file_name=None
        )

    # Load the model weights
    load_net = torch.load(model_path, map_location=lambda storage, loc: storage)

    # Remove unnecessary 'module.' prefix from keys
    for k, v in deepcopy(load_net).items():
        if k.startswith('module.'):
            load_net[k[7:]] = v
            load_net.pop(k)

    model.load_state_dict(load_net, strict=True)
    model.eval()
    model = model.to(device)

    return model