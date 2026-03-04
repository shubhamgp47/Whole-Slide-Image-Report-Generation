# UNI model

import torch
import timm
from timm.data.transforms_factory import create_transform
from timm.data import resolve_data_config
from torchvision import transforms
from huggingface_hub import login

def build_model(device, gpu_num, model_name, model_path):
    if model_name == 'uni1':
        # Login to HuggingFace (you may need to handle this differently)
        HF_TOKEN = "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"
        try:
            login(token=HF_TOKEN)  # Use your HF token
        except Exception as e:
            print(f"HuggingFace login failed: {e}")
            raise
            
        # Load UNI model
        model = timm.create_model(
            "hf-hub:MahmoodLab/UNI", 
            pretrained=True, 
            init_values=1e-5, 
            dynamic_img_size=True
        )
        
        # Move to device
        model.to(device)
        
        # Multi-GPU support
        if gpu_num > 1:
            model = torch.nn.parallel.DataParallel(model)
            
        model.eval()
        
        return model
    else:
        raise NotImplementedError(f'{model_name} is not implemented...')

def build_transform():
    """
    Use UNI's official transform creation method
    """
    # Create a temporary model to get the transform config
    temp_model = timm.create_model(
        "hf-hub:MahmoodLab/UNI", 
        pretrained=True, 
        init_values=1e-5, 
        dynamic_img_size=True
    )
    
    # Use official UNI transform creation
    transform = create_transform(**resolve_data_config(temp_model.pretrained_cfg, model=temp_model))
    return transform
