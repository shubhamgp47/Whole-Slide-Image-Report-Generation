#UNI2
import os
import torch
import timm
from timm.data.transforms_factory import create_transform
from timm.data import resolve_data_config
from torchvision import transforms
from huggingface_hub import login

def build_model(device, gpu_num, model_name, model_path):
    if model_name == 'uni2':
        # Set up local paths
        HF_TOKEN = "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"
        local_dir = "/home/woody/iwi5/iwi5204h/UNI2_model"
        
        # Login (can be skipped if using local files)
        try:
            login(token=HF_TOKEN)
        except:
            print("Login failed, using local files only")
        
        # Create model architecture
        timm_kwargs = {
            'model_name': 'vit_giant_patch14_224',
            'img_size': 224, 
            'patch_size': 14, 
            'depth': 24,
            'num_heads': 24,
            'init_values': 1e-5, 
            'embed_dim': 1536,
            'mlp_ratio': 2.66667*2,
            'num_classes': 0, 
            'no_embed_class': True,
            'mlp_layer': timm.layers.SwiGLUPacked, 
            'act_layer': torch.nn.SiLU, 
            'reg_tokens': 8, 
            'dynamic_img_size': True
        }
        
        # Create model
        model = timm.create_model(pretrained=False, **timm_kwargs)
        
        # Load weights from local file
        weights_path = os.path.join(local_dir, "pytorch_model.bin")
        if os.path.exists(weights_path):
            model.load_state_dict(torch.load(weights_path, map_location="cpu"), strict=True)
        else:
            raise FileNotFoundError(f"Model weights not found at {weights_path}")
        
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
    Use UNI2's transform - same as UNI (standard ImageNet preprocessing)
    """
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    
    return transform
