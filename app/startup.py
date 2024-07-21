import torch
import sys
import os
sys.path.append(os.path.abspath('../src/'))
from src.models import GOPT

def load_model():
    model = GOPT(embed_dim=24, num_heads=1, depth=3, input_dim=84)
    model = torch.nn.DataParallel(model)
    sd = torch.load('C:/Users/Admin/Desktop/Code/selfomy/api/pretrained_models/gopt_librispeech/best_audio_model.pth', map_location='cpu')
    model.load_state_dict(sd, strict=True)
    return model
