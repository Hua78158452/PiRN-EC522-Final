import os

import torch

# Check out the system assigned GPU id
count = torch.cuda.device_count()
print('There are', count, 'GPU/GPUs available!, The devices are:', os.getenv("CUDA_VISIBLE_DEVICES"), '\n')