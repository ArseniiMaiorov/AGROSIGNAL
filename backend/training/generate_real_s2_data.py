#!/usr/bin/env python3
import numpy as np
from pathlib import Path
from skimage.feature import canny
from scipy.nd

cp backend/models/sam_vit_h.pth backend/models/sam1_vit_h.pth

python3 -c "from segment_anything import sam_model_registry; sam = sam_model_registry['vit_h']('backend/models/sam1_vit_h.pth'); print('✅ SAM1 OK')"

python3 -c "from segment_anything import sam_model_registry; sam = sam_model_registry['vit_h']('backend/models/sam1_vit_h.pth'); print('✅ SAM1 OK')"

mkdir -p backend/training/unet_dataset/{train,val} backend/debug/runs/real_data/

python3 -c "
import numpy as np
from pathlib

