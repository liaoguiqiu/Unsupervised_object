from dataset import *
from model_dino_2 import *
import torch
import numpy as np
import cv2


# Hyperparameters.
seed = 0
batch_size = 1
num_slots = 3
num_iterations = 3
resolution = (128, 128)

# Load model.
resolution = (128, 128)
model = SlotAttentionAutoEncoder(resolution, num_slots, num_iterations, 64)
model.load_state_dict(torch.load('./tmp/model12.pth')['model_state_dict'])

test_set = PARTNET('train')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

image = test_set[106]['image']
image = image.unsqueeze(0).to(device)
recon_combined, recons, masks, slots,x_dino_OG = model(image)

# Convert tensors to numpy arrays.
image = image.squeeze(0).permute(1, 2, 0).cpu().numpy()
recon_combined = recon_combined.squeeze(0).permute(1, 2, 0).cpu().detach().numpy()
recons = recons.squeeze(0).cpu().detach().numpy()
masks = masks.squeeze(0).cpu().detach().numpy()

# Display images using cv2.
cv2.imshow('Image', image)
# cv2.imshow('Reconstructed', recon_combined)
for i in range(3):
    mask_resized = cv2.resize(masks[i], (128, 128), interpolation=cv2.INTER_NEAREST)
    mask_resized = mask_resized>0.5
    picture = image * mask_resized[:, :, np.newaxis]*254
    # picture = (recons[i] * masks[i] + (1 - masks[i])) * 255  # Convert to 0-255 range for display
    picture = picture.astype(np.uint8)
    cv2.imshow('Slot %s' % str(i + 1), picture)

cv2.waitKey(0)
cv2.destroyAllWindows()
