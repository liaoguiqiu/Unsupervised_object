from dataset import *
from model_dino_3 import *
import torch
import numpy as np
import cv2


# Hyperparameters.
seed = 0
batch_size = 1
num_slots = 7
num_iterations = 3
# resolution = (128, 128)
output_dir =  "C:/1projects/codes/Object_centric/output"
# Load model.
resolution = (128, 128)
model = SlotAttentionAutoEncoder(resolution, num_slots, num_iterations, 384)
model.load_state_dict(torch.load('./tmp/model13.pth')['model_state_dict'])

test_set = PARTNET('train',resolution=resolution)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

Img_num = len(test_set)
colors = [  # Define a color for each mask
    (255, 0, 0),   # Red
    (0, 255, 0),   # Green
    (0, 0, 255),   # Blue
    (255, 255, 0), # Yellow
    (0, 255, 255), # Cyan
    (255, 0, 255), # Magenta
    (255, 165, 0)  # Orange
]
for id in range(Img_num):
    image = test_set[id]['image']
    image = image.unsqueeze(0).to(device)
    recon_combined, recons, masks, slots,x_dino_OG = model(image)

    # Convert tensors to numpy arrays.
    image = image.squeeze(0).permute(1, 2, 0).cpu().numpy()
    recon_combined = recon_combined.squeeze(0).permute(1, 2, 0).cpu().detach().numpy()
    recons = recons.squeeze(0).cpu().detach().numpy()
    masks = masks.squeeze(0).cpu().detach().numpy()

    # Display images using cv2.
    cv2.imshow('Image', image)
    image = image *254
    composite_mask = np.zeros_like(image)
    # cv2.imshow('Reconstructed', recon_combined)
    for i in range(7):  # Assuming there are exactly 7 masks
        mask_resized = cv2.resize(masks[i], resolution, interpolation=cv2.INTER_NEAREST)
        mask_resized = mask_resized > 0.5  # Threshold the mask

        color_mask = np.zeros_like(image)  # Create a color mask
        for c in range(3):  # Apply the color to the mask
            color_mask[:, :, c] = mask_resized * colors[i][c]

        # Blend the color mask with the current state of the final image
        composite_mask = cv2.add(composite_mask, color_mask)
    final_image = cv2.addWeighted(image, 0.5, composite_mask, 0.5, 0)  # Adjust these weights to taste
    
    cv2.imwrite(output_dir+"/"+str(id)+".jpg", final_image.astype(np.uint8))
    cv2.imshow('Colored Masks Overlay', final_image.astype(np.uint8))
    cv2.waitKey(1)
    # cv2.destroyAllWindows()
