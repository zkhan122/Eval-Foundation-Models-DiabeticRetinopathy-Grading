import matplotlib.pyplot as plt
import torch

def show_images(dataset, train_labels, num_images):
    
    cols = 4
    rows = (num_images + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(20, 10))
    axes = axes.flatten()
    
    for idx in range(num_images):
        image, image_name, *_ = dataset[idx]
        
        # Denormalize
        img = image.clone()
        img[0] = img[0] * 0.229 + 0.485
        img[1] = img[1] * 0.224 + 0.456
        img[2] = img[2] * 0.225 + 0.406
        img = torch.clamp(img, 0, 1)
        
        axes[idx].imshow(img.permute(1, 2, 0))
        axes[idx].set_title(f'Label: {image_name}', fontsize=12)
        axes[idx].axis('off')
    
    for idx in range(num_images, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.show()

