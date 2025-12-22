

import matplotlib.pyplot as plt
import torch
from transformers import Conv1D
from tqdm import tqdm
import numpy as np
from torch.amp import autocast


def identity_transform(x):
    return x


# for data imbalance
def weighted_class_imbalance(dataset):
    labels = [label for _, label, _ in dataset]
    class_counts = np.bincount(labels)
    class_weights = 1.0 / class_counts
    class_weights = class_weights / class_weights.sum() * len(class_weights)
    return torch.FloatTensor(class_weights)



def train_one_epoch_retfound(model, dataloader, criterion, optimizer, device, epoch, scaler):
    model.train()
    running_loss = 0
    correct = 0
    total = 0

    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1} - Training")

    for batch_idx, (images, labels, sources) in enumerate(progress_bar):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True).long()

        # feed forward pass
        optimizer.zero_grad()


        if scaler is not None:
            with autocast(device_type="cuda"):
                outputs = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()



        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        progress_bar.set_postfix({
            "loss": running_loss / (batch_idx + 1),
            "acc": 100 * correct / total
        })


        epoch_loss = running_loss / len(dataloader)
        epoch_acc = 100 * correct / total

    return epoch_loss, epoch_acc



def train_one_epoch_urfound(model, dataloader, criterion, optimizer, device, epoch, scaler):
    model.train()
    running_loss = 0
    correct = 0
    total = 0

    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1} - Training")

    for batch_idx, (images, labels, sources) in enumerate(progress_bar):
        images = images.to(device)
        labels = labels.to(device).long()

        # feed forward pass
        optimizer.zero_grad()

        if scaler is not None:
            with autocast(device_type="cuda"):
                outputs = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        progress_bar.set_postfix({
            "loss": running_loss / (batch_idx + 1),
            "acc": 100 * correct / total
        })


        epoch_loss = running_loss / len(dataloader)
        epoch_acc = 100 * correct / total

    return epoch_loss, epoch_acc




def train_one_epoch_clip(model, dataloader, criterion, optimizer, device, epoch, scaler):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1} - Training")
    
    for batch_idx, (images, labels, sources) in enumerate(progress_bar):
        images = images.to(device)
        labels = labels.to(device).long()
        
        optimizer.zero_grad()
        
        if scaler is not None:
            # Mixed precision training
            with autocast(device_type="cuda"): 
                outputs_obj = model(images)
                image_features = outputs_obj.image_embeds
                outputs = model.classifier(image_features)
                loss = criterion(outputs, labels)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            # Regular training
            outputs_obj = model(images)
            image_features = outputs_obj.image_embeds
            outputs = model.classifier(image_features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
        # Update metrics
        running_loss += loss.item()
        progress_bar.set_postfix({'loss': running_loss / (batch_idx + 1)})


def validate_clip(model, dataloader, criterion, device):
    """Validate the model"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        pbar = tqdm(dataloader, desc='Validation')
        for batch_idx, (images, labels, sources) in enumerate(pbar):
            images = images.to(device)
            labels = labels.to(device)

            outputs_obj = model(images)
            image_features = outputs_obj.image_embeds
            outputs = model.classifier(image_features)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            pbar.set_postfix({
                'loss': running_loss / (batch_idx + 1),
                'acc': 100. * correct / total
            })

    val_loss = running_loss / len(dataloader)
    val_acc = 100. * correct / total

    return val_loss, val_acc


def validate_retfound(model, dataloader, criterion, device):
    """Validate the model"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        pbar = tqdm(dataloader, desc='Validation')
        for batch_idx, (images, labels, sources) in enumerate(pbar):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            pbar.set_postfix({
                'loss': running_loss / (batch_idx + 1),
                'acc': 100. * correct / total
            })

    val_loss = running_loss / len(dataloader)
    val_acc = 100. * correct / total

    return val_loss, val_acc


def validate_urfound(model, dataloader, criterion, device):
    """Validate the model"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        pbar = tqdm(dataloader, desc='Validation')
        for batch_idx, (images, labels, sources) in enumerate(pbar):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            pbar.set_postfix({
                'loss': running_loss / (batch_idx + 1),
                'acc': 100. * correct / total
            })

    val_loss = running_loss / len(dataloader)
    val_acc = 100. * correct / total

    return val_loss, val_acc


def show_images(dataset, train_labels, num_images, start_idx=0):

    cols = 4
    rows = (num_images + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(20, 10))
    axes = axes.flatten()

    for plot_idx, data_idx in enumerate(range(start_idx, start_idx + num_images)):
        if data_idx >= len(dataset):  # Safety check
            break

        image, label, dataset_name = dataset[data_idx]

        img = image.clone()
        img[0] = img[0] * 0.229 + 0.485
        img[1] = img[1] * 0.224 + 0.456
        img[2] = img[2] * 0.225 + 0.406
        img = torch.clamp(img, 0, 1)

        axes[plot_idx].imshow(img.permute(1, 2, 0))
        axes[plot_idx].set_title(f"Dataset: {dataset_name} \n Label: {label}", fontsize=12, pad=5)
        axes[plot_idx].axis('off')

    for idx in range(num_images, len(axes)):
        axes[idx].axis('off')

    plt.tight_layout()
    plt.show()


def get_specific_layer_names(model):
    # Create a list to store the layer names
    layer_names = []

    # Recursively visit all modules and submodules
    for name, module in model.named_modules():
        # Check if the module is an instance of the specified layers
        if isinstance(module, (torch.nn.Linear, torch.nn.Embedding, torch.nn.Conv2d, Conv1D)):
            # model name parsing

            layer_names.append('.'.join(name.split('.')[4:]).split('.')[0])

    return layer_names
