import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import OxfordIIITPet
import segmentation_models_pytorch as smp
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.tensorboard import SummaryWriter

# Dataset Class
class PetSegmentationDataset:
    def __init__(self, root='./data', split='train', transform=None, mask_transform=None):
        is_trainval = split in ['train', 'val']
        self.dataset = OxfordIIITPet(
            root=root,
            split='trainval' if is_trainval else 'test',
            target_types='segmentation',
            download=True,
            transform=None
        )
        self.split = split
        self.transform = transform
        self.mask_transform = mask_transform

        if split in ['train', 'val']:
            train_size = int(0.8 * len(self.dataset))
            val_size = len(self.dataset) - train_size
            train_dataset, val_dataset = random_split(self.dataset, [train_size, val_size])
            self.dataset = train_dataset if split == 'train' else val_dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, mask = self.dataset[idx]
        resize = transforms.Resize((256, 256))
        img = resize(img)
        mask = resize(mask)
        img = transforms.ToTensor()(img)
        
        # Normalize mask values: [1, 2, 3] -> [0, 1, 2]
        mask = np.array(mask)
        mask = mask - 1
        mask = torch.from_numpy(mask).long()
        
        return img, mask


# IOU Calculation
def calculate_iou(preds, masks, num_classes):
    iou_scores = []
    for cls in range(num_classes):
        pred_cls = preds == cls
        mask_cls = masks == cls
        intersection = (pred_cls & mask_cls).sum().float()
        union = (pred_cls | mask_cls).sum().float()
        if union > 0:
            iou_scores.append((intersection / union).item())
    return np.mean(iou_scores) if iou_scores else 0.0

# Training Configuration
from segmentation_models_pytorch.losses import DiceLoss
class HybridLoss(nn.Module):
    def __init__(self):
        super(HybridLoss, self).__init__()
        self.dice_loss = DiceLoss(mode='multiclass')
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, preds, targets):
        dice = self.dice_loss(preds, targets)
        ce = self.cross_entropy(preds, targets)
        return dice + ce  # 원하는 가중치를 곱할 수도 있음


class SegmentationTrainer:
    def __init__(self, model, device='cuda', log_dir='./logs'):
        self.model = model.to(device)
        self.device = device
        self.criterion = HybridLoss() 
        self.optimizer = optim.Adam(model.parameters(), lr=1e-4)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=5, gamma=0.5)
        self.writer = SummaryWriter(log_dir=log_dir)
        self.num_classes = model.segmentation_head[0].out_channels



    def train_epoch(self, dataloader, epoch, phase='train'):
        self.model.train() if phase == 'train' else self.model.eval()
        total_loss = 0
        iou_scores = []

        with torch.set_grad_enabled(phase == 'train'):
            for images, masks in tqdm(dataloader, desc=f"{phase.capitalize()} Epoch {epoch+1}", ncols=120):
                images, masks = images.to(self.device), masks.to(self.device)

                outputs = self.model(images)
                loss = self.criterion(outputs, masks)
                total_loss += loss.item()

                preds = torch.argmax(outputs, dim=1)
                iou = calculate_iou(preds, masks, self.num_classes)
                iou_scores.append(iou)

                if phase == 'train':
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

        mean_loss = total_loss / len(dataloader)
        mean_iou = np.mean(iou_scores)

        self.writer.add_scalar(f'{phase}/Loss', mean_loss, epoch)
        self.writer.add_scalar(f'{phase}/IOU', mean_iou, epoch)

        if phase == 'train':
            self.scheduler.step()

        return mean_loss, mean_iou

# Inference Example
def inference_from_dataset(model, dataloader, output_path, device='cuda:1'):
    model.eval()
    with torch.no_grad():
        for img, mask in dataloader:
            img, mask = img.to(device), mask.to(device)
            output = model(img)
            
            # Get the class with the highest probability for each pixel
            preds = torch.argmax(output, dim=1).cpu().numpy()

            # Check unique values to confirm the output range
            print(f'preds_class_value: {np.unique(preds)}')

            # Save the result with proper visualization
            plt.imshow(preds[0], cmap='viridis', vmin=0, vmax=2)
            plt.axis('off')
            plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
            plt.close()
            break



# Main Training Loop
def main():
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    mask_transform = transforms.Compose([
        transforms.Resize((256, 256)),
    ])

    # Create datasets with transformations
    train_dataset = PetSegmentationDataset(split='train', transform=train_transform, mask_transform=mask_transform)
    val_dataset = PetSegmentationDataset(split='val', transform=train_transform, mask_transform=mask_transform)

    # DataLoader
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=4)

    # Model setup
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=3,
        classes=3
    )

    # Trainer setup
    trainer = SegmentationTrainer(model, device=device)

    # Training loop
    num_epochs = 20
    best_val_iou = 0.0

    train_losses, val_losses = [], []
    train_ious, val_ious = [], []

    for epoch in range(num_epochs):
        train_loss, train_iou = trainer.train_epoch(train_loader, epoch, phase='train')
        val_loss, val_iou = trainer.train_epoch(val_loader, epoch, phase='val')

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_ious.append(train_iou)
        val_ious.append(val_iou)

        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Train IOU: {train_iou:.4f}')
        print(f'Epoch {epoch+1}/{num_epochs}, Val Loss: {val_loss:.4f}, Val IOU: {val_iou:.4f}')

        # Save the best model based on validation IOU
        if val_iou > best_val_iou:
            best_val_iou = val_iou
            torch.save(model.state_dict(), 'best_model.pth')
            print(f"Best model saved with IOU: {best_val_iou:.4f}")

    # Plot losses and IOUs
    epochs = range(1, num_epochs + 1)
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label='Train Loss', marker='o')
    plt.plot(epochs, val_losses, label='Val Loss', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_ious, label='Train IOU', marker='o')
    plt.plot(epochs, val_ious, label='Val IOU', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('IOU')
    plt.title('Training and Validation IOU')
    plt.legend()

    plt.tight_layout()
    plt.show()



# Inference Example (입력 이미지 경로와 출력 경로 설정)
def main_inference():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Model setup
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights=None,
        in_channels=3,
        classes=3
    )
    model.load_state_dict(torch.load("best_model.pth", map_location=device))
    model = model.to(device)

    # Inference with dataset
    val_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    val_dataset = PetSegmentationDataset(split='val', transform=val_transform)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    output_path = "segmentation_result.png"
    inference_from_dataset(model, val_loader, output_path, device)
    print("Inference completed and saved to:", output_path)

if __name__ == "__main__":
    main()
