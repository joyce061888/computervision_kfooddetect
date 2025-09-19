import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from dataset import KoreanFoodDataset
from model import KoreanFoodClassifier
import os

# -----------------------
# Step 1: Dataset + DataLoader
# -----------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])
train_dataset = KoreanFoodDataset("./images/train", transform=transform)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

print(f"Number of training samples: {len(train_dataset)}")
print(f"Sample labels (first 5): {train_dataset.samples[:5]}")

# Figure out number of groups and dishes per group dynamically
num_groups = len(train_dataset.group_to_idx)
num_dishes_per_group = []
for group_name in sorted(train_dataset.group_to_idx.keys()):
    dish_count = len([k for k in train_dataset.dish_to_idx.keys() if k[0] == group_name])
    num_dishes_per_group.append(dish_count)

# -----------------------
# Step 2: Model
# -----------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = KoreanFoodClassifier(num_groups, num_dishes_per_group).to(device)
print(f"Model loaded on device: {device}")

# -----------------------
# Step 3: Loss & Optimizer
# -----------------------
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion_group = torch.nn.CrossEntropyLoss()
criterion_dish = torch.nn.CrossEntropyLoss()

# -----------------------
# Step 4: Training loop
# -----------------------
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0.0

    for batch_idx, (images, group_labels, dish_labels) in enumerate(train_loader):
        images = images.to(device)
        group_labels = group_labels.to(device)
        dish_labels = dish_labels.to(device)

        optimizer.zero_grad()
        group_logits, dish_logits_list = model(images)

        # Group-level loss
        loss_group = criterion_group(group_logits, group_labels)

        # Dish-level loss: use the correct classifier for each sample
        dish_losses = []
        for i, g in enumerate(group_labels):
            logits = dish_logits_list[g][i].unsqueeze(0)  # select classifier for this group
            label = dish_labels[i].unsqueeze(0)
            dish_losses.append(criterion_dish(logits, label))
        loss_dish = torch.stack(dish_losses).mean()

        loss = loss_group + loss_dish
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    avg_loss = epoch_loss / len(train_loader)
    print(f"Epoch {epoch+1}/{num_epochs} - Avg Loss: {avg_loss:.4f}")

# -----------------------
# Step 5: Save Model
# -----------------------
torch.save(model.state_dict(), "best_model.pth")
print("Model saved as best_model.pth")
