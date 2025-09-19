import torch
from torchvision import transforms
from PIL import Image
from torch.utils.data import DataLoader
from dataset import KoreanFoodDataset
from model import KoreanFoodClassifier

# ----------------------------
# Define your food hierarchy
# ----------------------------
group_names = ["Jeongol", "Jjigae", "Tang"]  # example food groups
num_dishes_per_group = [2, 2, 2]  # adjust according to your dataset
dish_names = [
    ["Budae_jeongol", "Haemul_jeongol"],      # Jeongol
    ["Kimchi_jjigae", "Doenjang_jjigae"],     # Jjigae
    ["Samgyetang", "Seolleongtang"],          # Tang
]

# ----------------------------
# Load the trained model
# ----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = KoreanFoodClassifier(len(group_names), num_dishes_per_group)
model.load_state_dict(torch.load("best_model.pth", map_location=device))
model.to(device)
model.eval()
print("✅ Model loaded for prediction & evaluation.")

# ----------------------------
# Preprocessing
# ----------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# ----------------------------
# Single Image Prediction
# ----------------------------
def predict_image(image_path):
    image = Image.open(image_path).convert("RGB")
    img_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        group_logits, dish_logits_list = model(img_tensor)

        # Group prediction
        group_idx = torch.argmax(group_logits, dim=1).item()
        group_label = group_names[group_idx]

        # Dish prediction (within predicted group)
        dish_logits = dish_logits_list[group_idx]
        dish_idx = torch.argmax(dish_logits, dim=1).item()
        dish_label = dish_names[group_idx][dish_idx]

    print(f"\n📷 Image: {image_path}")
    print(f"Predicted Food Group: {group_label}")
    print(f"Predicted Dish: {dish_label}")
    return group_idx, dish_idx

# ----------------------------
# Dataset Evaluation
# ----------------------------
def evaluate_dataset(test_dir="./images/test"):
    test_dataset = KoreanFoodDataset(test_dir, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    correct_groups, total_groups = 0, 0
    correct_dishes, total_dishes = 0, 0

    with torch.no_grad():
        for images, group_labels, dish_labels in test_loader:
            images = images.to(device)
            group_labels = group_labels.to(device)
            dish_labels = dish_labels.to(device)

            group_logits, dish_logits_list = model(images)

            # Group prediction
            group_preds = torch.argmax(group_logits, dim=1)
            correct_groups += (group_preds == group_labels).sum().item()
            total_groups += group_labels.size(0)
            print(f"total_groups: ", total_groups)

            # Dish prediction (use predicted group)
            dish_preds = []
            for i, g in enumerate(group_preds):
                dish_pred = torch.argmax(dish_logits_list[g][i]).item()
                dish_preds.append(dish_pred)

            dish_preds = torch.tensor(dish_preds, device=device)
            correct_dishes += (dish_preds == dish_labels).sum().item()
            total_dishes += dish_labels.size(0)

    group_acc = correct_groups / total_groups * 100
    dish_acc = correct_dishes / total_dishes * 100
    print(f"\n Evaluation Results on {test_dir}:")
    print(f"Group Accuracy: {group_acc:.2f}%")
    print(f"Dish Accuracy: {dish_acc:.2f}%")

    #TODO: need to make sure accuracy is calculated correctly --> currently not.

# ----------------------------
# Run Tests
# ----------------------------
if __name__ == "__main__":
    # Example: predict on a couple of new images
    test_images = [
        "./images/test/Jjigae/Kimchi_jjigae/test_kimchijjigae.jpg",
        "./images/test/Tang/Seolleongtang/test_seoulleongtang.jpg"
    ]
    for img_path in test_images:
        predict_image(img_path)

    # Evaluate on the full test dataset
    evaluate_dataset("./images/test")
