import torch
from torch.utils.data import Dataset
from PIL import Image
import os

class KoreanFoodDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.transform = transform
        self.samples = []
        self.group_to_idx = {}
        self.dish_to_idx = {}

        groups = sorted([d for d in os.listdir(root_dir) if not d.startswith(".")])
        print("Group mapping:", {i: g for i, g in enumerate(groups)})

        for group_idx, group_name in enumerate(groups):
            group_path = os.path.join(root_dir, group_name)
            if not os.path.isdir(group_path) or group_name.startswith("."):
                continue  

            self.group_to_idx[group_name] = group_idx
            dishes = sorted([d for d in os.listdir(group_path) if not d.startswith(".")])

            for dish_idx, dish_name in enumerate(dishes):
                dish_path = os.path.join(group_path, dish_name)
                if not os.path.isdir(dish_path):
                    continue  

                # store mapping (group_name, dish_name) -> local dish index
                self.dish_to_idx[(group_name, dish_name)] = dish_idx

                for img_name in os.listdir(dish_path):
                    if img_name.startswith("."):
                        continue  
                    self.samples.append((
                        os.path.join(dish_path, img_name),
                        group_idx,      # group label
                        dish_idx        # local dish label
                    ))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, group_label, dish_label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, group_label, dish_label
