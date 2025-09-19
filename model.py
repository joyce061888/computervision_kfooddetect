import torch
import torch.nn as nn
import torchvision.models as models

class KoreanFoodClassifier(nn.Module):
    def __init__(self, num_groups, num_dishes_per_group):
        """
        num_groups: number of food groups (Level 1)
        num_dishes_per_group: list of number of dishes per group (Level 2)
        """
        super(KoreanFoodClassifier, self).__init__()

        # Backbone CNN (pretrained ResNet18)
        self.backbone = models.resnet18(weights=None)
        # Load local pretrained weights
        state_dict = torch.load("resnet18-f37072fd.pth")
        self.backbone.load_state_dict(state_dict)

        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()  # remove original classification head

        # Level 1: Food group classifier
        # This is a new fully connected layer that takes the feature vector (512-dim) and outputs num_groups logits.
        # Each logit corresponds to a food group (Jeongol, Jjigae, Tang, etc.).
        self.group_classifier = nn.Linear(in_features, num_groups)

        # Level 2: one classifier per food group
        self.dish_classifiers = nn.ModuleList(
            [nn.Linear(in_features, n) for n in num_dishes_per_group]
        )

    def forward(self, x):
        features = self.backbone(x)

        # Level 1
        group_logits = self.group_classifier(features)

        # Level 2: only return dish logits for each group
        # during training, we select based on the ground truth group label
        dish_logits_list = [classifier(features) for classifier in self.dish_classifiers]

        return group_logits, dish_logits_list
