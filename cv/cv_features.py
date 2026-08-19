import torch
from torchvision import models, transforms
from PIL import Image

print("Loading CV model...")

model = models.resnet18(weights="DEFAULT")

# Remove final classification layer
model.fc = torch.nn.Identity()

model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

print("CV model loaded!")


def get_image_features(image_path):

    image = Image.open(image_path).convert("RGB")

    image = transform(image)
    image = image.unsqueeze(0)

    with torch.no_grad():
        features = model(image)

    return features


image_path = input("\nEnter image path: ")

features = get_image_features(image_path)

print("\nCV features generated")
print("Shape:", features.shape)
print("Feature size:", features.shape[-1])
torch.save(
    features,
    "data/cv/cv_features.pt"
)

print("\nSaved:")
print("data/cv/cv_features.pt")