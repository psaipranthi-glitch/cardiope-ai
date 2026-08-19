from PIL import Image
from torchvision import transforms


IMAGE_PATH = "data/cv/sample_xray.jpg"


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


image = Image.open(IMAGE_PATH).convert("RGB")

image = transform(image)

image = image.unsqueeze(0)

print("Image preprocessing successful")
print("Shape:", image.shape)