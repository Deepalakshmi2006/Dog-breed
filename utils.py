import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image

# Load pre-trained ResNet-50
model = models.resnet50(pretrained=True)
model.eval()

# ImageNet class labels
from torchvision.models import ResNet50_Weights
labels = ResNet50_Weights.DEFAULT.meta["categories"]

# Transform input image
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
])

def predict_dog_breed(img: Image.Image):
    img_t = transform(img).unsqueeze(0)
    with torch.no_grad():
        output = model(img_t)
        _, idx = torch.max(output, 1)
        breed = labels[idx.item()]
        confidence = torch.nn.functional.softmax(output, dim=1)[0][idx.item()].item() * 100
    return breed, round(confidence, 2)
