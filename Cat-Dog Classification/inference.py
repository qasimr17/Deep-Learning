import torch
from torchvision import transforms
from PIL import Image


def transform_image(image):
    device = "cuda" if torch.cuda.is_available else "cpu"
    img = Image.open(image)
    img_dimensions = 224
    img_transforms = transforms.Compose([
        transforms.Resize((img_dimensions,img_dimensions)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225] )
        ])
    img = img_transforms(img).unsqueeze(0).to(device)
    return img

def predict(model, image):
    """Takes in a classifaction model and an image tensor and returns
    the prediction of the model on the image - Dog vs Cat."""
    image = transform_image(image)
    output = model(image)
    prediction = torch.argmax(output).item()
    prob = torch.sigmoid(output.data)[0][prediction].item()
    
    if prediction == 1:
        return "Dog", prob
    else:
        return "Cat", prob




