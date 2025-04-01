import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import requests
from io import BytesIO
from google.cloud import texttospeech

all_transforms = transforms.Compose([
	transforms.Resize((64, 64)),
	transforms.ToTensor(),
	transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class CNN(nn.Module):
	def __init__(self, num_classes):
		super(CNN, self).__init__()
		self.conv_layer1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3)
		self.conv_layer2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)
		self.max_pool1 = nn.MaxPool2d(kernel_size = 2, stride = 2)
		self.conv_layer3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3)
		self.max_pool2 = nn.MaxPool2d(kernel_size = 2, stride = 2)

		self.fc1 = nn.Linear(128 * 14 * 14, 128)
		self.relu1 = nn.ReLU()
		self.fc2 = nn.Linear(128, num_classes)

	def forward(self, x):
		out = self.conv_layer1(x)
		out = self.conv_layer2(out)
		out = self.max_pool1(out)
		out = self.conv_layer3(out)
		out = self.max_pool2(out)

		out = out.reshape(out.size(0), -1)
		out = self.fc1(out)
		out = self.relu1(out)
		out = self.fc2(out)
		return out

def predict(image):
	image = all_transforms(image.convert("RGB")).unsqueeze(0)
	model.eval()

	with torch.no_grad():
		output = model(image)
		predicted_class = torch.argmax(output, dim=1).item()

	return "fresh" if predicted_class == 0 else "rotten"


model = CNN(2)
model.load_state_dict(torch.load(r"/home/vedant/NavSight/model_state_dict.pth", weights_only=True))



from PIL import Image
import requests

response = requests.get("https://www.mashed.com/img/gallery/what-is-okra-and-what-does-it-taste-like/intro-1617380237.jpg")
print(predict(Image.open(BytesIO(response.content))))

