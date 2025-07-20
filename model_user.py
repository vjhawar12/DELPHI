import torch
import torch.nn as nn
import torchvision.transforms as transforms
import json

with open("params.json", "r") as file:
	params = json.load(file)

img_size = params["img_size"]
cl1_dim = params["conv_layer_1_dim"]
cl2_dim = params["conv_layer_2_dim"]
cl3_dim = params["conv_layer_3_dim"]
fc1_dim = params["fc1_dim"]

all_transforms = transforms.Compose([
	transforms.Resize((img_size, img_size)),
	transforms.ToTensor(),
	transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class CNN(nn.Module):
	def __init__(self, num_classes):
		super(CNN, self).__init__()
		self.conv_layer1 = nn.Conv2d(in_channels=3, out_channels=cl1_dim, kernel_size=3)
		self.conv_layer2 = nn.Conv2d(in_channels=cl1_dim, out_channels=cl2_dim, kernel_size=3)
		self.max_pool1 = nn.MaxPool2d(kernel_size = 2, stride = 2)
		self.conv_layer3 = nn.Conv2d(in_channels=cl2_dim, out_channels=cl3_dim, kernel_size=3)
		self.max_pool2 = nn.MaxPool2d(kernel_size = 2, stride = 2)

		self.fc1 = nn.Linear(cl3_dim * 14 * 14, fc1_dim)
		self.relu1 = nn.ReLU()
		self.fc2 = nn.Linear(fc1_dim, num_classes)

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


