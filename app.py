from flask import Flask, request, render_template
import torch
from torchvision import transforms
from PIL import Image
import os

# Initialize the Flask app
app = Flask(__name__)

# Define the WLeafNet model (reuse the class from your training code)
class WLeafNet(torch.nn.Module):
    def __init__(self, num_classes=19):
        super(WLeafNet, self).__init__()
        self.cbr1 = self._create_cbr_layer(3, 64)
        self.mp1 = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.cbr2 = self._create_cbr_layer(64, 128)
        self.mp2 = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.cbr3 = self._create_cbr_layer(128, 160)
        self.mp3 = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.cbr4 = self._create_cbr_layer(160, 192)
        self.mp4 = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.cbr5 = self._create_cbr_layer(192, 224)
        self.mp5 = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.cbr6 = self._create_cbr_layer(224, 320)
        self.mp6 = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.cbr7 = self._create_cbr_layer(320, 256)
        self.fc = torch.nn.Linear(256 * 3 * 3, num_classes)
        self.apply(self._init_weights)

    def _create_cbr_layer(self, in_channels, out_channels):
        return torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(inplace=True)
        )

    def _init_weights(self, layer):
        if isinstance(layer, torch.nn.Conv2d) or isinstance(layer, torch.nn.Linear):
            torch.nn.init.xavier_uniform_(layer.weight)
        if hasattr(layer, 'bias') and layer.bias is not None:
            layer.bias.data.fill_(0.0)

    def forward(self, x):
        x = self.cbr1(x)
        x = self.mp1(x)
        x = self.cbr2(x)
        x = self.mp2(x)
        x = self.cbr3(x)
        x = self.mp3(x)
        x = self.cbr4(x)
        x = self.mp4(x)
        x = self.cbr5(x)
        x = self.mp5(x)
        x = self.cbr6(x)
        x = self.mp6(x)
        x = self.cbr7(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# Load the trained model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = WLeafNet(num_classes=19)
model_path = './wleafnet_model_epoch_20.pth'

if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found: {model_path}")

model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()
model.to(device)

# Define the image preprocessing pipeline
transform = transforms.Compose([
    transforms.Resize((196, 196)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Class index-to-name mapping
class_mapping = {
    0: 'Asafoetida',
    1: 'Bay Leaf',
    2: 'Black Cardamom',
    3: 'Black Pepper',
    4: 'Caraway seeds',
    5: 'Cinnamom stick',
    6: 'Cloves',
    7: 'Coriander Seeds',
    8: 'Cubeb Pepper',
    9: 'Cumin seeds',
    10: 'Dry Ginger',
    11: 'Dry red Chilly',
    12: 'Fennel seeds',
    13: 'Green Cardamom',
    14: 'Mace',
    15: 'Nutmeg',
    16: 'Poppy Seeds',
    17: 'Star Anise',
    18: 'Stone Flowers'
}

# Define the classification route
@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # Check if a file was uploaded
        file = request.files.get('file')
        if not file or file.filename == '':
            return "No file uploaded or invalid file"

        # Save the uploaded image
        upload_dir = 'uploads'
        os.makedirs(upload_dir, exist_ok=True)
        file_path = os.path.join(upload_dir, file.filename)
        file.save(file_path)

        # Classify the image
        try:
            image = Image.open(file_path).convert('RGB')
            input_tensor = transform(image).unsqueeze(0).to(device)
            with torch.no_grad():
                outputs = model(input_tensor)
                _, predicted = outputs.max(1)

            # Get the class name from the class mapping
            predicted_class_index = predicted.item()
            predicted_class = class_mapping.get(predicted_class_index, "Unknown class")
            return f"Predicted Class: {predicted_class}"
        except Exception as e:
            return f"An error occurred: {e}"
        finally:
            # Cleanup uploaded file
            if os.path.exists(file_path):
                os.remove(file_path)

    return '''
    <!doctype html>
    <title>Upload Image for Classification</title>
    <h1>Upload an Image</h1>
    <form method="post" enctype="multipart/form-data">
      <input type="file" name="file">
      <input type="submit" value="Upload">
    </form>
    '''

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')

