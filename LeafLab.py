from flask import Flask, request, render_template_string, url_for
import torch
from torchvision import models, transforms
from PIL import Image
import io
import base64
import os
import logging

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

app = Flask(__name__)

# Load the model using environment variables
device = "cuda" if torch.cuda.is_available() else "cpu"
logging.info(f"Using device: {device}")
model = models.resnet18(pretrained=True)
model.fc = torch.nn.Linear(in_features=512, out_features=4)  
model.load_state_dict(torch.load(os.getenv('MODEL_PATH', 'model.pth'), map_location=device))
model = model.to(device)
model.eval()

# Define the image transformation
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Define the label mapping and solutions
labels = {0: 'Blight', 1: 'Common_Rust', 2: 'Gray_Leaf_Spot', 3: 'Healthy'}
solutions = {
    0: 'Apply fungicides and remove affected areas.',
    1: 'Use rust-resistant varieties and consider fungicide applications.',
    2: 'Ensure crop rotation and use fungicides as needed.',
    3: 'No action needed, plants are healthy.'
}

def transform_image(image_bytes):
    """Transforms image bytes to a tensor."""
    try:
        image = Image.open(io.BytesIO(image_bytes))
        image = image.convert('RGB')
        return transform(image).unsqueeze(0).to(device)
    except IOError as e:
        logging.error(f"Error processing image: {str(e)}")
        return None

def get_prediction(image_bytes):
    """Get model prediction for given image bytes."""
    tensor = transform_image(image_bytes)
    if tensor is None:
        return None, "Invalid image or unsupported image format."
    with torch.no_grad():
        outputs = model(tensor)
        _, predicted = torch.max(outputs, 1)
        label = labels[predicted.item()]
        solution = solutions[predicted.item()]
        return label, solution

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        file = request.files.get('file')
        if not file:
            return render_template_string(HTML_TEMPLATE, result="No file provided", solution="", image_data="")
        
        image_bytes = file.read()
        label, solution = get_prediction(image_bytes)
        if label is None:
            return render_template_string(HTML_TEMPLATE, result=solution, solution="", image_data="")

        # Convert image bytes to base64 for HTML display
        encoded_image = base64.b64encode(image_bytes).decode('utf-8')
        image_data = f"data:image/jpeg;base64,{encoded_image}"
        return render_template_string(HTML_TEMPLATE, result=label, solution=solution, image_data=image_data)

    return render_template_string(HTML_TEMPLATE, result="Upload an image", solution="", image_data="")


HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Leaf Lab</title>
    <!-- Bootstrap CSS -->
    <link href="https://maxcdn.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css" rel="stylesheet">
    <style>
        body { padding-top: 20px; }
        .container { max-width: 600px; margin: auto; }
        img { max-width: 100%; height: auto; }
    </style>
</head>
<body>
    <div class="container">
        <img src="{{ url_for('static', filename='WhatsApp Image 2024-08-13 at 1.15.08 AM.jpeg') }}" alt="Corn Image" style="width: auto; height: 50%;">
        <h1 class="text-center mb-4">Leaf Lab</h1>
        <form method="post" enctype="multipart/form-data" class="mb-3">
            <div class="custom-file">
                <input type="file" name="file" class="custom-file-input" id="customFile">
                <label class="custom-file-label" for="customFile">Choose file</label>
            </div>
            <button type="submit" class="btn btn-primary mt-2">Predict</button>
        </form>
        {% if image_data %}
            <img src="{{ image_data }}" alt="Uploaded Image" class="img-fluid mb-3">
        {% endif %}
        {% if result %}
            <div class="alert alert-info">Prediction Result: {{ result }}</div>
            <div class="alert alert-success">Solution: {{ solution }}</div>
        {% endif %}
        {% if error %}
            <div class="alert alert-danger">{{ error }}</div>
        {% endif %}
    </div>
    <!-- jQuery and Bootstrap JS -->
    <script src="https://code.jquery.com/jquery-3.5.1.slim.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/@popperjs/core@2.9.2/dist/umd/popper.min.js"></script>
    <script src="https://maxcdn.bootstrapcdn.com/bootstrap/4.5.2/js/bootstrap.min.js"></script>
    <script>
        // Update the text of the custom file label to the selected file name
        $('.custom-file-input').on('change', function() {
            var fileName = $(this).val().split('\\').pop();
            $(this).next('.custom-file-label').html(fileName);
        });
    </script>
</body>
</html>
'''

if __name__ == '__main__':
        app.run(host='0.0.0.0', port=5000, debug=True)
