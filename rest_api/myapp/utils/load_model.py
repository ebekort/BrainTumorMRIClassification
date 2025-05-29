from myapp.models.model import Model
import torch
import os
from django.conf import settings

# Global variable to hold the model
model = None

def load_model():
    global model
    if model is None:
        model_path = os.path.join(settings.BASE_DIR, 'myapp/models/model(7).pth')
        model = Model()  # Initialize your model class
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model.eval()  # Set to evaluation mode
        # If using GPU, uncomment the following line
        # model.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    return model

def predict(input_data):
    """
    Function to make predictions with the model.
    input_data: Preprocessed input (e.g., tensor).
    """
    model = load_model()
    with torch.no_grad():
        output = model(input_data)
    return output