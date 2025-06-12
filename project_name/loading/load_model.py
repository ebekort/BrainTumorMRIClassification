#This file loads the model from the rest_api to the local computer

import torch
import os
from project_name.models.main_model import Model

import torch
import os
from project_name.models.main_model import Model

def run():
    source_path = './rest_api/myapp/models/model(7).pth'
    dest_folder = './models'
    dest_path = os.path.join(dest_folder, 'model7.pth')

    os.makedirs(dest_folder, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Model()
    model.load_state_dict(torch.load(source_path, map_location=device))
    model.to(device)

    torch.save(model.state_dict(), dest_path)
    print(f'Model loaded from {source_path} to {dest_path}')

if __name__ == '__main__':
    run()

