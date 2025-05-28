import torch
from tqdm import tqdm
from sklearn.metrics import f1_score

def train_model(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=25):
    """
    Train the model.
    
    Args:
        model: The model to train.
        train_loader: DataLoader for training data.
        val_loader: DataLoader for validation data.
        criterion: Loss function.
        optimizer: Optimizer.
        num_epochs: Number of epochs to train.
    """
    train_losses = []
    val_losses = []
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs},", leave=True)
        for batch, (images, labels) in enumerate(progress_bar):
            # add loading bar that shows the current batch and the total number of batches with tqdm

            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            progress_bar.set_postfix({
            "batch": f"{batch+1}/{len(train_loader)}",
            "loss": f"{loss.item():.4f}"
        })
            
        epoch_loss = running_loss / len(train_loader)

        print(f'Epoch: {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}')

        # Validation
        model.eval()
        val_loss = 0.0
        summed_f1 = 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() 

                _, predicted = torch.max(outputs.data, 1)
                label = torch.argmax(labels, dim=1)
                f1 = f1_score(label, predicted, average='weighted')
                print(f'Validation F1 Score: {f1:.4f}')
                summed_f1 += f1
        summed_f1 /= len(val_loader)
        print(f'Validation F1 Score (Average): {summed_f1:.4f}')
        val_loss /= len(val_loader)
        print(f'Validation Loss: {val_loss:.4f}')

def train_baseline(baseline, train_loader):
    for images, labels in train_loader:
        baseline.partial_fit(images, labels)