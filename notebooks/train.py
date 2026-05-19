from data import train_loader, X_test, y_test
from model import NeuralNetwork
import torch
import torch.nn as nn

model = NeuralNetwork()
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(100):
    model.train()
    epoch_loss = 0
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        predictions = model(X_batch)
        loss = criterion(predictions, y_batch)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    model.eval()
    with torch.no_grad():
        test_predictions = model(X_test)
        test_loss = criterion(test_predictions, y_test)
        
        # accuracy
        probs = torch.sigmoid(test_predictions)
        predicted_labels = (probs > 0.5).float()
        accuracy = (predicted_labels == y_test).float().mean()
    
    print(f'Epoch {epoch} | Train Loss: {epoch_loss/len(train_loader):.4f} | Test Loss: {test_loss:.4f} | Accuracy: {accuracy*100:.1f}%')
    

