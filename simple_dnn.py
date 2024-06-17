import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms

# Setting random seed for reproducibility
torch.manual_seed(42)

# Hyperparameters
input_size = 28 * 28  # MNIST images are 28x28 pixels
hidden_size = 128
output_size = 10
learning_rate = 0.005
batch_size = 64
epochs = 10
test_correct, test_total = 0, 0

class SimpleNN(nn.Module):
    """
        Baseline DNN definition 
    """
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


def load_data(batch_size):
    transform = transforms.Compose([transforms.ToTensor(), 
                                    transforms.Normalize((0.5,), (0.5,))])
    
    train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


def train(train_loader):
    model = SimpleNN(input_size, hidden_size, output_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        for inputs, labels in train_loader:
            inputs = inputs.view(-1, input_size)  
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')
    return model

def evaluate(model, test_loader):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.view(-1, input_size)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    print(f'Test Accuracy: {accuracy * 100:.2f}%')
    print(f'Test Correct/Total: {correct}/{total}')
    return correct, total, accuracy


def main():
    train_loader, test_loader = load_data(batch_size)
    model = train(train_loader)
    test_correct, test_total, accuracy = evaluate(model, test_loader)
    torch.save(model.state_dict(), './models/base-model.pt')

if __name__ == "__main__":
    main()

