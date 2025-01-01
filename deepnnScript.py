import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pickle

# Define single-layer MLP
def create_single_layer_mlp():
    class SingleLayerNet(nn.Module):
        def _init_(self):
            super()._init_()
            n_input = 2376
            n_hidden = 256
            n_classes = 2
            
            # Network layers: single hidden layer and output layer
            self.layer_1 = nn.Linear(n_input, n_hidden)
            self.out_layer = nn.Linear(n_hidden, n_classes)

        def forward(self, x):
            x = F.relu(self.layer_1(x))
            x = self.out_layer(x)
            return x

    return SingleLayerNet()

# Define deep MLP with two hidden layers
def create_deep_mlp():
    class DeepNet(nn.Module):
        def _init_(self):
            super()._init_()
            n_input = 2376
            n_hidden_1 = 256
            n_hidden_2 = 256
            n_classes = 2
            
            # Network layers: two hidden layers and output layer
            self.layer_1 = nn.Linear(n_input, n_hidden_1)
            self.layer_2 = nn.Linear(n_hidden_1, n_hidden_2)
            self.out_layer = nn.Linear(n_hidden_2, n_classes)

        def forward(self, x):
            x = F.relu(self.layer_1(x))
            x = F.relu(self.layer_2(x))
            x = self.out_layer(x)
            return x

    return DeepNet()

# Preprocessing function (no changes needed)
def preprocess():
    pickle_obj = pickle.load(file=open('face_all.pickle', 'rb'))
    features = pickle_obj['Features']
    labels = pickle_obj['Labels']
    train_x = features[0:21100] / 255
    valid_x = features[21100:23765] / 255
    test_x = features[23765:] / 255

    labels = np.squeeze(labels)
    train_y = labels[0:21100]
    valid_y = labels[21100:23765]
    test_y = labels[23765:]

    class dataset(Dataset):
        def __init__(self, X, y):  # Changed to use double underscores
            self.X = X
            self.y = y

        def __len__(self):  # Changed to use double underscores
            return len(self.y)

        def __getitem__(self, idx):  # Changed to use double underscores
            return self.X[idx], self.y[idx]

    trainset = dataset(train_x, train_y)
    validset = dataset(valid_x, valid_y)
    testset = dataset(test_x, test_y)

    return trainset, validset, testset

# Training and evaluation functions
def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        pred = model(X.float())
        loss = loss_fn(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X.float())
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f}")

# Parameters
learning_rate = 0.0001
training_epochs = 50
batch_size = 100

# Device setup
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using {device} device")

# Load data
trainset, validset, testset = preprocess()
train_dataloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
valid_dataloader = DataLoader(validset, batch_size=batch_size, shuffle=False)
test_dataloader = DataLoader(testset, batch_size=batch_size, shuffle=False)

# Training and testing for each model
for model_type in ['Single Layer MLP', 'Deep MLP']:
    print(f"\nTraining {model_type}")
    if model_type == 'Single Layer MLP':
        model = create_single_layer_mlp().to(device)
    else:
        model = create_deep_mlp().to(device)

    # Define loss and optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    # Train model
    for t in range(training_epochs):
        train(train_dataloader, model, loss_fn, optimizer)

    # Evaluate model
    print(f"Results for {model_type}:")
    test(test_dataloader, model, loss_fn)