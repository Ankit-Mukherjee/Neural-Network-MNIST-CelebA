# Import necessary libraries
import matplotlib.pyplot as plt
import torch
from torch import nn
from torchvision import datasets
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np
from sklearn.metrics import confusion_matrix
import time
from datetime import timedelta

# Image size for MNIST dataset
img_size = 28
img_shape = (img_size, img_size)  # Tuple with height and width of images

# Define CNN model
def create_cnn():
    class cnn(nn.Module):
        def __init__(self):
            super().__init__()

            # Convolutional Layer 1
            filter_size1 = 5
            num_filters1 = 16

            # Convolutional Layer 2
            filter_size2 = 5
            num_filters2 = 36

            # Fully-connected layer
            fc_size = 128

            # Number of color channels for the images: 1 channel for gray-scale
            num_channels = 1

            # Number of classes, one class for each of 10 digits
            num_classes = 10

            self.layer_conv1 = nn.Conv2d(num_channels, num_filters1, (filter_size1, filter_size1), padding='same')
            self.bn1 = nn.BatchNorm2d(num_filters1)  # Batch Normalization
            self.layer_conv2 = nn.Conv2d(num_filters1, num_filters2, (filter_size2, filter_size2), padding='same')
            self.bn2 = nn.BatchNorm2d(num_filters2)  # Batch Normalization
            self.pool = nn.MaxPool2d(2)
            self.layer_fc1 = nn.Linear(1764, fc_size)
            self.layer_fc2 = nn.Linear(fc_size, num_classes)

        def forward(self, x):
            x = self.pool(F.relu(self.bn1(self.layer_conv1(x))))  # Add BatchNorm and ReLU
            x = self.pool(F.relu(self.bn2(self.layer_conv2(x))))  # Add BatchNorm and ReLU
            x = torch.flatten(x, 1)
            x = F.relu(self.layer_fc1(x))
            x = self.layer_fc2(x)
            return x

    return cnn()


# Training function
def train(dataloader, model, loss_fn, optimizer, scheduler=None):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X.float())
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if scheduler:
            scheduler.step()  # Update the learning rate

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

# Test function
def test(dataloader, model, loss_fn, show_example_errors=False, show_confusion_matrix=False):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0

    # Collecting misclassification examples for plotting
    mis_example, mis_example_pred, mis_example_true = [], [], []
    # Collecting labels for confusion matrix
    y_true, y_pred = [], []

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X.float())
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

            mis_example.extend(X[(pred.argmax(1) == y) == False])
            mis_example_pred.extend(pred[(pred.argmax(1) == y) == False].argmax(1))
            mis_example_true.extend(y[(pred.argmax(1) == y) == False])

            y_true.extend(y.cpu().numpy())
            y_pred.extend(pred.argmax(1).cpu().numpy())

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

    # Plot mis-classifications if desired
    if show_example_errors:
        print("Example errors:")
        plot_images(images=mis_example[:9], cls_true=mis_example_true[:9], cls_pred=mis_example_pred[:9])

    # Plot confusion matrix if desired
    if show_confusion_matrix:
        print("Confusion Matrix:")
        plot_confusion_matrix(cls_pred=y_pred, cls_true=y_true)


# Function to plot images
def plot_images(images, cls_true, cls_pred=None):
    assert len(images) == len(cls_true) == 9

    # Create figure with 3x3 sub-plots
    fig, axes = plt.subplots(3, 3)
    fig.subplots_adjust(hspace=0.3, wspace=0.3)

    for i, ax in enumerate(axes.flat):
        ax.imshow(images[i].reshape(img_shape), cmap='binary')

        if cls_pred is None:
            xlabel = "True: {0:.0f}".format(cls_true[i])
        else:
            xlabel = "True: {0:.0f}, Pred: {1:.0f}".format(cls_true[i], cls_pred[i])

        ax.set_xlabel(xlabel)
        ax.set_xticks([])
        ax.set_yticks([])

    plt.show()

# Function to plot confusion matrix
def plot_confusion_matrix(cls_pred, cls_true):
    cm = confusion_matrix(y_true=cls_true, y_pred=cls_pred)
    print(cm)
    plt.matshow(cm)
    plt.colorbar()
    tick_marks = np.arange(10)
    plt.xticks(tick_marks, range(10))
    plt.yticks(tick_marks, range(10))
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

# Function to optimize training
def optimize(iterations, train_dataloader, model, cost, optimizer, scheduler=None):
    start_time = time.time()
    for _ in range(iterations):
        train(train_dataloader, model, cost, optimizer, scheduler)
    end_time = time.time()
    time_dif = end_time - start_time
    print("Time usage: " + str(timedelta(seconds=int(round(time_dif)))))


# Define hyperparameters and initialize device
learning_rate = 1e-4
train_batch_size = 64
test_batch_size = 256
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

# Initialize the CNN model, loss function, and optimizer
model = create_cnn().to(device)
cost = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Define learning rate scheduler
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

# Load the MNIST dataset with augmentation for training set
train_transform = transforms.Compose([
    transforms.RandomRotation(10),  # Random rotation
    transforms.RandomHorizontalFlip(),  # Random horizontal flip
    transforms.ToTensor(),
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
])

trainset = datasets.MNIST(root='./data', train=True, download=True, transform=train_transform)
testset = datasets.MNIST(root='./data', train=False, transform=test_transform)
train_dataloader = DataLoader(trainset, batch_size=train_batch_size, shuffle=True)
test_dataloader = DataLoader(testset, batch_size=test_batch_size, shuffle=False)

# Initial test evaluation
test(test_dataloader, model, cost)

# Train the model with optimization for 1 iteration
optimize(1, train_dataloader, model, cost, optimizer, scheduler)
test(test_dataloader, model, cost, show_example_errors=True, show_confusion_matrix=True)

# Continue training for 9 more iterations
optimize(9, train_dataloader, model, cost, optimizer, scheduler)
test(test_dataloader, model, cost, show_example_errors=False)