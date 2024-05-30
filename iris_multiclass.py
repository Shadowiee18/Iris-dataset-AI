import os.path
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch import nn
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

flowers = {0: 'Setosa', 1: 'Verticolor', 2: 'Virginica'}
device = 'cuda' if torch.cuda.is_available() else 'cpu'
df = pd.read_csv(r"iris.csv")
label_encoder = preprocessing.LabelEncoder()
X_data = df.iloc[:, 0:4]
df['variety'] = label_encoder.fit_transform(df['variety'])
y = df['variety']
X = torch.tensor(X_data.values)
y = torch.tensor(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.3)
X_train, X_test, y_train, y_test = (X_train.to(device, torch.float), X_test.to(device, torch.float),
                                    y_train.to(device, torch.long), y_test.to(device, torch.long))


class FlowerModel(nn.Module):
    def __init__(self, input_features, output_features, neurons=16):
        super().__init__()
        # self.layer_1 = nn.Linear(in_features=2, out_features=5)
        # self.layer_2 = nn.Linear(in_features=5, out_features=1)
        torch.manual_seed(42)
        torch.cuda.manual_seed(42)
        self.neuralnetwork = nn.Sequential(
            nn.Linear(in_features=input_features, out_features=neurons),
            nn.ReLU(),
            nn.Linear(in_features=neurons, out_features=neurons),
            nn.ReLU(),
            nn.Linear(in_features=neurons, out_features=output_features)
        )

    def forward(self, x):
        # return self.layer_2(self.layer_1(x))
        return self.neuralnetwork(x)


model = FlowerModel(4, len(y)).to(device)
loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(params=model.parameters(), lr=0.1)


def accuracy_func(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct / len(y_pred)) * 100
    return acc


# Training
epochs = 150
test_loss_data = []
test_accuracy_data = []


def train(model, X_train, X_test, y_train, y_test, epochs):
    best_acc = -np.inf
    best_weights = None
    for epoch in range(epochs):
        model.train()
        # 1. Forward pass
        y_logits = model(X_train)  # model outputs raw logits
        y_pred = torch.softmax(y_logits, dim=1).argmax(
            dim=1)  # go from logits -> prediction probabilities -> prediction labels
        # 2. Calculate loss and accuracy
        loss = loss_function(y_logits, y_train)
        acc = accuracy_func(y_true=y_train,
                            y_pred=y_pred)

        # 3. Optimizer zero grad
        optimizer.zero_grad()

        # 4. Loss backwards
        loss.backward()

        # 5. Optimizer step
        optimizer.step()

        # Testing
        model.eval()
        with torch.inference_mode():
            # 1. Forward pass
            test_logits = model(X_test)
            test_pred = torch.softmax(test_logits, dim=1).argmax(dim=1)
            # 2. Calculate test loss and accuracy
            test_loss = loss_function(test_logits, y_test)
            test_acc = accuracy_func(y_true=y_test,
                                     y_pred=test_pred)
            test_loss_data.append(test_loss.cpu())
            test_accuracy_data.append(test_acc)
            if test_acc > best_acc:
                best_acc = test_acc
                best_weights = model.state_dict()
            last_weights = model.state_dict()
        # Print out what's happening
        if epoch % 10 == 0:
            print(
                f"Epoch: {epoch} | Loss: {loss:.5f}, Acc: {acc:.2f}% |"
                f" Test Loss: {test_loss:.5f}, Test Acc: {test_acc:.2f}%")
    save_weights(best_weights, last_weights, model)


def save_weights(best_weights, last_weights, model):
    path = Path(f'run')
    if not path.exists():
        path.mkdir(parents=True)
    count = len(os.listdir('run'))
    if count == 0:
        path = Path(f'run/train/weights/')
    else:
        path = Path(f'run/train{count}/weights/')
    path.mkdir(parents=True)
    model.load_state_dict(best_weights)
    torch.save(model, os.path.join(path, 'best.pt'))
    model.load_state_dict(last_weights)
    torch.save(model, os.path.join(path, 'last.pt'))


train(model, X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test, epochs=epochs)
plt.figure('Loss')
plt.plot(test_loss_data)
plt.xlabel('epochs')
plt.ylabel('test_loss')
plt.figure('Accuracy')
plt.plot(test_accuracy_data)
plt.xlabel('epochs')
plt.ylabel('test_accuracy')
plt.show()

