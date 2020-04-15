# -*- coding: utf-8 -*-
import torch
import numpy as np
import sys
if len(sys.argv) != 2:
    print("Usage:", sys.argv[0], "<num threads>")
    sys.exit(0)
torch.set_num_threads(int(sys.argv[1]))
torch.set_num_interop_threads(min(int(sys.argv[1]), 4))

train_x = np.load("train_images.npy")
train_y = np.load("train_labels.npy")
print(train_x.shape)

# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
N = 6000
H = 28
W = 28
train_x = train_x[:N, :]
train_x = train_x.reshape((N, 1, H, W))
train_y = train_y[:N].ravel()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


# Create random Tensors to hold inputs and outputs
x = torch.tensor(train_x, device=device, dtype=torch.float32)
y = torch.tensor(train_y, device=device, dtype=torch.long)
x.requires_grad = False
y.requires_grad = False

class Flatten(torch.nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

# Use the nn package to define our model as a sequence of layers. nn.Sequential
# is a Module which contains other Modules, and applies them in sequence to
# produce its output. Each Linear Module computes output from input using a
# linear function, and holds internal Tensors for its weight and bias.
model = torch.nn.Sequential(
    torch.nn.Conv2d(1, 3, 3, stride=1, padding=1),
    torch.nn.BatchNorm2d(3),
    torch.nn.ReLU(),

    torch.nn.Conv2d(3, 8, 3, stride=2, padding=1),
    Flatten(),
    torch.nn.Linear(1568, 10),
)

# The nn package also contains definitions of popular loss functions; in this
# case we will use Mean Squared Error (MSE) as our loss function.
loss_fn = torch.nn.CrossEntropyLoss()

learning_rate = 1e-4
for t in range(100):
    # Forward pass: compute predicted y by passing x to the model. Module objects
    # override the __call__ operator so you can call them like functions. When
    # doing so you pass a Tensor of input data to the Module and it produces
    # a Tensor of output data.
    y_pred = model(x)

    # Compute and print loss. We pass Tensors containing the predicted and true
    # values of y, and the loss function returns a Tensor containing the
    # loss.
    loss = loss_fn(y_pred, y)
    # if t % 100 == 99:
    print(t, loss.item())

    # Zero the gradients before running the backward pass.
    model.zero_grad()

    # Backward pass: compute gradient of the loss with respect to all the learnable
    # parameters of the model. Internally, the parameters of each Module are stored
    # in Tensors with requires_grad=True, so this call will compute gradients for
    # all learnable parameters in the model.
    loss.backward()

    # Update the weights using gradient descent. Each parameter is a Tensor, so
    # we can access its gradients like we did before.
    with torch.no_grad():
        for param in model.parameters():
            param -= learning_rate * param.grad
