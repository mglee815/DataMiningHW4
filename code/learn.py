import argparse
import numpy
import torch
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import random
import matplotlib.pyplot as plt
from yaml import parse

#Using GPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Train with {device}")

# Set seed
random.seed(42)
torch.manual_seed(42)
if device == 'cuda':
    torch.cuda.manual_seed_all(42)

#Get Data Set
mnist_train = dsets.MNIST(root='MNIST_data/',
                          train=True,
                          transform=transforms.ToTensor(),
                          download=True)

mnist_test = dsets.MNIST(root='MNIST_data/',
                         train=False,
                         transform=transforms.ToTensor(),
                         download=True)


#Set parameter

parser = argparse.ArgumentParser()
parser.add_argument('--Question', type = int)
parser.add_argument('--lr', type = float, default = 0.001)

args = parser.parse_args()

Question = args.Question
training_epochs = 50
batch_size = 32
learning_rate  = args.lr

error = torch.nn.CrossEntropyLoss().to(device)
sigmoid = torch.nn.Sigmoid().to(device)

data_loader = torch.utils.data.DataLoader(dataset=mnist_train,
                                          batch_size=batch_size,
                                          shuffle=True,
                                          drop_last=True)
 

if Question == 1:
    linear1 = torch.nn.Linear(784, 10, bias=True)

    torch.nn.init.xavier_uniform_(linear1.weight)

    model = torch.nn.Sequential(linear1).cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

elif Question == 2:
    linear1 = torch.nn.Linear(784, 128, bias=True)
    linear2 = torch.nn.Linear(128, 10, bias=True)

    torch.nn.init.xavier_uniform_(linear1.weight)
    torch.nn.init.xavier_uniform_(linear2.weight)

    model = torch.nn.Sequential(
        linear1, sigmoid,
        linear2
    ).cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


elif Question == 3 or Question == 4:
    print(Question)
    linear1 = torch.nn.Linear(784, 128, bias=True)
    linear2 = torch.nn.Linear(128, 4, bias=True)
    linear3 = torch.nn.Linear(4, 10, bias=True)

    torch.nn.init.xavier_uniform_(linear1.weight)
    torch.nn.init.xavier_uniform_(linear2.weight)
    torch.nn.init.xavier_uniform_(linear3.weight)

    model = torch.nn.Sequential(
        linear1, sigmoid,
        linear2, sigmoid,
        linear3
    ).cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


elif Question == 5 or Question == 6:
    print(Question)
    linear1 = torch.nn.Linear(784, 128, bias=True)
    linear2 = torch.nn.Linear(128, 2, bias=True)
    linear3 = torch.nn.Linear(2, 10, bias=True)

    torch.nn.init.xavier_uniform_(linear1.weight)
    torch.nn.init.xavier_uniform_(linear2.weight)
    torch.nn.init.xavier_uniform_(linear3.weight)

    model = torch.nn.Sequential(
        linear1, sigmoid,
        linear2, sigmoid,
        linear3
    ).cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    



#Train
print("Start Train Model")
total_batch = len(data_loader)
model.train()    # set the model to train mode (dropout=True)

cost_log = []
acc_log = []

X_test = mnist_test.test_data.view(-1, 28 * 28).float().to(device)
Y_test = mnist_test.test_labels.to(device)

for epoch in range(training_epochs):
    avg_cost = 0

    for X, Y in data_loader:
        X = X.view(-1, 28 * 28).to(device)
        Y = Y.to(device)

        optimizer.zero_grad()
        hypothesis = model(X)
        cost = error(hypothesis, Y)
        cost.backward()
        optimizer.step()
        avg_cost += cost / total_batch

        with torch.no_grad():
            model.eval()
            pred = model(X)
            correct = torch.argmax(pred, 1) == Y
            accuracy = correct.float().mean().cpu()

    print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.9f}'.format(avg_cost), 'Accuracy = ' , '{:.5f}'.format(accuracy))
    cost_log.append(float(avg_cost.cpu()))
    acc_log.append(accuracy)
    change = abs(numpy.mean(acc_log[-6:-1]) - numpy.float(accuracy))
    print(change)
    if change < 0.0001:
        print("Early Stop")
        break

    with torch.no_grad():
        model.eval()    # set the model to evaluation mode (dropout=False)

        # Test the model using test sets
        prediction = model(X_test)
        correct_prediction = torch.argmax(prediction, 1) == Y_test
        accuracy = correct_prediction.float().mean()
        print('Test Data Accuracy:', accuracy.item())


print('Learning finished')


with torch.no_grad():
    model.eval()    # set the model to evaluation mode (dropout=False)

    # Test the model using test sets
    prediction = model(X_test)
    correct_prediction = torch.argmax(prediction, 1) == Y_test
    accuracy = correct_prediction.float().mean()
    print('Test Data Accuracy:', accuracy.item())

plt.plot(cost_log)
plt.title(f"Loss of Question {Question} with {learning_rate}lr")
plt.savefig(f"/home/mglee/VSCODE/git_folder/DataMiningHW4/plot/{Question}_{learning_rate}_log.png")


