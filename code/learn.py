import argparse
import numpy
import torch
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import random
import matplotlib.pyplot as plt
from yaml import parse
import pandas as pd

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


#Set parameter by argument parser
parser = argparse.ArgumentParser()
parser.add_argument('--Question', type = int)
parser.add_argument('--lr', type = float, default = 0.001)

args = parser.parse_args()

Question = args.Question
training_epochs = 100
batch_size = 32
learning_rate  = args.lr

#Set loss function
error = torch.nn.CrossEntropyLoss().to(device)

#Set data_loader
data_loader = torch.utils.data.DataLoader(dataset=mnist_train,
                                          batch_size=batch_size,
                                          shuffle=True,
                                          drop_last=True)
 

#Bulid differente model by Question number
if Question == 1:
    
    class mnist_model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear1 = torch.nn.Linear(784, 10, bias=True).to(device)
            #nn.CrossEntropyLoss include softmax function.
            #self.sigmoid = torch.nn.Sigmoid().to(device)
            #initialize weight
            torch.nn.init.xavier_uniform_(self.linear1.weight)
  
        #set forward function
        def forward(self, x, flag = False):
            x = self.linear1(x)
            return x

    #create model by given set. Also define optimizer
    model = mnist_model() 
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

elif Question == 2:
    
    class mnist_model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear1 = torch.nn.Linear(784, 128, bias=True).to(device)
            self.linear2 = torch.nn.Linear(128, 10, bias=True).to(device)
            self.sigmoid = torch.nn.Sigmoid().to(device)
            torch.nn.init.xavier_uniform_(self.linear1.weight)
            torch.nn.init.xavier_uniform_(self.linear2.weight)
            
        def forward(self, x, flag = False):
            #Set multi(2)layer nn 
            x = self.linear1(x)
            x = self.sigmoid(x)
            x = self.linear2(x)
            return x

    model = mnist_model() 
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


#Question 3 and 4 use same model
elif Question == 3 or Question == 4:
    
    class mnist_model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear1 = torch.nn.Linear(784, 128, bias=True).to(device)
            self.linear2 = torch.nn.Linear(128, 4, bias=True).to(device)
            self.linear3 = torch.nn.Linear(4, 10, bias=True).to(device)    
            self.sigmoid = torch.nn.Sigmoid().to(device)
            torch.nn.init.xavier_uniform_(self.linear1.weight)
            torch.nn.init.xavier_uniform_(self.linear2.weight)
            torch.nn.init.xavier_uniform_(self.linear3.weight)
            
            
        def forward(self, x, flag = False):
            x = self.linear1(x)
            x = self.sigmoid(x)
            x = self.linear2(x)
            hidden_out = self.sigmoid(x)
            x = self.linear3(hidden_out)
            #If Question == 4 -> return hidden layer's output with forward result
            if flag:
                return x, hidden_out
            return x

    model = mnist_model() 
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


elif Question == 5 or Question == 6:
    
    class mnist_model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear1 = torch.nn.Linear(784, 128, bias=True).to(device)
            self.linear2 = torch.nn.Linear(128, 2, bias=True).to(device)
            self.linear3 = torch.nn.Linear(2, 10, bias=True).to(device)    
            self.sigmoid = torch.nn.Sigmoid().to(device)
            torch.nn.init.xavier_uniform_(self.linear1.weight)
            torch.nn.init.xavier_uniform_(self.linear2.weight)
            torch.nn.init.xavier_uniform_(self.linear3.weight)
            
        
        def forward(self, x, flag = False):
            x = self.linear1(x)
            x = self.sigmoid(x)
            x = self.linear2(x)
            hidden_out = self.sigmoid(x)
            x = self.linear3(hidden_out)
            if flag:
                return x, hidden_out
            return x

    model = mnist_model() 
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)



#========================================================#
#Train part
#========================================================#
#========================================================#
print("Start Train Model")

# # of iteration
total_batch = len(data_loader)
model.train()    # set the model to train mode (dropout=True)

#empty list for cost and accuracy
cost_log = []
acc_log = []
test_acc_log = []

X_train = mnist_train.train_data.view(-1, 28 * 28).float().to(device)
Y_train = mnist_train.train_labels.to(device)

X_test = mnist_test.test_data.view(-1, 28 * 28).float().to(device)
Y_test = mnist_test.test_labels.to(device)


#Train start
for epoch in range(training_epochs):
    avg_cost = 0

    for X, Y in data_loader:
        # for each mini batch
        X = X.view(-1, 28 * 28).to(device)
        Y = Y.to(device)

        #reset gradient
        optimizer.zero_grad()
        #get result of forward
        hypothesis = model(X)
        #get error by cross entropy
        cost = error(hypothesis, Y)
        #backward and renew weights
        cost.backward()
        optimizer.step()
        #cumsum cost of this mini-batch
        avg_cost += cost / total_batch

    #for train data, get accuracy with current model
    with torch.no_grad():
        model.eval()
        pred = model(X_train)
        correct = torch.argmax(pred, 1) == Y_train
        accuracy = correct.float().mean().cpu()

    print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.9f}'.format(avg_cost), 'Accuracy = ' , '{:.5f}'.format(accuracy))
    cost_log.append(float(avg_cost.cpu()))
    acc_log.append(accuracy)
    
    # compare with five recent accuracy, There is no meaningful change -> early stop
    if abs(numpy.mean(acc_log[-6:-1]) - numpy.float(accuracy)) < 0.00001:
        print("Early Stop")
        break
    #for test data, get accuracy with current model
    with torch.no_grad():
        # set the model to evaluation mode
        model.eval()

        test_pred = model(X_test)
        test_correct = torch.argmax(test_pred, 1) == Y_test
        test_accuracy = test_correct.float().mean()
        test_acc_log.append(test_accuracy)
    
    
print('Learning finished')


#Test with final model.
with torch.no_grad():
    model.eval()
    # If question number is 4 or 6,
    # It needs output of hidden layer
    if Question == 4 or Question == 6:
        #If flag == True, It return hidden output
        prediction, hidden_out = model(X_test, flag = True)
    else:
        prediction = model(X_test)
    correct_prediction = torch.argmax(prediction, 1) == Y_test
    accuracy = correct_prediction.float().mean()
    print('Test Data Accuracy:', accuracy.item())


#Save plot of changing accuracy
# plt.plot(acc_log, label = 'train accuracy')
# plt.plot(test_acc_log, label = 'test accuracy')
# plt.xlabel('Epoch')
# plt.ylabel('Accuracy')
# plt.ylim(0.3, 1.1)
# plt.legend()
# plt.title(f"Accuracy of Question {Question} with {learning_rate}lr")
# plt.savefig(f"/home/mglee/VSCODE/git_folder/DataMiningHW4/plot/{Question}_{learning_rate}_log.png")
# print(f"Save figure as {Question}_{learning_rate}_log.png")

#Save plot of changing loss
plt.plot(cost_log, label = 'Loss ')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title(f"Loss of Question {Question} with {learning_rate}lr")
plt.savefig(f"/home/mglee/VSCODE/git_folder/DataMiningHW4/plot/{Question}_{learning_rate}_loss.png")
print(f"Save figure as {Question}_{learning_rate}_loss.png")

#optional task
if Question == 4:
    df = pd.DataFrame(hidden_out.cpu())
    df['label'] = Y_test.cpu()
    print(f" average of hidden layer by class : \
        {df.groupby('label').mean()}")

if Question == 6:
    df = pd.DataFrame(hidden_out.cpu())
    df.columns = ['X1', 'X2']
    df['label'] = Y_test.cpu()
    fig, ax = plt.subplots()
    for name, group in df.groupby('label'):
        ax.plot(group.X1, group.X2, marker='o', linestyle='', label=name)
    plt.savefig(f"/home/mglee/VSCODE/git_folder/DataMiningHW4/plot/{Question}_hidden_output.png")
    print(f"Save figure as {Question}_hidden_output.png")