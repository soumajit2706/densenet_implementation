import torch 
import torch.nn as nn 
import torchvision.transforms as transforms 
import torchvision.datasets as dsets 
from torch.autograd import Variable 
from torch.utils.data import Dataset,DataLoader


####################################Load the input and label tensor for training and testing data##########################
train_input_tensor = torch.load('./yahoo_answers_csv/train_input_tensor.pt')
train_label_tensor = torch.load('./yahoo_answers_csv/train_label_tensor.pt')
test_input_tensor = torch.load('./yahoo_answers_csv/test_input_tensor.pt')
test_label_tensor = torch.load('./yahoo_answers_csv/test_label_tensor.pt')


###################################create dataloader class for training and testing data###################################
batch_size =100
class train_textdata(Dataset):
    def __init__(self):
        self.len = train_input_tensor.shape[0]
        self.x_data = train_input_tensor
        self.y_data = train_label_tensor

    def __getitem__(self,index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len

dataset = train_textdata()
train_loader = DataLoader(dataset=dataset,batch_size=batch_size,shuffle=True)

class test_textdata(Dataset):
    def __init__(self):
        self.len = test_input_tensor.shape[0]
        self.x_data = test_input_tensor
        self.y_data = test_label_tensor

    def __getitem__(self,index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len

dataset = test_textdata()
test_loader = DataLoader(dataset=dataset,batch_size=batch_size,shuffle=False)

###################################################Deep learning network architecture########################################
class CNN(nn.Module):
    def __init__(self,tl_feature_map,tl_output_feature_map,kernel,stride,padding):
        super(CNN, self).__init__()
        self.cnn1 = torch.nn.Conv1d(tl_feature_map,tl_output_feature_map,kernel,stride,padding)
        self.ml1 = torch.nn.MaxPool1d(2)
        
    def forward(self,x):
        out = self.cnn1(x)
        out = self.ml1(out)
        return out 

class denseblock(nn.Module):
    def __init__(self,feature_map,growth_rate):
        super(denseblock, self).__init__()
        self.db1 = nn.Sequential(
            nn.BatchNorm1d(feature_map,affine=False),
            nn.ReLU(),
            nn.Conv1d(feature_map,growth_rate,1),
            nn.BatchNorm1d(growth_rate,affine=False),
            nn.ReLU(),
            nn.Conv1d(growth_rate,growth_rate,3,1,1),
        )
    def forward(self,x):
        out = self.db1(x)
        return out

class FNN(nn.Module):
    def __init__(self,input_size,output_dim):
        super(FNN, self).__init__()
        self.fc1 = nn.Linear(input_dim,output_dim)
        self.softmax = nn.Softmax()

    def forward(self,x):
        out = self.fc1(x)
        out = self.softmax(out)
        return out


###################################################Creating model object########################################
tl_feature_map = 1
tl_output_feature_map = 16
kernel,stride,padding = 4,2,1
tl1 = CNN(tl_feature_map,tl_output_feature_map,kernel,stride,padding)

db = []
feature_map = 16
growth_rate = 16
for i in range(6):
    db.append(denseblock(feature_map,growth_rate))
    feature_map += growth_rate

tl_feature_map = 16
tl_output_feature_map = 16
kernel,stride,padding = 4,2,1
tl2 = CNN(tl_feature_map,tl_output_feature_map,kernel,stride,padding)

input_dim = 16*64 
out_dim = 10
fn = FNN(input_dim,out_dim)
loss_function = nn.CrossEntropyLoss()

if torch.cuda.is_available():
    tl1.cuda()
    for i in range(6):
        db[i].cuda()
    tl2.cuda()
    fn.cuda()


###################################################Defining optimizer and loss function########################################

num_epochs =20

loss_function = nn.CrossEntropyLoss()

params = list(tl1.parameters()) + list(db[0].parameters()) + list(db[1].parameters()) + list(db[2].parameters()) + list(db[3].parameters()) + list(db[4].parameters()) + list(db[5].parameters()) + list(tl2.parameters()) + list(fn.parameters())

optimizer = torch.optim.SGD(params, lr=0.05)

###################################################Training the data########################################

for ep in range(num_epochs):
    for i,(text,label) in enumerate(train_loader): 
        if torch.cuda.is_available():
            text = torch.reshape(text,(batch_size,1,1024))
            text = Variable(text.cuda())
            label = Variable(label.cuda())
        optimizer.zero_grad()
        out = tl1.forward(text)
        for j in range(6):
            temp = db[j].forward(out)
            out = torch.cat((out,temp),dim=1)
        del out
        out = tl2.forward(temp)
        out = torch.reshape(out,(batch_size,16*64))
        predict = fn.forward(out)
        loss = loss_function(predict,label)
        loss.backward()
        optimizer.step()
    print(loss)
    k = 0.0
    for i,(images,label) in enumerate(test_loader):
        if torch.cuda.is_available():
            text = torch.reshape(text,(batch_size,1,1024))
            text = Variable(text.cuda())
            label = Variable(label.cuda())
        _,predicted_label = torch.max(predict,1)
        k = k + torch.eq(label, predicted_label).sum() 
        k = k.type(torch.DoubleTensor)
    print(k,torch.div(k, float(len(test_label_tensor))),loss)

