import torch
from sys import exit
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
class transition(nn.Module):
    def __init__(self,tl_feature_map,tl_output_feature_map):
        super(transition, self).__init__()
        self.cnn1 = torch.nn.Conv2d(tl_feature_map,tl_output_feature_map,kernel_size=(1,1),stride=(1,1))
        self.ml1 = torch.nn.AvgPool2d(kernel_size=2,stride=2)
        
    def forward(self,x):
        out = self.cnn1(x)
        out = self.ml1(out)
        return out 

class Bottleneck(nn.Module):
    def __init__(self,feature_map,growth_rate):
        super(Bottleneck, self).__init__()
        self.db1 = nn.Sequential(
            nn.BatchNorm2d(feature_map,affine=False),
            nn.ReLU(),
            nn.Conv2d(feature_map,growth_rate,kernel_size=1),
            nn.BatchNorm2d(growth_rate,affine=False),
            nn.ReLU(),
            nn.Conv2d(growth_rate,growth_rate,kernel_size=3,stride=1,padding=1),
        )
    def forward(self,x):
        out = self.db1(x)
        return out


class Denseblock(nn.Module):
    def __init__(self,feature_map,growth_rate):
        super(Denseblock, self).__init__()
        self.bn1 = Bottleneck(growth_rate,growth_rate)
        self.bn2 = Bottleneck(2*growth_rate,growth_rate)
        self.bn3 = Bottleneck(3*growth_rate,growth_rate)
        self.bn4 = Bottleneck(4*growth_rate,growth_rate)
        self.bn5 = Bottleneck(5*growth_rate,growth_rate)
        self.bn6 = Bottleneck(6*growth_rate,growth_rate)
    def forward(self,x):
        out1 = self.bn1(x)
        dense1 = torch.cat([x,out1], 1)
        out2 = self.bn2(dense1)
        dense2 = torch.cat([x,out1,out2], 1)
        out3 = self.bn3(dense2)
        dense3 = torch.cat([x,out1,out2,out3], 1)
        out4 = self.bn4(dense3)
        dense4 = torch.cat([x,out1,out2,out3,out4], 1)
        out5 = self.bn5(dense4)
        dense5 = torch.cat([x,out1,out2,out3,out4,out5], 1)
        out6 = self.bn6(dense5)
        return out6

class Densenet(nn.Module):
    def __init__(self,input_size,growth_rate,do):
        super(Densenet, self).__init__()
        self.embedding = nn.Embedding(97, 16)
        self.fc1 = nn.Conv2d(input_size,growth_rate,kernel_size=(4,1),stride=(2,1),padding=(1,0))
        self.ml1 = torch.nn.MaxPool2d(kernel_size=(2,1),stride=(2,1))
        self.denseblock1 = Denseblock(growth_rate,growth_rate)
        self.tl1 = transition(growth_rate,growth_rate)
        self.denseblock2 = Denseblock(growth_rate,growth_rate)
        self.tl2 = transition(growth_rate,growth_rate)
        self.denseblock3 = Denseblock(growth_rate,growth_rate)
        self.tl3 = transition(growth_rate,growth_rate)
        self.denseblock4 = Denseblock(growth_rate,growth_rate)
        self.tl4 = transition(growth_rate,growth_rate)
        self.al2 = torch.nn.AvgPool2d(kernel_size=(16,1),stride=(16,1))
        self.fnn = nn.Linear(growth_rate,10)
        self.softmax = nn.Softmax()
        self.do = nn.Dropout(p=do)
    def forward(self,x):
        out = self.embedding(x)
        #print(out.shape)
        out = self.fc1(out)
        #print(out.shape)
        out = self.ml1(out)
        #print(out.shape)
        out = self.do(self.tl1(self.denseblock1(out)))
        #print(out.shape)
        out = self.do(self.tl2(self.denseblock2(out)))
        #print(out.shape)
        out = self.do(self.tl3(self.denseblock3(out)))
        #print(out.shape)
        out = self.do(self.tl4(self.denseblock4(out)))
        #print(out.shape)
        out = self.al2(out)
        #print(out.shape)
        out = torch.reshape(out,(batch_size,growth_rate))
        #print(out.shape)
        out = self.fnn(out)
        #print(out[0])
        out = self.softmax(out)
        #print(out[0])
        return out

###################################################Creating model object########################################
filename = "densenet_gr_5_do_0_lr_1"
growth_rate=5
do = 0
lr = 0.1
model = Densenet(1,growth_rate,do)
if torch.cuda.is_available():
    model.cuda()
###################################################Defining optimizer and loss function########################################

num_epochs =50

loss_function = nn.CrossEntropyLoss()

optimizer = torch.optim.SGD(model.parameters(), lr=lr)
print(sum(p.numel() for p in model.parameters()))
exit()

#print('Number of model parameters: {}'.format(
#        len([p.data.nelement() for p in model.parameters()])))
#print(model)
###################################################Training the data########################################

for ep in range(num_epochs):
    epoch_file = open(filename,'a')
    epoch_file.write('Epoch {}/{}   ---   \n'.format(ep+1, num_epochs))
    epoch_file.close()
    for i,(text,label) in enumerate(train_loader): 
        if torch.cuda.is_available():
            text = torch.reshape(text,(batch_size,1,1024))
            text = text.long()
            text = Variable(text.cuda())
            label = Variable(label.cuda())
        optimizer.zero_grad()
        predict = model.forward(text)
        temp = 0
        loss = loss_function(predict,label)
        loss.backward()
        optimizer.step()
     
    k = 0.0 
    for i,(text,label) in enumerate(test_loader):
        if torch.cuda.is_available():
            text = torch.reshape(text,(batch_size,1,1024))
            text = text.long()
            text = Variable(text.cuda())
            label = Variable(label.cuda())
        predict = model.forward(text)
        yhat = torch.argmax(predict, dim=1)
        k += (yhat.eq(label)).sum().item()

    test_acc = k/float(len(test_label_tensor))
    epoch_file = open(filename,'a')
    epoch_file.write('testing acc: {}\n'.format(test_acc))
    if ep %10 == 0:
        torch.save({'epoch':ep,'model_state_dict':model.state_dict(),'optimizer_state_dict':optimizer.state_dict()},filename+'%d.pth'%(ep))
