import torch
import torch.nn as nn

class Action_Conditioned_FF(nn.Module):
    def __init__(self,input_size=6,hidden_size=200,output_size=1):
        super(Action_Conditioned_FF,self).__init__()
        self.l1=torch.nn.Linear(input_size,hidden_size)
#         self.nonlinear_activation=nn.Sigmoid()
        self.l2=nn.Linear(hidden_size,1)
        self.relu=nn.ReLU()

    def forward(self, input1):
        output=self.l1(input1)
        output=self.relu(output)
        output=self.l2(output)
        return output


    def evaluate(self, model, test_loader, loss_function):
        loss=0
        counter1=len(test_loader)
        for idx,sample in enumerate(test_loader):
            loss+=loss_function(model.forward(sample['input']),sample['label']).item()
        return loss/counter1

def main():
    model = Action_Conditioned_FF()
    batch_size = 1
    data_loaders = Data_Loaders(batch_size)
    model.evaluate(model,data_loaders.test_loader,nn.MSELoss())
    
    
if __name__ == '__main__':
    main()