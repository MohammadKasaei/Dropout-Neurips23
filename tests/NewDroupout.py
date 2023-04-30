import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.init as init

import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import norm
import statistics
import math

device = torch.device('cuda:' + str(0) if torch.cuda.is_available() else 'cpu')


class CtrlbDropout(nn.Module):
    def __init__(self,p=0.1,active = True):
        super(CtrlbDropout, self).__init__()       
        self._p = p
        self._active = active
        self._iter = 1
        self.drops = 0

    def _tensor_to_output(self,tensor):
     
    #   output_tensor = torch.zeros(tensor.shape).to(device)
    #   output_tensor[tensor < self._p] = 1
    
    #   output_tensor = torch.ones(tensor.shape).to(device)
    #   _, idx = torch.topk(tensor, int(self._p*tensor.shape[1])+1,dim = 1, sorted=False)
    #   output_tensor = output_tensor.scatter(dim=1, index=idx, value=0)

      return torch.bernoulli(1-tensor)
    
    
    def _assembleCtrlb(self,x):
      g = x**2
      orderedS = torch.abs(g)**0.5
      
     




      



      

    #   max_vals, idx = torch.topk(orderedS, math.ceil(self._p*orderedS.shape[1]),dim = 1, sorted=False)
    #   m = torch.mean(max_vals,dim=1).unsqueeze(1)
    #   prob = torch.clamp(orderedS/(m),0,1)

      prob = (orderedS/(torch.max(orderedS,dim=1)[0].unsqueeze(1)))
      top_half, top_idx = torch.topk(prob, math.floor(0.1*prob.shape[1]),dim = 1, largest=True, sorted=False)
      btm_half, btm_idx = torch.topk(prob, math.floor(0.1*prob.shape[1]),dim = 1, largest=False,sorted=False)
      scalling = prob.gather(1,top_idx) - (prob.gather(1,top_idx) - prob.gather(1,btm_idx))


      prob = prob.scatter(1, top_idx, scalling)

      prob = torch.clamp(prob,0,1)

      
      
      # g = torch.diag_embed(x).to(device)      
      # r = torch.svd(g,compute_uv=True)
      # us =  torch.bmm(r.U,torch.diag_embed(r.S))
      # norm = torch.norm(us,dim=1).unsqueeze(1)
      # orderedS = torch.bmm(norm,r.V.mT).squeeze()

      # prob = torch.softmax(r.S,1)

      # su  = torch.diagonal((torch.diag_embed(r.S)*r.U),dim1=1, dim2=2)
      # su = r.U*r.S
      # prob = torch.softmax(orderedS,1)
      # prob = torch.exp(-orderedS/torch.max(orderedS))
    #   prob = torch.softmax(-orderedS/torch.max(orderedS),1)




      #  u, s, v  = svd(x)
      #  prob = torch.softmax(s,1)
      return self._tensor_to_output(prob)

    def forward(self,x):
      if not self.training:
          return x
      
      with torch.no_grad():
         drop = self._assembleCtrlb(x)
         self.drops = (self.drops + (torch.mean((torch.sum(drop, dim=1)/drop.shape[1])))) / 2.0
        #  print (self.drops,"\t",  torch.mean((torch.sum(drop, dim=1)/drop.shape[1])))
         
        #  self.drops /= self._iter
         self._iter += 1.0
      return x*drop




class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__() 
        
        def _kaiming_init(model):
            for name, param in model.named_parameters():
                if 'weight' in name:
                    init.kaiming_normal_(param.data, mode='fan_in', nonlinearity='relu')
                elif 'bias' in name:
                    param.data.fill_(0.0)
          
        # self.nn = nn.Sequential (nn.Linear(28*28,100),                                                                 
        #                          nn.ReLU(),
        #                          nn.Dropout(p=0.2),                                 
        #                          nn.Linear(100,100),
        #                          nn.ReLU(),
        #                          nn.Dropout(p=0.1),
        #                          nn.Linear(100,10),
        #                         #  nn.Dropout(p=0.2)
        #                    )

        # self.nn = nn.Sequential (nn.Linear(28*28,300),                                                                 
        #                          nn.ReLU(),
        #                          CtrlbDropout(0.005),                                 
        #                          nn.Linear(300,100),
        #                          nn.ReLU(),
        #                          CtrlbDropout(0.02),                                 
        #                          nn.Linear(100,10),
        #                          CtrlbDropout(0.1)                          
        #                         )
        # self.nn = nn.Sequential (nn.Linear(28*28,300),                                                                 
        #                          nn.ReLU(),
        #                          CtrlbDropout(0.003),                                 
        #                          nn.Linear(300,100),
        #                          nn.ReLU(),
        #                          CtrlbDropout(0.01),                                 
        #                          nn.Linear(100,10),
        #                          CtrlbDropout(0.1)                          
        #                         )        
        self.nn = nn.Sequential (nn.Linear(28*28,100),                                                                 
                            nn.ReLU(),
                            # CtrlbDropout(0.2),                                  
                            nn.Linear(100,100),
                            nn.ReLU(),
                            # CtrlbDropout(0.2),                                 
                            nn.Linear(100,10),
                            # CtrlbDropout(0.2)                          
                        )     

        _kaiming_init(self.nn)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x += torch.randn_like(x)        
        x = self.nn(x)
        return F.log_softmax(x)
    

def train(epoch):
  network.train()
  for batch_idx, (data, target) in enumerate(train_loader):
    optimizer.zero_grad()
    output = network(data.to(device))
    loss = F.nll_loss(output, target.to(device))
    loss.backward()
    optimizer.step()
    if batch_idx % log_interval == 0:
      print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        epoch, batch_idx * len(data), len(train_loader.dataset),
        100. * batch_idx / len(train_loader), loss.item()))
      train_losses.append(loss.item())
      train_counter.append(
        (batch_idx*64) + ((epoch-1)*len(train_loader.dataset)))
      get_drops(network.nn)
  # torch.save(network.state_dict(), '/results/model_higher_dp_value.pth')
  # torch.save(optimizer.state_dict(), '/results/optimizer.pth')



def test():
  network.eval()
  test_loss = 0
  correct = 0
  with torch.no_grad():
    for data, target in test_loader:
      output = network(data.to(device))
      target = target.to(device)
      test_loss += F.nll_loss(output, target, size_average=False).item()
      pred = output.data.max(1, keepdim=True)[1]
      correct += pred.eq(target.data.view_as(pred)).sum()
  test_loss /= len(test_loader.dataset)
  test_losses.append(test_loss)
  print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.000f}%)\n'.format(
    test_loss, correct, len(test_loader.dataset),
    100. * correct / len(test_loader.dataset)))


def get_drops(net):
    for layer in net:        
        if hasattr(layer,'drops'):
           print(f"drops: {layer.drops}")
        # activations.append(x)
    # return activations


def get_activations(net, x):
    activations = []
    x = torch.flatten(x, 1)
    for layer in net:
        x = layer(x)
        activations.append(x)
    return activations


def singular_values(act):
   g = torch.diag_embed(act).to(device)      
   r = torch.svd(g,compute_uv=True)
  #  us =  torch.bmm(r.U,torch.diag_embed(r.S))
   return r.S.cpu().detach().numpy()


if __name__ == "__main__":
    n_epochs = 100
    batch_size_train = 128
    batch_size_test = 1000
    learning_rate = 0.001
    log_interval = 100
    random_seed = 25

    training = True
    saveModel = True
    modelPath = "models/model2.pth"


    # torch.backends.cudnn.enabled = False
    torch.manual_seed(random_seed)

 

    test_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('data/', train=False, download=True,
                                transform=torchvision.transforms.Compose([
                                torchvision.transforms.ToTensor(),
                                torchvision.transforms.Normalize(
                                    (0.1307,), (0.3081,))
                                ])),
    batch_size=batch_size_test, shuffle=True)

    if training:

        network = Net().to(device)
        optimizer = optim.Adam(network.parameters(), lr=learning_rate)


        train_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST('data/', train=True, download=True,
                                    transform=torchvision.transforms.Compose([
                                    torchvision.transforms.ToTensor(),
                                    torchvision.transforms.Normalize(
                                        (0.1307,), (0.3081,))
                                    ])),
        batch_size=batch_size_train, shuffle=True)

        train_losses = []
        train_counter = []
        test_losses = []
        test_counter = [i*len(train_loader.dataset) for i in range(n_epochs + 1)]

        for epoch in range(1, n_epochs + 1):
            train(epoch)
            
            test()
            
        
        if saveModel:
          torch.save(network.state_dict(), modelPath)
    else:
        vanila_nn = Net().to(device)
        network2 = Net().to(device)
        vanila_nn.eval()
        network2.eval()

        vanila_nn.load_state_dict(torch.load('models/model_vanilla.pth'))
        network2.load_state_dict(torch.load('models/model2.pth'))


        examples = enumerate(test_loader)
        batch_idx, (example_data, example_targets) = next(examples)
        fig = plt.figure()
        for i in range(6):
            plt.subplot(2,3,i+1)
            plt.tight_layout()
            plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
            plt.title("Ground Truth: {}".format(example_targets[i]))
            plt.xticks([])
            plt.yticks([])
        plt.show()


        with torch.no_grad():
            output = network2(example_data.to(device))

        fig = plt.figure()
        for i in range(6):
            plt.subplot(2,3,i+1)
            plt.tight_layout()
            plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
            plt.title("Prediction: {}".format(output.data.max(1, keepdim=True)[1][i].item()))
            plt.xticks([])
            plt.yticks([])

        plt.show()


        for k in range(5):
            batch_idx, (example_data, example_targets) = next(examples)
            with torch.no_grad():
                acts = get_activations (vanila_nn.nn,example_data[0].to(device))
                acts2 = get_activations (network2.nn,example_data[0].to(device))

                i = 0
                for act1,act2 in zip(acts,acts2):
                    i+=1
                    fig, axs = plt.subplots(2)
                    s1 = singular_values(act1)
                    s2 = singular_values(act2)
                    counts, bins = np.histogram(s2)
                    axs[0].hist(bins[:-1], bins, weights=counts,color= 'r',ec="k")                   
                    axs[0].set_ylabel('Ours')
                    axs[0].set_title(f"layer: {i}")

                    counts, bins = np.histogram(s1)
                    axs[1].hist(bins[:-1], bins, weights=counts,color= 'b',ec="k")
                    axs[1].set_ylabel('Vanila')
                    plt.show()    





