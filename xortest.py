#%matplotlib notebook
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import numpy as np
import time
from IPython.display import clear_output
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim

def draw_points(x,y,z, fig, ax1, epoch):
    ngridx = 100
    ngridy = 100
    ax1.clear()
    xi = np.linspace(-0.1, 1.1, ngridx)
    yi = np.linspace(-0.1, 1.1, ngridy)
    triang = tri.Triangulation(x, y)
    interpolator = tri.LinearTriInterpolator(triang, z)
    Xi, Yi = np.meshgrid(xi, yi)
    zi = interpolator(Xi, Yi)
    ax1.contour(xi, yi, zi, levels=14, linewidths=0.5, colors='k')
    cntr1 = ax1.contourf(xi, yi, zi, levels=14, cmap="RdBu_r")
#    fig.colorbar(cntr1, ax=ax1)
    ax1.plot(x, y, 'ko', ms=3)
    ax1.set(xlim=(0, 1), ylim=(0, 1))
    ax1.set_title('Epocha: %d' % (epoch))
#    fig.canvas.draw()
    plt.draw()
    plt.pause(0.00000001)

class XorNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2,10)
        self.fc2 = nn.Linear(10,1)
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

m = XorNet()
loss_fn = nn.MSELoss()
optimizer = optim.Adam(m.parameters(), lr=1e-3)

training_epochs = 1000
minibatch_size = 32

# input-output pairs
pairs = [(np.asarray([0.0,0.0]), [0.0]),
         (np.asarray([0.0,1.0]), [1.0]),
         (np.asarray([1.0,0.0]), [1.0]),
         (np.asarray([1.0,1.0]), [0.0])]

state_matrix = np.vstack([x[0] for x in pairs])

label_matrix = np.vstack([x[1] for x in pairs])
npts = 200
fig, ax1 = plt.subplots(1,1)
#plt.show()
for i in range(training_epochs):
        
    for batch_ind in range(4):
        # wrap the data in variables
        minibatch_state_var = Variable(torch.Tensor(state_matrix))
        minibatch_label_var = Variable(torch.Tensor(label_matrix))
                
        # forward pass
        y_pred = m(minibatch_state_var)
        
        # compute and print loss
        loss = loss_fn(y_pred, minibatch_label_var)

        # reset gradients
        optimizer.zero_grad()
        
        # backwards pass
        loss.backward()
        
        # step the optimizer - update the weights
        optimizer.step()
        
    x = np.random.uniform(0, 1, npts)
    y = np.random.uniform(0, 1, npts)
    sv = Variable(torch.Tensor([np.asarray([x[ii], y[ii]]) for ii in range(len(x))]))
    pred = m(sv)
    z = [float(pred[ii][0]) for ii in range(len(pred))]
    draw_points(x,y,z, fig, ax1, i)
