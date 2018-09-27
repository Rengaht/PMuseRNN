# Practical PyTorch: Generating Names with a Conditional Character-Level RNN
# https://github.com/spro/practical-pytorch

import glob
import unicodedata
import string
import random
import time
import math

import torch
import torch.nn as nn
from cdata import *
from model import *
# from generate import *



# Training the Network
def train(category_tensor, input_line_tensor, target_line_tensor):
    hidden = rnn.init_hidden()
    optimizer.zero_grad()
    loss = 0
    
    for i in range(input_line_tensor.size()[0]):
        output, hidden = rnn(category_tensor, input_line_tensor[i], hidden)
        loss += criterion(output, target_line_tensor[i].unsqueeze(0))

    loss.backward()
    optimizer.step()
    
    return output, loss.data[0] / input_line_tensor.size()[0]

def time_since(t):
    now = time.time()
    s = now - t
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)



print('#vocs= ',n_vocs)
print('#category= ',n_categories)
print(all_categories)


n_epochs = 2400
print_every = 100
plot_every = 500
all_losses = []
loss_avg = 0 # Zero every plot_every epochs to keep a running average
hidden_size = 32
learning_rate = 0.0003


if torch.cuda.is_available():
    print('use gpu...')
else:
    print('no gpu available...')

rnn = RNN(n_categories, n_vocs, hidden_size, n_vocs)
optimizer = torch.optim.Adam(rnn.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()

start = time.time()

def save():
    torch.save(rnn, 'conditional-char-rnn.pt')

try:
    print("Training for %d epochs..." % n_epochs)
    for epoch in range(1, n_epochs + 1):
        if epoch % print_every == 0:        
            print('epoch',epoch,'-',epoch+print_every)
        output, loss = train(*random_training_set())
        loss_avg += loss        

        if epoch % print_every == 0:
            print('%s (%d %d%%) %.4f' % (time_since(start), epoch, epoch / n_epochs * 100, loss))
            #if loss<2:
            #    break
            # tmp=tmpGenerate(rnn) 
            # print('test generating...')           
            # print(tmp)
        
        if epoch % plot_every == 0:
            all_losses.append(loss_avg / plot_every)
            loss_avg = 0
    save()
except KeyboardInterrupt:
    print("Saving before quit...")
    save()

