
'''
Giving an N x M attention matrix, returns the same matrix, but performs masking to determine where to block gradients.  


'''

import torch
import torch.nn as nn
import numpy

from sparse_attn import Sparse_attention
from torch.autograd import Variable, Function

class Sparse_grad_attention(torch.autograd.Function):
    def __init__(self, top_k):
        super(Sparse_grad_attention,self).__init__()

        self.sa = Sparse_attention(top_k=top_k)

    def forward(self, inp): 
        
        sparsified = self.sa(inp)
        self.save_for_backward(inp, sparsified)

        return inp

    def backward(self, grad_output):
        inp, sparsified = self.saved_tensors
        #print('sparsified', sparsified)
        return (grad_output) * (sparsified > 0.0).float()

if __name__ == "__main__":
    k = 2
    sga = Sparse_grad_attention(k)
    sa = Sparse_attention(k)

    x = torch.from_numpy(numpy.array([[[0.1, 0.0, 0.3, 0.2, 0.4],[0.5,0.4,0.1,0.0,0.0]]]))
    x = x.reshape((2,5))

    x = Variable(x, requires_grad=True)

    print(x)
    print('output', sga(x))

    (sga(x).sum()).backward()

    print('sparse grad', x.grad)

    x = Variable(x.data, requires_grad=True)

    (sa(x).sum()).backward()

    print('normal grad', x.grad)


