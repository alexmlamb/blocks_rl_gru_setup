


import rnn_models
import torch
import torch.nn as nn


class BlockWrapper(nn.Module):
    
    def __init__(self, ntokens, nhid, n_out, dropout=0.0, num_blocks=4, update_topk=4):
        super(BlockWrapper, self).__init__()
        self.myrnn = rnn_models.RNNModel("LSTM", ntokens, nhid, nhid,
                            nlayers=1, dropout=dropout, tie_weights=False,
                            use_cudnn_version=False, use_adaptive_softmax=False,
                            cutoffs=[10000], discrete_input=False, n_out = n_out, num_blocks=num_blocks, update_topk=update_topk, skip_inp=True, skip_out=True, use_gru=True).cuda()
        self.nhid = nhid

    def forward(self, inp, h):
        assert len(h.shape) == 3
        assert len(inp.shape) == 3
        #hx = h[:,:,:self.nhid]
        #cx = h[:,:,self.nhid:]
        #print('hx input shape', hx.shape, 'cx input shape', cx.shape)
        ob, hb, extra_loss = self.myrnn(inp, h)
        #hb = torch.cat([hb[0],hb[1]], dim=2)
        return ob,hb


if __name__ == "__main__":
    nhid = 600
    ntokens = 10

    blocks = BlocksWrapper(ntokens, nhid, n_out=nhid)
    gru = torch.nn.GRU(ntokens, nhid).cuda()

    x = torch.randn(50, 64, 10).cuda()

    h0 = torch.randn(1, 64, nhid).cuda()
    h0_blocks = torch.randn(1, 64, nhid*2).cuda()

    og, hg = gru(x, h0)
    print('gru of x: o,h', og.shape, hg.shape)

    ob, hb = blocks(x, h0_blocks)
    print('block res: o,h', ob.shape, hb.shape)



