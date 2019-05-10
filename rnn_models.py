
import torch.nn as nn
import torch
from attention import MultiHeadAttention
from layer_conn_attention import LayerConnAttention
from BlockLSTM import BlockLSTM
from BlockGRU import BlockGRU
import random

class RNNModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, rnn_type, ntoken, ninp, nhid, nlayers, dropout=0.5, tie_weights=False, use_cudnn_version=True,
                 use_adaptive_softmax=False, cutoffs=None, discrete_input=True, num_blocks=6, n_out=None, update_topk=3, skip_inp=False, skip_out=False, use_gru=True): 
        super(RNNModel, self).__init__()
        print('using blocks model!')
        self.use_gru = use_gru
        self.skip_inp = skip_inp
        self.skip_out = skip_out
        self.update_topk = update_topk
        self.use_cudnn_version = use_cudnn_version
        self.drop = nn.Dropout(dropout)
        if n_out == None:
            n_out = ntoken
        print('number of inputs, ninp', ninp)
        if discrete_input:
            self.encoder = nn.Embedding(ntoken, ninp)
        else:
            self.encoder = nn.Linear(ntoken, ninp)
        self.num_blocks = num_blocks
        self.nhid = nhid
        self.block_size = nhid // self.num_blocks
        print('number of blocks', self.num_blocks)
        self.discrete_input = discrete_input

        self.sigmoid = nn.Sigmoid()
        if False:
            raise Exception('not defined')
            if rnn_type in ['LSTM', 'GRU']:
                self.rnn = getattr(nn, rnn_type)(ninp, nhid, nlayers, dropout=dropout)
            else:
                try:
                    nonlinearity = {'RNN_TANH': 'tanh', 'RNN_RELU': 'relu'}[rnn_type]
                except KeyError:
                    raise ValueError("""An invalid option for `--model` was supplied,
                                     options are ['LSTM', 'GRU', 'RNN_TANH' or 'RNN_RELU']""")
                self.rnn = nn.RNN(ninp, nhid, nlayers, nonlinearity=nonlinearity, dropout=dropout)
        else:
            #tried reducing size
            self.mha = MultiHeadAttention(n_head=4, d_model_read=self.block_size, d_model_write=self.block_size, d_model_out=self.block_size, d_k=16, d_v=16, num_blocks_read=num_blocks, num_blocks_write=num_blocks, topk=num_blocks, grad_sparse=False)


            self.inp_att = MultiHeadAttention(n_head=1, d_model_read=self.block_size, d_model_write=self.nhid, d_model_out=self.block_size, d_k=128, d_v=128, num_blocks_read=num_blocks, num_blocks_write=2,residual=False, topk=num_blocks, grad_sparse=False)

            if rnn_type in ['LSTM', 'GRU']:
                rnn_type = str(rnn_type) + 'Cell'
                rnn_modulelist = []
                blockrnn_lst = []
                dropout_lst = []
                for i in range(nlayers):
                    blocklst = []
                    for block in range(num_blocks):
                        if i == 0:
                            if use_gru:
                                self.block_lstm = BlockGRU(self.block_size*num_blocks, nhid, k=num_blocks)
                            else:
                                self.block_lstm = BlockLSTM(self.block_size*num_blocks, nhid, k=num_blocks)
                            blocklst.append(getattr(nn, rnn_type)(ninp, nhid//num_blocks))
                        else:
                            blocklst.append(getattr(nn, rnn_type)(self.block_size, nhid//num_blocks))
                    blocklst = nn.ModuleList(blocklst)
                    rnn_modulelist.append(blocklst)
                    dropout_lst.append(nn.Dropout(dropout))

                print('number of layers', nlayers)
                print('number of modules', len(rnn_modulelist))
                self.rnn = nn.ModuleList(rnn_modulelist)
                self.dropout_lst = nn.ModuleList(dropout_lst)
            else:
                raise ValueError("non-cudnn version of (RNNCell) is not implemented. use LSTM or GRU instead")

        if not use_adaptive_softmax:
            self.use_adaptive_softmax = use_adaptive_softmax
            self.decoder = nn.Linear(nhid, n_out)
            # Optionally tie weights as in:
            # "Using the Output Embedding to Improve Language Models" (Press & Wolf 2016)
            # https://arxiv.org/abs/1608.05859
            # and
            # "Tying Word Vectors and Word Classifiers: A Loss Framework for Language Modeling" (Inan et al. 2016)
            # https://arxiv.org/abs/1611.01462
            if tie_weights:
                print('tying weights!')
                if nhid != ninp:
                    raise ValueError('When using the tied flag, nhid must be equal to emsize')
                self.decoder.weight = self.encoder.weight
        else:
            # simple linear layer of nhid output size. used for adaptive softmax after
            # directly applying softmax at the hidden states is a bad idea
            self.decoder_adaptive = nn.Linear(nhid, nhid)
            self.use_adaptive_softmax = use_adaptive_softmax
            self.cutoffs = cutoffs
            if tie_weights:
                print("Warning: if using adaptive softmax, tie_weights cannot be applied. Ignored.")

        self.init_weights()

        self.rnn_type = rnn_type
        self.nhid = nhid
        self.nlayers = nlayers

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        if not self.use_adaptive_softmax:
            self.decoder.bias.data.zero_()
            self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, input, hidden):

        extra_loss = 0.0

        if self.skip_inp:
            emb = input
        else:
            emb = self.drop(self.encoder(input))
        
        if False:
            output, hidden = self.rnn(emb, hidden)
        else:
            # for loop implementation with RNNCell
            layer_input = emb
            new_hidden = [[], []]
            for idx_layer in range(0, self.nlayers):
                #print('idx layer', idx_layer)
                output = []
                #self.block_lstm.blockify_params()
                #print('hidden shape', hidden[0].shape)
                if self.use_gru:
                    hx = hidden[idx_layer]
                else:
                    hx, cx = hidden[0][idx_layer], hidden[1][idx_layer]
                print_rand = random.uniform(0,1)
                for idx_step in range(input.shape[0]):
                    hxl = []
                    if not self.use_gru:
                        cxl = []

                    if idx_layer == 0:
                        inp_use = layer_input[idx_step]
                        
                        #inp_use = inp_use.repeat(1,self.num_blocks)
                        
                        #use attention here.  
                        #print('inp use shape', inp_use.shape, 'self.nhid', self.nhid)
                        #print('hx shape', hx.shape)
                        inp_use = inp_use.reshape((inp_use.shape[0], 1, self.nhid))
                        inp_use = torch.cat([inp_use, torch.zeros_like(inp_use)], dim=1)
                        inp_use, iatt, _ = self.inp_att(hx.reshape((hx.shape[0], self.num_blocks, self.block_size)), inp_use, inp_use)
                        inp_use = inp_use.reshape((inp_use.shape[0], self.block_size*self.num_blocks))

                        null_score = iatt.mean((0,1))[1]

                        topkval = self.update_topk
                        if False and print_rand < 0.01:
                            print('inp attention on step', input.shape[0], '(total steps)', idx_step, iatt[0])
                            print('iat shape', iatt.shape)
                            #print('mask shape', mask.shape)
                            #print('mask at 0', mask[0])
                            print('iat summed', iatt.mean((0,1)))
                            print('iat null_score', null_score)


                        topk_mat = torch.topk(iatt[:,:,0], dim=1, k=topkval)[0][:,-1] #64 x 1
                        topk_mat = topk_mat.reshape((inp_use.shape[0],1)).repeat(1,self.num_blocks) #64 x num_blocks
                        mask = torch.gt(iatt[:,:,0], topk_mat - 0.01).float()
                        if False and print_rand < 0.001:
                            print('step', idx_step, 'out of', input.shape[0])
                            print('att at 0', iatt[0])
                            print('mask at 0', mask[0])
                        mask = mask.reshape((inp_use.shape[0],self.num_blocks,1)).repeat((1,1,self.block_size)).reshape((inp_use.shape[0], self.num_blocks*self.block_size))
                        
                        mask = mask.detach()

                        #print('inp use shape', inp_use.shape)
                    else:
                        #inp_use = layer_input[idx_step]#[:,block*self.block_size : (block+1)*self.block_size]
                        inp_use = layer_input[idx_step]
                        inp_use = inp_use.reshape((inp_use.shape[0], self.num_blocks, self.block_size))
                        inp_use,_,_ = self.layer_conn_att(hx.reshape((hx.shape[0], self.num_blocks, self.block_size)), inp_use, inp_use)
                        #inp_use,_,_ = self.layer_conn_att(inp_use, inp_use, inp_use)
                        #print('inp use shape', inp_use.shape)
                        inp_use = inp_use.reshape((inp_use.shape[0], self.nhid*self.num_blocks*self.fan_in))

                    #print('layer, inp shape', idx_layer, inp_use.shape)

                    if self.use_gru:
                        hx_new = self.block_lstm(inp_use, hx)
                    else:
                        hx_new, cx_new = self.block_lstm(inp_use, hx, cx)

                    #print('null score', null_score, 'above threshold on step', idx_step, 'out of', input.shape[0])

                    if True:
                        hx_old = hx*1.0
                        if not self.use_gru:
                            cx_old = cx*1.0

                        hx_new = hx_new.reshape((hx_new.shape[0], self.num_blocks, self.block_size))
                        hx_new,attn_out,extra_loss_att = self.mha(hx_new,hx_new,hx_new)
                        hx_new = hx_new.reshape((hx_new.shape[0], self.nhid))
                        extra_loss += extra_loss_att

                        #hx = hx_new
                        #cx = cx_new

                        hx = (mask)*hx_new + (1-mask)*hx_old
                        if not self.use_gru:
                            cx = (mask)*cx_new + (1-mask)*cx_old

                        #print('hx min max', hx.min(), hx.max(), 'cx min max', cx.min(), cx.max())

                    output.append(hx)
                output = torch.stack(output)
                if idx_layer + 1 < self.nlayers:
                    output = self.dropout_lst[idx_layer](output)
                layer_input = output
                new_hidden[0].append(hx)
                if not self.use_gru:
                    new_hidden[1].append(cx)
            new_hidden[0] = torch.stack(new_hidden[0])
            if not self.use_gru:
                new_hidden[1] = torch.stack(new_hidden[1])
            if self.use_gru:
                hidden = new_hidden[0]
            else:
                hidden = tuple(new_hidden)

        output = self.drop(output)

        if self.skip_out:
            return output, hidden, extra_loss

        if not self.use_adaptive_softmax:
            #print('not use adaptive softmax, size going into decoder', output.size())
            decoded = self.decoder(output.view(output.size(0) * output.size(1), output.size(2)))
            return decoded.view(output.size(0), output.size(1), decoded.size(1)), hidden, extra_loss
        else:
            decoded = self.decoder_adaptive(output.view(output.size(0) * output.size(1), output.size(2)))
            return decoded.view(output.size(0), output.size(1), decoded.size(1)), hidden, extra_loss

    def init_hidden(self, bsz):
        weight = next(self.block_lstm.parameters())
        if self.rnn_type == 'LSTM' or self.rnn_type == 'LSTMCell':
            return (weight.new_zeros(self.nlayers, bsz, self.nhid),
                    weight.new_zeros(self.nlayers, bsz, self.nhid))
        else:
            return weight.new_zeros(self.nlayers, bsz, self.nhid)

