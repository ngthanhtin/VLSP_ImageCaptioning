from common import *
from configure import *
from bms import *

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, PackedSequence
#https://github.com/pytorch/pytorch/issues/1788
# https://stackoverflow.com/questions/51030782/why-do-we-pack-the-sequences-in-pytorch

from fairseq_transformer import *
from tnt_patch import *
#from timm.models.tnt import *


class Net(nn.Module):

    def __init__(self,):
        super(Net, self).__init__()
        self.cnn = TNT()
        self.image_encode = nn.Identity()

        #---
        self.text_pos    = PositionEncode1D(text_dim,max_length)
        self.token_embed = nn.Embedding(vocab_size, text_dim)
        self.text_decode = TransformerDecode(decoder_dim, ff_dim, num_head, num_layer)

        #---
        self.logit  = nn.Linear(decoder_dim, vocab_size)
        self.dropout = nn.Dropout(p=0.5)

        #----
        # initialization
        self.token_embed.weight.data.uniform_(-0.1, 0.1)
        self.logit.bias.data.fill_(0)
        self.logit.weight.data.uniform_(-0.1, 0.1)


    @torch.jit.unused
    def forward(self, patch, coord, token, patch_pad_mask, token_pad_mask):
        device = patch.device
        batch_size = len(patch)
        #---
        patch = patch*2-1
        image_embed = self.cnn(patch, coord, patch_pad_mask)
        image_embed = self.image_encode(image_embed).permute(1,0,2).contiguous()

        text_embed = self.token_embed(token)
        text_embed = self.text_pos(text_embed).permute(1,0,2).contiguous()

        max_of_length = token_pad_mask.shape[-1]
        text_mask = np.triu(np.ones((max_of_length, max_of_length)), k=1).astype(np.uint8)
        text_mask = torch.autograd.Variable(torch.from_numpy(text_mask)==1).to(device)

        #----
        # <todo> perturb mask as aug
        text_pad_mask = token_pad_mask[:,:,0]==0
        image_pad_mask = patch_pad_mask[:,:,0]==0
        x = self.text_decode(text_embed[:max_of_length], image_embed, text_mask, text_pad_mask, image_pad_mask)
        x = x.permute(1,0,2).contiguous()
        l = self.logit(x)

        logit = torch.zeros((batch_size, max_length, vocab_size),device=device)
        logit[:,:max_of_length]=l
        return logit

    #submit function has not been coded. i will leave it as an exercise for the kaggler
    #@torch.jit.export
    # def forward_argmax_decode(self, patch, coord, mask):
    #
    #     image_dim   = 384
    #     text_dim    = 384
    #     decoder_dim = 384
    #     num_layer = 3
    #     num_head  = 8
    #     ff_dim    = 1024
    #
    #     STOI = {
    #         '<sos>': 190,
    #         '<eos>': 191,
    #         '<pad>': 192,
    #     }
    #     max_length = 278 # 275
    #
    #
    #     #---------------------------------
    #     device = patch.device
    #     batch_size = len(patch)
    #
    #     patch = patch*2-1
    #     image_embed = self.cnn(patch, coord, mask)
    #     image_embed = self.image_encode(image_embed).permute(1,0,2).contiguous()
    #
    #     token = torch.full((batch_size, max_length), STOI['<pad>'],dtype=torch.long, device=device)
    #     text_pos = self.text_pos.pos
    #     token[:,0] = STOI['<sos>']
    #
    #
    #     #-------------------------------------
    #     eos = STOI['<eos>']
    #     pad = STOI['<pad>']
    #     # https://github.com/alexmt-scale/causal-transformer-decoder/blob/master/tests/test_consistency.py
    #     # slow version
    #     # if 0:
    #     #     for t in range(max_length-1):
    #     #         last_token = token [:,:(t+1)]
    #     #         text_embed = self.token_embed(last_token)
    #     #         text_embed = self.text_pos(text_embed).permute(1,0,2).contiguous() #text_embed + text_pos[:,:(t+1)] #
    #     #
    #     #         text_mask = np.triu(np.ones((t+1, t+1)), k=1).astype(np.uint8)
    #     #         text_mask = torch.autograd.Variable(torch.from_numpy(text_mask)==1).to(device)
    #     #
    #     #         x = self.text_decode(text_embed, image_embed, text_mask)
    #     #         x = x.permute(1,0,2).contiguous()
    #     #
    #     #         l = self.logit(x[:,-1])
    #     #         k = torch.argmax(l, -1)  # predict max
    #     #         token[:, t+1] = k
    #     #         if ((k == eos) | (k == pad)).all():  break
    #
    #     # fast version
    #     if 1:
    #         #incremental_state = {}
    #         incremental_state = torch.jit.annotate(
    #             Dict[str, Dict[str, Optional[Tensor]]],
    #             torch.jit.annotate(Dict[str, Dict[str, Optional[Tensor]]], {}),
    #         )
    #         for t in range(max_length-1):
    #             #last_token = token [:,:(t+1)]
    #             #text_embed = self.token_embed(last_token)
    #             #text_embed = self.text_pos(text_embed) #text_embed + text_pos[:,:(t+1)] #
    #
    #             last_token = token[:, t]
    #             text_embed = self.token_embed(last_token)
    #             text_embed = text_embed + text_pos[:,t] #
    #             text_embed = text_embed.reshape(1,batch_size,text_dim)
    #
    #             x = self.text_decode.forward_one(text_embed, image_embed, incremental_state)
    #             x = x.reshape(batch_size,decoder_dim)
    #             #print(incremental_state.keys())
    #
    #             l = self.logit(x)
    #             k = torch.argmax(l, -1)  # predict max
    #             token[:, t+1] = k
    #             if ((k == eos) | (k == pad)).all():  break
    #
    #     predict = token[:, 1:]
    #     return predict




# loss #################################################################
def seq_cross_entropy_loss(logit, token, length):
    truth = token[:, 1:]
    L = [l - 1 for l in length]
    logit = pack_padded_sequence(logit, L, batch_first=True).data
    truth = pack_padded_sequence(truth, L, batch_first=True).data
    loss = F.cross_entropy(logit, truth, ignore_index=STOI['<pad>'])
    return loss

# https://www.aclweb.org/anthology/2020.findings-emnlp.276.pdf
def seq_focal_cross_entropy_loss(logit, token, length):
    gamma = 0.5 # {0.5,1.0}
    #label_smooth = 0.90

    #---
    truth = token[:, 1:]
    L = [l - 1 for l in length]
    logit = pack_padded_sequence(logit, L, batch_first=True).data
    truth = pack_padded_sequence(truth, L, batch_first=True).data
    #loss = F.cross_entropy(logit, truth, ignore_index=STOI['<pad>'])
    #non_pad = torch.where(truth != STOI['<pad>'])[0]  # & (t!=STOI['<sos>'])


    # ---
    #p = F.softmax(logit,-1)
    #logp = - torch.log(torch.clamp(p, 1e-4, 1 - 1e-4))

    logp = F.log_softmax(logit, -1)
    logp = logp.gather(1, truth.reshape(-1,1)).reshape(-1)
    p = logp.exp()

    loss = - ((1 - p) ** gamma)*logp  #focal
    #loss = - ((1 + p) ** gamma)*logp  #anti-focal
    loss = loss.mean()
    return loss



# check #################################################################


def run_check_net():
    patch, coord, num_patch, patch_pad_mask = make_dummy_data()
    batch_size = len(patch)

    token  = np.full((batch_size, max_length), STOI['<pad>'], np.int64) #token
    length = np.random.randint(5, max_length-2, batch_size)
    length = np.sort(length)[::-1].copy()


    max_of_length = max(length)
    token_pad_mask  = np.zeros((batch_size, max_of_length, max_of_length))
    for b in range(batch_size):
        l = length[b]
        t = np.random.choice(vocab_size,l)
        t = np.insert(t,0,     STOI['<sos>'])
        t = np.insert(t,len(t),STOI['<eos>'])
        L = len(t)
        token[b,:L]=t
        token_pad_mask [b, :L, :L] = 1

    token = torch.from_numpy(token).long()
    token_pad_mask = torch.from_numpy(token_pad_mask).byte()



    #---
    net = Net()
    net.train()

    logit = net(patch, coord, token, patch_pad_mask, token_pad_mask)
    loss = seq_cross_entropy_loss(logit, token, length)


    print('vocab_size',vocab_size)
    print('max_length',max_length)
    print('')
    print(length)
    print(length.shape)
    print(token.shape)
    print('---')

    print(logit.shape)
    print(loss)
    print('---')


    #---
    # print('torch.jit.script(net)')
    # net.eval()
    # net = torch.jit.script(net)
    #
    # predict = net.forward_argmax_decode(patch, coord, mask)
    # print(predict.shape)


# main #################################################################
if __name__ == '__main__':
     run_check_net()