import torch 
import torch.nn as nn
import numpy as np

# def weights_init(m):
#     classname = m.__class__.__name__
#     if classname.find('Conv') != -1:
#         weight_shape = list(m.weight.data.size())
#         fan_in = np.prod(weight_shape[1:4])
#         fan_out = np.prod(weight_shape[2:4]) * weight_shape[0]
#         w_bound = np.sqrt(6. / (fan_in + fan_out))
#         m.weight.data.uniform_(-w_bound, w_bound)
#         m.bias.data.fill_(0)
#     elif classname.find('Linear') != -1:
#         weight_shape = list(m.weight.data.size())
#         fan_in = weight_shape[1]
#         fan_out = weight_shape[0]
#         w_bound = np.sqrt(6. / (fan_in + fan_out))
#         m.weight.data.uniform_(-w_bound, w_bound)
#         m.bias.data.fill_(0)

def weights_init(m):
    if isinstance(m, nn.Linear):
        weight_shape = list(m.weight.data.size())

class ActorCritic(nn.Module):
    def __init__(self, num_inputs, action_space):
        super(ActorCritic, self).__init__()
        # https://github.com/openai/universe-starter-agent/blob/293904f01b4180ecf92dd9536284548108074a44/model.py#L51-L52
        self.convs = nn.ModuleList()
        for i in range(4):
            if i == 0:
                self.convs.append(nn.Conv2d(num_inputs, 32, (3, 3), stride=(2, 2), padding="same"))
            else: 
                self.convs.append(nn.Conv2d(32, 32, (3, 3), stride=(2, 2), padding="same"))
    
    def forward(self, inputs):
        
        # Introduce a "fake" batch dimension of 1 after flatten so that we can do LSTM over time dim
        # Flatten tensor while maintaing batch size
        torch.flatten()

        
