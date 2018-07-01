"""
File used for models, loss functions, and custom layers
"""

import torch
import torch.nn.functional as F
import torch.nn as nn
import bcolz

def geometric_unit(pred_coords, pred_torsions, bond_angles, bond_lens):
    for i in range(3):
        #coordinates of last three atoms
        A, B, C = pred_coords[-3], pred_coords[-2], pred_coords[-1]
        
        #internal coordinates
        T = bond_angles[i]
        R = bond_lens[i]
        P = pred_torsions[:, i]

        #6x3 one triplet for each sample in the batch
        D2 = torch.stack([-R*torch.ones(P.size())*torch.cos(T), 
                          R*torch.cos(P)*torch.sin(T),
                          R*torch.sin(P)*torch.sin(T)], dim=1)

        #bsx3 one triplet for each sample in the batch
        BC = C - B
        bc = BC/torch.norm(BC, 2, dim=1, keepdim=True)

        AB = B - A

        N = torch.cross(AB, bc)
        n = N/torch.norm(N, 2, dim=1, keepdim=True)

        M = torch.stack([bc, torch.cross(n, bc), n], dim=2)

        D = torch.bmm(M, D2.view(-1,3,1)).squeeze() + C
        pred_coords = torch.cat([pred_coords, D.view(1,-1,3)])
    
    return pred_coords

def pair_dist(x, y=None):
    #norm takes sqrt, undo that by squaring
    #x_norm = torch.pow(torch.norm(x, 2), 2).view(-1,1)
    x_norm = (x**2).sum(1).view(-1, 1)
    if y is not None:
        y_t = torch.t(y)
        #y_norm = torch.pow(torch.norm(x, 2), 2).view(1,-1)
        y_norm = (y**2).sum(1).view(1, -1)
    else:
        y_t = torch.t(x)
        y_norm = x_norm.view(1,-1)
    
    dist = x_norm + y_norm - 2*torch.mm(x, y_t)
    if y is None:
        #enforce all zeros along the diagonal
        dist = dist - torch.diag(torch.diag(dist))
    dist = F.relu(dist)
    
    return torch.pow(dist, 0.5)

def batch_pair_dist(x):
    x = x.permute(dims=(1,0,2))
    x_norm = (x**2).sum(2).view(x.size(0), -1, 1)
    
    y_t = x.permute(0,2,1)
    y_norm = x_norm.view(x.size(0), 1, -1)
    
    dist = x_norm + y_norm - 2*torch.bmm(x, y_t)
    dist = F.relu(dist)
    
    return torch.pow(dist, 0.5)

class dRMSD(nn.Module):
    def __init__(self):
        super(dRMSD, self).__init__()

    def forward(self, x, y):
        #put batch on the first dimension
        x = x.permute(dims=(1,0,2))
        y = y.permute(dims=(1,0,2))
        
        dRMSD = torch.tensor([0.])
        for i in range(x.size(0)):
            #3 to exclude random first 3 coords
            #get indices where coordinates are not [0.,0.,0.]
            #sum across row for accurate results, there may be a more efficient way?
            idx = torch.tensor([p for p,co in enumerate(y[i]) if co.sum() != 0], dtype=torch.long)
            xdist_mat = pair_dist(torch.index_select(x[i], 0, idx))
            ydist_mat = pair_dist(torch.index_select(y[i], 0, idx))
            
            D = ydist_mat - xdist_mat
            dRMSD += torch.norm(D, 2)/(((x.size(1)**2)/2 - x.size(1))**0.5)
            
        return dRMSD/x.size(0) #average over the batch
    
def create_emb_layer(aa2vec_path):
    aa2vec = torch.tensor(bcolz.open(aa2vec_path), requires_grad=True)
    vocab_sz, embed_dim = aa2vec.size()
    emb_layer = nn.Embedding(vocab_sz, embed_dim)
    emb_layer.load_state_dict({'weight': aa2vec})

    return emb_layer, vocab_sz, embed_dim