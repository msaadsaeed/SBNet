
import torch.nn as nn


def make_fc_1d(f_in, f_out):
    return nn.Sequential(nn.Linear(f_in, f_out), 
                        nn.BatchNorm1d(f_out),
                        nn.ReLU(inplace=True),
                        nn.Dropout(p=0.5))


'''
Embedding Extraction Module
'''        

class EmbedBranch(nn.Module):
    def __init__(self, feat_dim, embedding_dim):
        super(EmbedBranch, self).__init__()
        self.fc1 = make_fc_1d(feat_dim, embedding_dim).cuda()
        self.fc2 = make_fc_1d(embedding_dim, embedding_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = nn.functional.normalize(x) 
        return x

'''
Main Module
'''

class FOP(nn.Module):
    def __init__(self, args, feat_dim, n_class):
        super(FOP, self).__init__()
        
        self.embed_branch = EmbedBranch(feat_dim, args.dim_embed)
        
        self.logits_layer = nn.Linear(args.dim_embed, n_class)

        if args.cuda:
            self.cuda()

    def forward(self, feats):
        feats = self.embed_branch(feats)
        logits = self.logits_layer(feats)
        
        return feats, logits
    
    def train_forward(self, feats):
        
        comb = self(feats)
        return comb
