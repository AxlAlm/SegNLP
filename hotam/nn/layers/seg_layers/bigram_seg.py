

import torch.nn as nn
import torch.nn.functional as F

class BigramSegLayer(nn.Module):

    """
        https://www.aclweb.org/anthology/P16-1105.pdf

    """

    def __init__(
                    self,
                    input_size,
                    hidden_size,
                    output_size,
                    label_emb_dim,
                    dropout=0.0,
                    ):
        super().__init__()
        self.label_emb_dim = label_emb_dim
        self.clf = nn.Sequential(
                                        nn.Linear(input_size, hidden_size),
                                        nn.Tanh(),
                                        nn.Dropout(dropout),
                                        nn.Linear(hidden_size, output_size),
                                    )

    def forward(self,
                X,
                lengths,
                ):

        batch_size = X.size(0)
        max_lenght = X.size(1)
        hidden_size = X.size(-1)
        device = X.device

        logits = torch.zeros(batch_size, max_lenght, self.label_emb_dim, device=device)
        probs = torch.zeros(batch_size, max_lenght, self.label_emb_dim, device=device)
        preds = torch.zeros(batch_size, max_lenght, device=device)
        one_hots = torch.zeros(batch_size, max_lenght, self.label_emb_dim, device=device)

        for i in range(seq_length): 
            
            one_hot = F.one_hot(preds[:,i-1], num_classes=self.label_emb_dim)
            one_hots[:,i] = one_hot
            x = th.cat((X[:, i], one_hot), dim=-1)


            logit = self.clf(x)  # (B, NE-OUT)
            logits[:,i ] = logit    
            probs[:,i ] = F.softmax(logit, dim=1)  
            preds[:,i ] = F.argmax(probs[i: ], dim=1)  

        return logits, probs, preds, one_hots
