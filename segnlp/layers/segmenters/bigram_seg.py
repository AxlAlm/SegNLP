
#pytroch
import torch
from torch.functional import Tensor 
import torch.nn as nn
import torch.nn.functional as F


class BigramSeg(nn.Module):
    """
        https://www.aclweb.org/anthology/P16-1105.pdf

    """
    def __init__(
        self,
        input_size,
        hidden_size,
        output_size,
        dropout=0.0,
    ):
        super().__init__()
        self.output_size = output_size
        self.clf = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, output_size),
        )

    def forward(
        self,
        input: Tensor,
        ):

        # sizes
        batch_size = input.size(0)
        max_lenght = input.size(1)
        size = [batch_size, max_lenght, self.output_size]
        device = input.device

        # construct tensors
        logits = torch.zeros(size, device=device)
        preds = torch.zeros(batch_size, max_lenght, dtype=torch.long, device=device)

        # predict labels token by token
        for i in range(max_lenght):
            prev_label_one_hot = F.one_hot(
                                preds[:, i - 1 ],
                                num_classes=self.output_size
                                )
            x = torch.cat((input[:, i], prev_label_one_hot), dim=-1)
            logit = self.clf(x)  # (B, NE-OUT)
            logits[:, i] = logit
            preds[:, i] = torch.argmax(logits[:, i], dim=1)
          
   
        return logits, preds
    