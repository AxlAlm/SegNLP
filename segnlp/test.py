# %%
import torch


def CrossEntropyLoss_custom(logs, targets):
    out = torch.diag(logs[:, targets])
    return -out


def NLLLoss_custom(logs, targets):
    out = torch.diag(logs[:, targets])
    return -out


log_softmax = torch.nn.LogSoftmax(dim=1)
soft_max = torch.nn.Softmax(dim=1)
cross_entropy_loss = torch.nn.CrossEntropyLoss(reduction="none")
nll_loss = torch.nn.NLLLoss(reduction="none")

# %%
x = torch.randn(3, 5)
z = torch.zeros(3, 5)
x_log = log_softmax(x)
x_soft = soft_max(x)

i = torch.cat((x, z), dim=0)
i_log = torch.cat((x_log, z), dim=0)
i_soft = torch.cat((x_soft, z), dim=0)
y = torch.LongTensor([0, 1, 2, 0, 0, 0])

out_cross_entropy = cross_entropy_loss(i, y)
out_cross_entropy_custom = CrossEntropyLoss_custom(i_log, y)
out_nll_loss = nll_loss(i_log, y)
out_nll_loss_custom = NLLLoss_custom(i_log, y)

print("Torch CrossEntropyLoss:\t\t", out_cross_entropy)
print("Custom CrossEntropyLoss:\t", out_cross_entropy_custom)
print("Torch NLL loss:\t\t\t", out_nll_loss)
print("Custom NLL loss:\t\t", out_nll_loss_custom)
print()

# %%
out_cross_entropy = cross_entropy_loss(i, y)
out_cross_entropy_custom = CrossEntropyLoss_custom(i_soft, y)
out_nll_loss = nll_loss(i_soft, y)
out_nll_loss_custom = NLLLoss_custom(i_soft, y)

print("Torch CrossEntropyLoss:\t\t", out_cross_entropy)
print("Custom CrossEntropyLoss:\t", out_cross_entropy_custom)
print("Torch NLL loss:\t\t\t", out_nll_loss)
print("Custom NLL loss:\t\t", out_nll_loss_custom)

# %%
import torch

a = torch.randint(-2, 3, (3,5))
b = a < 0
idxs = torch.nonzero(b)
print(a)

a[idxs[:, 0], idxs[:, 1]] = idxs[:, 1] * 10
print(a)
# %%
