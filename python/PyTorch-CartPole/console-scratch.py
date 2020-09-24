import numpy as np
import torch
import tracemalloc

tracemalloc.start()

# snapshot = tracemalloc.take_snapshot()
# top_stats = snapshot.statistics('lineno')
#
# print("[ Top 10 ]")
# for stat in top_stats[:10]:
#     print(stat)

# # Creation
# x = torch.zeros(5, 3, dtype=torch.double)
# print(x)
# x = x.new_ones(5, 3, dtype=torch.double)      # new_* methods take in sizes
# print(x)
# w = torch.tensor([[1,2], [2,1]])
# print(w)
# x = torch.rand(5,3)    # From uniform distribution 0-1
# x = torch.randn(5,3)   # From normal distribution mean 0 variance 1
# print(x)
# y = torch.randn_like(x)
# print(y)

# # Addition
# print(x+y)
# print(torch.add(x, y))
# y.add_(x)
# print(y)

# # ID
# print(x[:-2, 1])

# # Resize tensor
# x = torch.randn(4, 4)
# y = x.view(16)
# z = x.view(-1, 8)  # the size -1 is inferred from other dimensions
# print(x.size(), y.size(), z.size())

# # Numpy
# a = torch.ones(5)
# print(a)
# b = a.numpy()
# print(b)
# a.add_(1)           # Keeps reference
# print(a)
# print(b)
# a = a.add(1)        # Breaks reference
# print(a)
# print(b)
# a.add_(1)           # Stays broken
# print(a)
# print(b)

# a = np.ones(5, dtype=np.float32)
# b = torch.from_numpy(a)
# np.add(a, 1, out=a)     # Keeps reference
# print(a)
# print(b)
# a += 1                  # Keeps reference
# a[2] = 9
# print(a)
# print(b)
# a = np.add(a, 1)        # Breaks reference
# print(a)
# print(b)
# np.add(a, 1, out=a)     # Stays broken
# print(a)
# print(b)


# # GPU - use ``torch.device`` objects to move tensors in and out of GPU
# if torch.cuda.is_available():
#     device = torch.device("cuda")          # a CUDA device object
#     x = torch.rand(5, 3)
#     y = torch.ones_like(x, device=device)  # directly create a tensor on GPU
#     x = x.to(device)                       # or just use strings ``.to("cuda")``
#     z = x + y
#     print(z)
#     print(z.to("cpu", torch.double))       # ``.to`` can also change dtype together!
# else:
#     print('Nope, cuda is not available.')

