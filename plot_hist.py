import matplotlib.pyplot as plt
import numpy as np
from residual_calculation import residual
from basis_forward_propagation import decode_data

y = []
for i in ['00', '01', '02', '03', '04','05', '06', '10', '11', '20', '23', '24', '25', '27', '28', '29', '30', '32', '33']:

    data = decode_data(f"hardware/data/jana{i}")
    name = f"jana{i}"
    f_a, tau_a, = residual(data, name)
    if len(y) == 0:
        y = f_a
    else: 
        y = np.append(y, f_a, axis=0)

fig, ax = plt.subplots((3))
ax[0].hist(y[:, 0], bins=20)   
ax[0].set_xlabel('N')
ax[0].set_ylabel('count')

ax[0].set_title('f_a X')
ax[1].hist(y[:, 1], bins=20)
ax[1].set_xlabel('N')
ax[1].set_ylabel('count')

ax[1].set_title('f_a Y')

ax[2].hist(y[:, 2], bins=20)
ax[2].set_xlabel('N')
ax[2].set_ylabel('count')

ax[2].set_title('f_a Z')

plt.tight_layout()


plt.savefig('pdf/Supervised learning/Hist f.png')

