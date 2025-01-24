import pandas as pd
import numpy as np
# add the path to python_bindings to the system path
import sys
sys.path.append('./build/python_bindings') 

from pcm_market_share import conditional_share_with_jacobian, unconditional_share_with_jacobian
import time
from  Tasmanian import TasmanianSparseGrid

# create time now variable
start = time.time()
delta = [1.5,4,6,7]
p = [.1,2.2, 8, 12]
sigma_p = 1.0
res = conditional_share_with_jacobian(delta, p, sigma_p)
delta[0] += 1e-5
res1 = conditional_share_with_jacobian(delta, p, sigma_p)
print(f'market share and jacobian \n{res[0]}\n\n{res[1]}')
print(f'numerical jacobian \n{(res1[0]-res[0])/1e-5}')


# prepare stuff for unconditional share
num_prod = 5
num_dimensions = 6
x = np.random.rand(num_prod, num_dimensions)
delta = [i for i in range(1, num_prod+1)]
p = [i**1.2 for i in range(1, num_prod+1)]
sigma_p = 1.0

sigma_x = np.ones(num_dimensions)

grid = TasmanianSparseGrid()
grid.makeGlobalGrid(num_dimensions, 1, 3, "tensor", "gauss-hermite")
points = grid.getPoints() * (2**0.5)                # List of points in the grid
weights = grid.getQuadratureWeights() /(np.pi**(0.5*num_dimensions))    # Quadrature weights
print(f'points size: {len(points)}')
# calculate unconditional share
res = unconditional_share_with_jacobian(delta, x, p, sigma_p, sigma_x, points, weights)
print(f'unconditional share and jacobian \n{res[0].round(3)}\n\n{res[1].round(3)}')

delta[0] += 1e-5
res1 = unconditional_share_with_jacobian(delta, x, p, sigma_p, sigma_x, points, weights)
print(f'numerical jacobian \n{((res1[0]-res[0])/1e-5).round(3)}')

print(f'Time taken: {time.time()-start}')