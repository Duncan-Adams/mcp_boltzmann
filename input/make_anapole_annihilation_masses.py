import numpy as np

R = 3
mchi_min_MeV = 1e-3
mchi_max_MeV = 1e3

num_masses = 20

masses = np.geomspace(mchi_min_MeV, mchi_max_MeV, num_masses)

with open(f'anapole_mcp_annihilation_R_{R}.csv', 'w') as f:

    for mass in masses:
        print(f'{R*mass:.3e}, {1e-2*R*mass:.3e}, {1e5*R*mass:.3e}', file=f)
    
