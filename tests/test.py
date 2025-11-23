import drift as dft
import numpy as np


def main():
    gal = dft.bg_feature()


#     N = 2000
#     state0 = np.zeros((N, 6), dtype=np.float64)
#     state0[:, 0] = 1.0 + 0.02 * np.random.rand(N)
#     state0[:, 4] = 1.0
#     gmc = dft.part_group(state0)
#     N = 100000
#     state1 = np.zeros((N, 6), dtype=np.float64)
#     state1[:, 0] = 1.0 + 0.02 * np.random.rand(N)
#     state1[:, 4] = 1.0
#     iso = dft.test_group(state1)
#     sim = dft.simulation([gal, gmc, iso])
#     _dfs = sim.run()


if __name__ == "__main__":
    main()
