import numpy as np
import lapsvm
from semi_cross_val import semi_cross_valid
from tqdm import tqdm

class pso():
    def __init__(self, n=50, dim=3, w=0.9, c1=2, c2=2, iteration=200, max_value=10000, min_value=0.0001, k=5):
            self.n = n
            self.dim = dim
            self.w = w
            self.c1 = c1
            self.c2 = c2
            self.iteration = iteration
            self.max_value = max_value
            self.min_value = min_value
            self.k = k

    def _cal_current_f(self, particle_loc, x_i, x_u, y):
        current_f = []
        for i in range(self.n):
            options = {'gamma_A': particle_loc[i, 0],
                       'gamma_I': particle_loc[i, 1],
                       'k_neighbor': 5,
                       'kernal_param': particle_loc[i, 2],
                       't': 1}
            machine = lapsvm.LapSVM(options)
            current_f.append(semi_cross_valid(x_i, x_u, y, model=machine, k=self.k))
        return current_f

    def fit(self, x_i, x_u, y):
        particle_loc = np.random.rand(self.n, self.dim) * (self.max_value - self.min_value) + self.min_value
        particle_dir = np.random.rand(self.n, self.dim)
        current_f = self._cal_current_f(particle_loc, x_i, x_u, y)
        pbest = particle_loc
        self.gbest = particle_loc[np.argmax(current_f), :]
        pbest_f = current_f.copy()
        gbest_f = max(current_f)

        for i in tqdm(range(self.iteration), desc='pso_iteration'):
            #print('\riteration is : %d / %d' %(i+1, self.iteration), flush=True)
            particle_dir = self.w * particle_dir + self.c1 * np.random.random() * (
                        pbest - particle_loc) + self.c2 * np.random.random() * (self.gbest - particle_loc)
            particle_loc += particle_dir
            particle_loc[particle_loc > self.max_value] = self.max_value
            particle_loc[particle_loc < self.min_value] = self.min_value
            current_f = self._cal_current_f(particle_loc, x_i, x_u, y)

            for i in range(self.n):
                if current_f[i] > pbest_f[i]:
                    pbest[i, :] = particle_loc[i, :]
                    pbest_f[i] = current_f[i]

            if max(pbest_f) > gbest_f:
                self.gbest = pbest[np.argmax(pbest_f), :]
                gbest_f = max(pbest)
        return self.gbest

if __name__ == '__main__':
    from sklearn.datasets import make_moons

    X, Y = make_moons(n_samples=200, noise=0.05)
    ind_0 = np.nonzero(Y == 0)[0]
    ind_1 = np.nonzero(Y == 1)[0]
    Y[ind_0] = -1
    ind_l0 = np.random.choice(ind_0, 5, False)
    ind_u0 = np.setdiff1d(ind_0, ind_l0)
    ind_l1 = np.random.choice(ind_1, 5, False)
    ind_u1 = np.setdiff1d(ind_1, ind_l1)
    x_i = np.vstack([X[ind_l0, :], X[ind_l1, :]])
    y = np.hstack([Y[ind_l0], Y[ind_l1]])
    x_u = np.vstack([X[ind_u0, :], X[ind_u1, :]])

    param_opt = pso()
    param = param_opt.fit(x_i, x_u, y)
    print(param)


'''particle_loc = np.random.rand(n, dim) * (max_value - min_value) + min_value
particle_dir = np.random.rand(n, dim)

current_f = []
for i in range(n):
    options = {'gamma_A': particle_loc[i, 0],
               'gamma_I': particle_loc[i, 1],
               'k_neighbor': 5,
               'kernal_param': particle_loc[i, 2],
               't': 1}
    machine = lapsvm.LapSVM(options)
    current_f.append(semi_cross_valid(x_i, x_u, y, model=machine, k=k))
pbest = particle_loc
gbest = particle_loc[np.argmax(current_f), :]
pbest_f = current_f.copy()
gbest_f = max(current_f)

for i in range(iteration):
    particle_dir = w * particle_dir + c1 * np.random.random() * (pbest - particle_loc) + c2 * np.random.random() * (gbest - particle_loc)
    particle_loc += particle_dir
    particle_loc[particle_loc > max_value] = max_value
    particle_loc[particle_loc < min_value] = min_value

    for i in range(n):
        options = {'gamma_A': particle_loc[i, 0],
                   'gamma_I': particle_loc[i, 1],
                   'k_neighbor': 5,
                   'kernal_param': particle_loc[i, 2],
                   't': 1}
        machine = lapsvm.LapSVM(options)
        current_f.append(semi_cross_valid(x_i, x_u, y, model=machine, k=k))

    for i in range(n):
        if current_f[i] > pbest_f[i]:
            pbest[i, :] = particle_loc[i, :]
            pbest_f[i] = current_f[i]

    if max(pbest_f) > gbest_f:
        gbest = pbest[np.argmax(pbest_f), :]
        gbest_f = max(pbest)

print(gbest)'''
