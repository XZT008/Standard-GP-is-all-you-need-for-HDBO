import torch
from sklearn.preprocessing import StandardScaler

from infras.randutils import *
from data.synfuncs import *


class BayesOptDataset(object):
    def __init__(
            self,
            func,
            n_init,
            method,
            seed,
            X=None,
            Y=None
    ):

        self.func = func
        self.n_init = n_init

        self.dim = func.dim

        if X is None and Y is None:
            self.X = generate_with_bounds(
                N=n_init,
                lb=np.zeros(self.dim),
                ub=np.ones(self.dim),
                method=method,
                seed=seed,
            )

            self.y = self.func.query(self.X)

        else:
            self.X = X
            self.y = Y

        self.scaler = StandardScaler()
        # NOTE this is NOT usually a correct way to normalize data, however, since in BO
        # there are few training examples and NOT test on testing data
        self.scaler.fit(self.y)

    def get_data(self, normalize=True, tensor=True, device=torch.device('cpu')):
        X = self.X
        y = self.y

        if normalize:
            y = self.scaler.transform(y)

        if tensor:
            X = torch.tensor(X, dtype=torch.float64).to(device)
            y = torch.tensor(y, dtype=torch.float64).to(device)

        return X, y

    def _clip_inputs(self, X):
        X = np.clip(a=X, a_min=np.zeros(self.dim), a_max=np.ones(self.dim))
        return X

    def add(self, X):
        if isinstance(X, torch.Tensor):
            Xstar = X.data.cpu().numpy()
        else:
            Xstar = X
        if Xstar.ndim == 1:
            Xstar = Xstar.reshape([1,-1])

        Xstar = self._clip_inputs(Xstar)
        ystar = self.func.query(Xstar)

        self.X = np.vstack([self.X, Xstar])
        self.y = np.vstack([self.y, ystar])

        self.scaler = StandardScaler()
        self.scaler.fit(self.y)

    def get_curr_max_unnormed(self):
        return np.max(self.y)