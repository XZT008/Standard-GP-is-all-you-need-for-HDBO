from sklearn.preprocessing import StandardScaler

from infras.randutils import *
from data.synfuncs import *
from tqdm import tqdm

class RealDataset(object):
    def __init__(
            self,
            func,
            n_init,
            method,
            seed,
            X=None,
            Y=None,
            if_tqdm=False
    ):
        self.seed = seed
        self.func = func
        self.n_init = n_init

        self.dim = func.dims
        if X is not None and Y is not None:
            self.X = X
            self.y = Y
        else:
            self.X = generate_with_bounds(
                N=n_init,
                lb=np.zeros(self.dim),
                ub=np.ones(self.dim),
                method=method,
                seed=seed,
            )
            if not if_tqdm:
                self.y = np.array([self.func(x) for x in self.X], dtype=np.double)
            else:
                self.y = np.array([self.func(x) for x in tqdm(self.X)], dtype=np.double)
        self.scaler = StandardScaler()
        self.scaler.fit(self.y.reshape(-1, 1))

    def get_data(self, normalize=True, tensor=True, device=torch.device('cpu')):
        X = self.X
        y = self.y

        if normalize:
            y = self.scaler.transform(y.reshape(-1, 1))

        if tensor:
            X = torch.tensor(X).to(torch.float64).to(device)
            y = torch.tensor(y).to(torch.float64).to(device)

        return X, y

    def add(self, X):
        if isinstance(X, torch.Tensor):
            Xstar = X.data.cpu().numpy()
        else:
            Xstar = X

        if Xstar.ndim == 1:
            Xstar = Xstar.reshape([1,-1])

        ystar = np.array([self.func(x) for x in Xstar], dtype=float)

        self.X = np.vstack([self.X, Xstar])
        self.y = np.concatenate([self.y, ystar])

        self.scaler = StandardScaler()
        self.scaler.fit(self.y.reshape(-1, 1))

    def get_curr_max_unnormed(self):
        return np.max(self.y)

