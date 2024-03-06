import torch
import numpy as np
from sklearn.preprocessing import MinMaxScaler

np.random.seed(0)


class FuncRosenbrock(object):

    def __init__(
            self,
            dim,
            maximize=True,
    ):
        self.dim = dim
        self.dims = dim
        self.maximize = maximize

        self.lb = -2.048 * np.ones(dim)
        self.ub = 2.048 * np.ones(dim)

        self.inputs_scaler = MinMaxScaler()
        self.inputs_scaler.fit(np.vstack([self.lb, self.ub]))

        self._opt_inputs = np.ones(self.dim)

    def _scale_inputs(self, X):
        if X.ndim == 1 and X.size == self.dim:
            X = X.reshape([1, self.dim])

        assert X.shape[1] == self.dim
        Xr = self.inputs_scaler.inverse_transform(X)
        return Xr

    def get_opts(self, ):
        if self._opt_inputs.ndim == 1:
            self._opt_inputs = self._opt_inputs.reshape([1, -1])

        Xopts = self.inputs_scaler.transform(self._opt_inputs)
        yopts = self.query(Xopts)
        return Xopts, yopts

    def query(self, X):
        X = self._scale_inputs(X)
        num = X.shape[0]
        F = np.zeros([num, self.dim - 1])
        for i in range(self.dim - 1):
            F[:, i] = 100 * ((X[:, i + 1] - X[:, i] ** 2) ** 2) + (X[:, i] - 1.0) ** 2
        #
        f = F.sum(1).reshape([-1, 1])

        if self.maximize:
            return -f
        else:
            return f


class FuncStybTang(object):

    def __init__(
            self,
            dim,
            maximize=True,
    ):
        self.dim = dim
        self.dims = dim
        self.maximize = maximize

        self.lb = -5.0 * np.ones(dim)
        self.ub = 5.0 * np.ones(dim)

        self.inputs_scaler = MinMaxScaler()
        self.inputs_scaler.fit(np.vstack([self.lb, self.ub]))

        self._opt_inputs = -2.903534 * np.ones(self.dim)

    def _scale_inputs(self, X):
        if X.ndim == 1 and X.size == self.dim:
            X = X.reshape([1, self.dim])

        assert X.shape[1] == self.dim
        Xr = self.inputs_scaler.inverse_transform(X)
        return Xr

    def get_opts(self, ):
        if self._opt_inputs.ndim == 1:
            self._opt_inputs = self._opt_inputs.reshape([1, -1])

        Xopts = self.inputs_scaler.transform(self._opt_inputs)
        yopts = self.query(Xopts)
        return Xopts, yopts

    def query(self, X):
        X = self._scale_inputs(X)
        F = X ** 4 - 16 * (X ** 2) + 5 * X
        f = 0.5 * F.sum(1).reshape([-1, 1])
        if self.maximize:
            return -f
        else:
            return f


class FuncAckley(object):
    def __init__(
            self,
            dim,
            maximize=True,
    ):
        self.dim = dim
        self.dims = dim
        self.maximize = maximize

        self.lb = -32.768 * np.ones(dim)
        self.ub = 32.768 * np.ones(dim)

        self.inputs_scaler = MinMaxScaler()
        self.inputs_scaler.fit(np.vstack([self.lb, self.ub]))

        self._opt_inputs = np.zeros(self.dim)

    def _scale_inputs(self, X):
        if X.ndim == 1 and X.size == self.dim:
            X = X.reshape([1, self.dim])

        assert X.shape[1] == self.dim
        Xr = self.inputs_scaler.inverse_transform(X)
        return Xr

    def get_opts(self, ):
        if self._opt_inputs.ndim == 1:
            self._opt_inputs = self._opt_inputs.reshape([1, -1])

        Xopts = self.inputs_scaler.transform(self._opt_inputs)
        yopts = self.query(Xopts)
        return Xopts, yopts

    def query(self, X):
        X = self._scale_inputs(X)
        part1 = -20 * np.exp(-0.2 * np.sqrt((np.mean(X ** 2, axis=1))))
        part2 = -np.exp(np.mean(np.cos(2 * np.pi * X), axis=1))
        f = part1 + part2 + 20 + np.exp(1)
        if self.maximize:
            return -f.reshape([-1, 1])
        else:
            return f.reshape([-1, 1])


class FuncRosenbrock100(object):

    def __init__(
            self,
            dim,
            maximize=True,
    ):
        self.dim = dim
        self.dims = dim
        self.latent_dim = 100
        self.maximize = maximize

        self.lb = -2.048 * np.ones(dim)
        self.ub = 2.048 * np.ones(dim)

        self.inputs_scaler = MinMaxScaler()
        self.inputs_scaler.fit(np.vstack([self.lb, self.ub]))

        self._opt_inputs = np.ones(self.dim)

    def _scale_inputs(self, X):
        if X.ndim == 1 and X.size == self.dim:
            X = X.reshape([1, self.dim])

        assert X.shape[1] == self.dim
        Xr = self.inputs_scaler.inverse_transform(X)
        Xr = Xr[:, :self.latent_dim]
        return Xr

    def get_opts(self, ):
        if self._opt_inputs.ndim == 1:
            self._opt_inputs = self._opt_inputs.reshape([1, -1])

        Xopts = self.inputs_scaler.transform(self._opt_inputs)
        yopts = self.query(Xopts)
        return Xopts, yopts

    def query(self, X):
        X = self._scale_inputs(X)
        num = X.shape[0]
        F = np.zeros([num, self.latent_dim - 1])
        for i in range(self.latent_dim - 1):
            F[:, i] = 100 * ((X[:, i + 1] - X[:, i] ** 2) ** 2) + (X[:, i] - 1.0) ** 2
        #
        f = F.sum(1).reshape([-1, 1])

        if self.maximize:
            return -f
        else:
            return f


class FuncAckley150(object):
    def __init__(
            self,
            dim,
            maximize=True,
    ):
        self.dim = dim
        self.dims = dim
        self.latent_dim = 150
        self.maximize = maximize

        self.lb = -32.768 * np.ones(dim)
        self.ub = 32.768 * np.ones(dim)

        self.inputs_scaler = MinMaxScaler()
        self.inputs_scaler.fit(np.vstack([self.lb, self.ub]))

        self._opt_inputs = np.zeros(self.dim)

    def _scale_inputs(self, X):
        if X.ndim == 1 and X.size == self.dim:
            X = X.reshape([1, self.dim])

        assert X.shape[1] == self.dim

        Xr = self.inputs_scaler.inverse_transform(X)
        Xr = Xr[:, :self.latent_dim]
        return Xr

    def get_opts(self, ):
        if self._opt_inputs.ndim == 1:
            self._opt_inputs = self._opt_inputs.reshape([1, -1])

        Xopts = self.inputs_scaler.transform(self._opt_inputs)
        yopts = self.query(Xopts)
        return Xopts, yopts

    def query(self, X):
        X = self._scale_inputs(X)
        part1 = -20 * np.exp(-0.2 * np.sqrt((np.mean(X ** 2, axis=1))))
        part2 = -np.exp(np.mean(np.cos(2 * np.pi * X), axis=1))
        f = part1 + part2 + 20 + np.exp(1)
        if self.maximize:
            return -f.reshape([-1, 1])
        else:
            return f.reshape([-1, 1])


class FuncHartmann6(object):
    def __init__(
            self,
            dim,
            maximize=True,
    ):
        self.dim = dim
        self.dims = dim
        self.latent_dim = 6
        self.maximize = maximize

        self.lb = np.zeros(self.dim)
        self.ub = np.ones(self.dim)

        self.inputs_scaler = MinMaxScaler()
        self.inputs_scaler.fit(np.vstack([self.lb, self.ub]))

        self._opt_inputs = np.array([0.20169, 0.150011, 0.476874, 0.275332, 0.311652, 0.6573])

        A = [
            [10, 3, 17, 3.5, 1.7, 8],
            [0.05, 10, 17, 0.1, 8, 14],
            [3, 3.5, 1.7, 10, 17, 8],
            [17, 8, 0.05, 10, 0.1, 14],
        ]
        P = [
            [1312, 1696, 5569, 124, 8283, 5886],
            [2329, 4135, 8307, 3736, 1004, 9991],
            [2348, 1451, 3522, 2883, 3047, 6650],
            [4047, 8828, 8732, 5743, 1091, 381],
        ]

        self.A = np.array(A)
        self.P = np.array(P)
        self.ALPHA = np.array([1.0, 1.2, 3.0, 3.2])

    def _scale_inputs(self, X):
        if X.ndim == 1 and X.size == self.dim:
            X = X.reshape([1, self.dim])

        assert X.shape[1] == self.dim
        Xr = self.inputs_scaler.inverse_transform(X)
        Xr = Xr[:, :self.latent_dim]
        return Xr

    def get_opts(self, ):
        if self._opt_inputs.ndim == 1:
            self._opt_inputs = self._opt_inputs.reshape([1, -1])

        Xopts = self.inputs_scaler.transform(self._opt_inputs)
        yopts = self.query(Xopts)
        return Xopts, yopts

    def query(self, X):
        X = self._scale_inputs(X)
        inner_sum = np.sum(self.A * (np.expand_dims(X, axis=-2) - 0.0001 * self.P) ** 2, axis=-1)
        H = -(np.sum(self.ALPHA * np.exp(-inner_sum), axis=-1))

        if self.maximize:
            return -H.reshape([-1, 1])
        else:
            return H.reshape([-1, 1])


class FuncRosenbrock_V1(object):

    def __init__(
            self,
            dim,
            maximize=True,
    ):
        self.dim = dim
        self.dims = dim
        self.maximize = maximize

        self.lb = -2.048 * np.ones(dim)
        self.ub = 2.048 * np.ones(dim)

        self.inputs_scaler = MinMaxScaler()
        self.inputs_scaler.fit(np.vstack([self.lb, self.ub]))

        self._opt_inputs = np.linspace(-2.0, 2.0, num=self.dim)

    def _scale_inputs(self, X):
        if X.ndim == 1 and X.size == self.dim:
            X = X.reshape([1, self.dim])

        assert X.shape[1] == self.dim
        Xr = self.inputs_scaler.inverse_transform(X)
        return Xr

    def get_opts(self, ):
        if self._opt_inputs.ndim == 1:
            self._opt_inputs = self._opt_inputs.reshape([1, -1])

        Xopts = self.inputs_scaler.transform(self._opt_inputs)
        yopts = self.query(Xopts)
        return Xopts, yopts

    def query(self, X):
        X = self._scale_inputs(X)

        offset = np.linspace(-2.0, 2.0, num=self.dim) - np.ones(self.dim, )
        X = X - offset
        num = X.shape[0]
        F = np.zeros([num, self.dim - 1])
        for i in range(self.dim - 1):
            F[:, i] = 100 * ((X[:, i + 1] - X[:, i] ** 2) ** 2) + (X[:, i] - 1.0) ** 2
        #
        f = F.sum(1).reshape([-1, 1])

        if self.maximize:
            return -f
        else:
            return f


class FuncStybTang_V1(object):

    def __init__(
            self,
            dim,
            maximize=True,
    ):
        self.dim = dim
        self.dims = dim
        self.maximize = maximize

        self.lb = -5.0 * np.ones(dim)
        self.ub = 5.0 * np.ones(dim)

        self.inputs_scaler = MinMaxScaler()
        self.inputs_scaler.fit(np.vstack([self.lb, self.ub]))

    def _scale_inputs(self, X):
        if X.ndim == 1 and X.size == self.dim:
            X = X.reshape([1, self.dim])

        assert X.shape[1] == self.dim
        Xr = self.inputs_scaler.inverse_transform(X)
        return Xr

    def query(self, X):
        X = self._scale_inputs(X)
        offset = np.linspace(0.0, 7.5, num=self.dim)

        X = X - offset
        F = X ** 4 - 16.0 * (X ** 2) + 5.0 * X
        f = 0.5 * F.sum(1).reshape([-1, 1])
        if self.maximize:
            return -f
        else:
            return f


class FuncRosenbrock100_V1(object):

    def __init__(
            self,
            dim,
            maximize=True,
    ):
        self.dim = dim
        self.dims = dim
        self.latent_dim = 100
        self.maximize = maximize

        self.lb = -2.048 * np.ones(dim)
        self.ub = 2.048 * np.ones(dim)

        self.inputs_scaler = MinMaxScaler()
        self.inputs_scaler.fit(np.vstack([self.lb, self.ub]))

        self._opt_inputs = np.ones(self.dim)

    def _scale_inputs(self, X):
        if X.ndim == 1 and X.size == self.dim:
            X = X.reshape([1, self.dim])

        assert X.shape[1] == self.dim
        Xr = self.inputs_scaler.inverse_transform(X)
        Xr = Xr[:, :self.latent_dim]
        return Xr

    def get_opts(self, ):
        if self._opt_inputs.ndim == 1:
            self._opt_inputs = self._opt_inputs.reshape([1, -1])

        Xopts = self.inputs_scaler.transform(self._opt_inputs)
        yopts = self.query(Xopts)
        return Xopts, yopts

    def query(self, X):
        X = self._scale_inputs(X)

        offset = np.linspace(-2.0, 2.0, num=self.latent_dim) - np.ones(self.latent_dim, )
        X = X - offset
        num = X.shape[0]
        F = np.zeros([num, self.latent_dim - 1])
        for i in range(self.latent_dim - 1):
            F[:, i] = 100 * ((X[:, i + 1] - X[:, i] ** 2) ** 2) + (X[:, i] - 1.0) ** 2
        #
        f = F.sum(1).reshape([-1, 1])

        if self.maximize:
            return -f
        else:
            return f



