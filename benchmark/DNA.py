import numpy as np
import LassoBench


class DNA_Lasso:
    def __init__(self,):
        self.dims = 180
        self.lb = np.zeros(180, )
        self.ub = np.ones(180, )

    def __call__(self, x):
        x = np.array(x)
        if x.ndim == 0:
            x = np.expand_dims(x, 0)
        assert x.ndim == 1
        x_real = 2.0*(x-0.5)
        real_bench = LassoBench.RealBenchmark(pick_data='DNA')
        loss = real_bench.evaluate(x_real)
        return -loss
