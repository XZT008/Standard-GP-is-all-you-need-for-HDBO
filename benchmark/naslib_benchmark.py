from naslib.search_spaces import (
    NasBench101SearchSpace,
    NasBench201SearchSpace,
    NasBenchNLPSearchSpace,
    NasBenchASRSearchSpace,
    TransBench101SearchSpaceMacro,
    TransBench101SearchSpaceMicro
)
from naslib.utils import get_dataset_api
from naslib.search_spaces.core.query_metrics import Metric
from naslib.search_spaces.nasbenchasr.conversions import copy_structure
import numpy as np
import random


search_spaces = {
    'nasbench101': NasBench101SearchSpace,
    'nasbench201': NasBench201SearchSpace,
    'nlp': NasBenchNLPSearchSpace,
    'asr': NasBenchASRSearchSpace,
    'transbench101_micro': TransBench101SearchSpaceMicro,
    'transbench101_macro': TransBench101SearchSpaceMacro,
}

tasks = {
    'nasbench101': ['cifar10'],
    'nasbench201': ['cifar100', 'ImageNet16-120'],
    'darts': ['cifar10'],
    'nlp': ['treebank'],
    'asr': ['timit'],
    'transbench101_micro': [
        'class_scene',
        'class_object',
        'jigsaw',
        'room_layout',
        'segmentsemantic',
        'normal',
        'autoencoder'
    ],
    'transbench101_macro': [
        'class_scene',
        'class_object',
        'jigsaw',
        'room_layout',
        'segmentsemantic',
        'normal',
        'autoencoder'
    ]
}


ASR_OP_NAMES = ['linear', 'conv5', 'conv5d2', 'conv7', 'conv7d2', 'zero']
asr_search_space = [[len(ASR_OP_NAMES)] + [2]*(idx+1) for idx in range(3)]


class NasBench201:
    def __init__(self, dataset="cifar100"):
        self.dims = 30
        self.n_category = [5, 5, 5, 5, 5, 5]

        assert sum(self.n_category) == self.dims
        self.graph = NasBench201SearchSpace()
        self.dataset = dataset
        self.dataset_api = get_dataset_api(search_space="nasbench201", dataset=self.dataset)
        
        self.lb = np.zeros(self.dims)
        self.ub = np.ones(self.dims)
        self.opt_val = 1.0
        
    def __call__(self, x):
        assert len(x) == self.dims
        assert x.ndim == 1
        assert np.all(x <= self.ub) and np.all(x >= self.lb)
        op_indices = []
        i = 0
        for idx, j in enumerate(self.n_category):
            choice = np.argmax(x[i: i+j])
            op_indices.append(choice)
            i += j
        op_indices = np.array(op_indices)
        graph = self.graph.clone()
        graph.set_op_indices(op_indices)
        result = graph.query(Metric.VAL_ACCURACY, dataset=self.dataset, dataset_api=self.dataset_api)
        return result / 100.0


if __name__ == '__main__':
    nas_problem = NasBench201()
    result = nas_problem(np.random.uniform(0, 1, nas_problem.dims))
    print(result)
    result = nas_problem(np.random.uniform(0, 1, nas_problem.dims))
    print(result)
