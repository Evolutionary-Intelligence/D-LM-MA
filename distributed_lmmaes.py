from pypoplib.lmmaes import LMMAES
from pypoplib.distributed_es import DistributedES


class DistributedLMMAES(DistributedES):
    """Distributed Limited-Memory Matrix Adaptation Evolution Strategy (D-LM-MA).

    Qiqi Duan, Guochen Zhou, Chang Shao, Yijun Yang and Yuhui Shi.
    Collective Learning of Low-Memory Matrix Adaptation for Large-Scale Black-Box Optimization.
    """
    def __init__(self, problem, options):
        DistributedES.__init__(self, problem, options)
        self._customized_class = LMMAES
