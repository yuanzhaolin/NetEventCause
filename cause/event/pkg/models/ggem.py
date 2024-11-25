from typing import List, Dict, Tuple, Union
from collections import defaultdict

import numpy as np

from cause.event.pkg.utils.pp import ogata_thinning_univariate
from scipy.stats import gamma


class EventNode:
    def __init__(
        self,
        idx: int,
        parent: int=-1,
        alpha: float = 1,
        beta: float = 10,
        window: float = 0,
        ratio: float = 1.0,
        intensity_base: float = 0,
    ):
        self.idx = idx
        self.parent = parent

        self.intensity_base = intensity_base

        if self.parent != -1:
            self.alpha = alpha
            self.beta = beta
            self.gamma_dist = gamma(alpha, scale=beta)
            self.window = window
            self.ratio = ratio

        # if intensity_params is None:
        # else:
        #     assert (
        #         self.parents
        #         and intensity_params.ndim == len(self.parents)
        #         or (not self.parents and intensity_params > 0)
        #     ), "`intensity_params` needs to be consistent with `parents`"
        #     self.intensity_params = intensity_params

    def get_base_intensity(self, t: float) -> float:
        """Evaluate the intensity value at a time.

        Args:
            t (float):
            history (Dict[str, float]): stores the timestamp of the most recent
              event for each parent, if existent.

        Returns:
            float: intensity valued evaluated at time t.
        """
        return self.intensity_base

    def get_conditional_intensity(self, t: float, history: Dict[int, Tuple[float, int]]) -> float:

        intensity = self.ratio * self.gamma_dist.pdf(
            t - history[self.parent][0]
        ) if t - history[self.parent][0] <= self.window else 0.0

        # intensity = min(intensity, 0.5)
        return intensity

    def upper_bound(self, t) -> float:
        # if type(self.intensity_params) is float:
        #     return self.intensity_params

        if self.parent == -1:
            return self.intensity_base
        else:
            # Reducing the upper_bound to accelerate the simulation.
            # cur_intensity = self.get_conditional_intensity(t)
            # cur_intensity = self.get_conditional_intensity(t)

            return min(0.1, max(self.intensity_base, self.ratio * self.gamma_dist.pdf((self.alpha-1) * self.beta)))


class GammaGraphicalEventModel:
    def __init__(self, nodes: List[EventNode] = []):
        assert len(nodes) > 0
        assert set(range(len(nodes))) == set(node.idx for node in nodes)
        self.nodes = sorted(nodes, key=lambda node: node.idx)

    def simulate(
        self, init_t: float = 0, max_t: float = float("inf")
    ) -> List[Tuple[float, int]]:
        t = init_t
        T = max_t
        events = []
        most_recent = defaultdict(list)

        while t < T:
            temp_P = []

            # generate candiate timestamp for each type
            for node in self.nodes:
                # simulating root events
                ts = ogata_thinning_univariate(
                    intensity=lambda t: node.get_base_intensity(t),
                    upper_bound=node.upper_bound,
                    n_events=1,
                    init_t=t,
                    max_t=T,
                )
                if ts:
                    temp_P.append((ts[0], node.idx, -1))

                # simulate derivative events
                for cause_t, cause_id in reversed(most_recent[node.parent]):
                    if t - cause_t > node.window:
                        break

                    ts = ogata_thinning_univariate(
                        intensity=lambda t: node.get_conditional_intensity(t, {node.parent: (cause_t, cause_id)}),
                        upper_bound=node.upper_bound,
                        n_events=1,
                        init_t=t,
                        max_t=T,
                    )
                    if ts:
                        temp_P.append((ts[0], node.idx, cause_id))


                # if node.parent and node.parent in most_recent:
                #     ts = ogata_thinning_univariate(
                #         intensity=lambda t: node.get_conditional_intensity(t, most_recent),
                #         upper_bound=node.upper_bound,
                #         n_events=1,
                #         init_t=t,
                #         max_t=T,
                #     )
                #     if ts:
                #         temp_P.append((ts[0], node.idx, most_recent[node.parent][1]))

            if temp_P:
                t, idx, cause = min(temp_P, key=lambda x: x[0])
                if t < T:
                    events.append((t, idx, cause))
                    most_recent[idx].append((t, len(events)-1))
            else:
                break

        return events