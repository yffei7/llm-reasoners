from typing import Generic, Optional, NamedTuple, Callable, Hashable

import numpy as np
from tqdm import trange

from reasoners.algorithm import MCTS, MCTSNode, MCTSAggregation


class MCTSCustom(MCTS):
    def __init__(self,
                 output_trace_in_each_iter: bool = False,
                 w_exp: float = 1.,
                 depth_limit: int = 5,
                 n_iters: int = 10,
                 cum_reward: Callable[[list[float]], float] = sum,
                 calc_q: Callable[[list[float]], float] = np.mean,
                 simulate_strategy: str | Callable[[list[float]], int] = 'max',
                 output_strategy: str = 'max_reward',
                 uct_with_fast_reward: bool = True,
                 aggregator: Optional[MCTSAggregation] = None,
                 disable_tqdm: bool = True,
                 node_visualizer: Callable[[MCTSNode], dict] = lambda x: x.__dict__):
        super().__init__(output_trace_in_each_iter, w_exp, depth_limit, n_iters, cum_reward, calc_q, simulate_strategy, output_strategy, uct_with_fast_reward, aggregator, disable_tqdm, node_visualizer)

    def _expand(self, node: MCTSNode):
        if node.state is None:
            node.state, aux = self.world_model.step(node.parent.state, node.action)
            node.reward, node.reward_details = self.search_config. \
                reward(node.state, node.action, **node.fast_reward_details, **aux)
            node.is_terminal = self.world_model.is_terminal(node.state)

        if node.is_terminal:
            return

        children = []
        actions = self.search_config.get_actions(node.state)
        for action in actions:
            fast_reward, fast_reward_details = self.search_config.fast_reward(node.state, action)
            child = MCTSNode(state=None, action=action, parent=node,
                             fast_reward=fast_reward, fast_reward_details=fast_reward_details, calc_q=self.calc_q)
            children.append(child)

        node.children = children
