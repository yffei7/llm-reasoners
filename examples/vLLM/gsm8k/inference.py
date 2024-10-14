from typing import Type, Callable, Optional, Literal

import numpy as np

# from reasoners.benchmark import GSM8KEvaluator

from reasoners import LanguageModel, Reasoner, SearchAlgorithm
from reasoners.algorithm import MCTS, MCTSNode, MCTSAggregation

from world_model import GSM8kWorldModel, GSM8kPromptDict
from search_config import GSM8kConfig, GSM8kUsefulPrompt
import utils
from reasoners.lm.vllm import VllmModel
from gsm8k_custom import GSM8KCustomEvaluator
from mcts_custom import MCTSCustom


def node_visualizer(x: MCTSNode):
    if not x.state:
        return {}
    return {"question": x.state[-1].sub_question, "answer": x.state[-1].sub_answer}

def rap_gsm8k(base_model: LanguageModel,
              prompt: GSM8kPromptDict,
              useful_prompt: GSM8kUsefulPrompt,
              search_algo: Type[SearchAlgorithm] = MCTSCustom,
              resume: int = 0,
              n_action: int = 4,
              n_confidence: int = 8,
              depth_limit: int = 5,
              force_terminating_on_depth_limit: bool = True,
              batch_size: int = 2,
              temperature: float = 0.8,
              early_stop_base: int = 2,
              early_stop_threshold: float = 0.5,
              reward_alpha: float = 0.5,
              reward_confidence_default: float = 0.8,
              cum_reward: Callable[[list[float]], float] = np.mean,
              calc_q: Callable[[list[float]], float] = max,
              log_dir: Optional[str] = None,
              disable_log: bool = False,
              disable_tqdm: bool = False,
              output_trace_in_each_iter: bool = True,
              aggregate: bool = True,
              **search_algo_params):

    if aggregate:
        aggregator = MCTSAggregation(utils.retrieve_answer, weight_policy='edge')
    else:
        aggregator = None

    search_algo_params |= {'cum_reward': cum_reward, 'calc_q': calc_q, 'disable_tqdm': disable_tqdm,
                           'output_trace_in_each_iter': output_trace_in_each_iter,
                           'node_visualizer': node_visualizer, 'aggregator': aggregator, "simulate_strategy":'random'}
    world_model = GSM8kWorldModel(base_model=base_model,
                                  n_confidence=n_confidence, batch_size=batch_size, temperature=temperature,
                                  early_stop_base=early_stop_base, early_stop_threshold=early_stop_threshold)
    config = GSM8kConfig(base_model=base_model, useful_prompt=useful_prompt,
                         n_actions=n_action, batch_size=batch_size, temperature=temperature,
                         reward_alpha=reward_alpha, reward_confidence_default=reward_confidence_default,
                         force_terminating_on_depth_limit=force_terminating_on_depth_limit, depth_limit=depth_limit,
                         output_extractor=utils.retrieve_answer,
                         answer_extractor=utils.retrieve_answer_from_dataset)
    search_algo = search_algo(**search_algo_params)
    reasoner = Reasoner(world_model=world_model, search_config=config, search_algo=search_algo)

    evaluator = GSM8KCustomEvaluator(output_extractor=utils.retrieve_answer,
                               answer_extractor=utils.retrieve_answer_from_dataset,
                               init_prompt=prompt,
                               sample_prompt_type="rap",
                               disable_log=disable_log,
                               disable_tqdm=disable_tqdm)

    accuracy = evaluator.evaluate(reasoner, num_shot=4, resume=resume, log_dir=log_dir)
    print(accuracy)

if __name__ == '__main__':
    import os
    import sys
    import json
    import warnings
    import fire

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if local_rank != 0:
        sys.stdout = open(os.devnull, 'w')
        warnings.filterwarnings('ignore')

    def main(base_lm: str = '/dataset/crosspipe/OriginModel/Llama-3-8B',
             batch_size: int = 1,
             useful_prompt: str = 'examples/RAP/gsm8k/prompts/useful_examples.json',
             prompt: str = 'examples/RAP/gsm8k/prompts/prompt_pool.json',
             disable_log: bool = False,
             disable_tqdm: bool = False,
             **kwargs):
      
        with open(useful_prompt) as f:
            useful_prompt = json.load(f)
        with open(prompt) as f:
            prompt = json.load(f)
        
        base_model = VllmModel(model=base_lm, additional_prompt="CONTINUE")

        rap_gsm8k(base_model=base_model,
                  useful_prompt=useful_prompt,
                  prompt=prompt,
                  batch_size=batch_size,
                  disable_log=disable_log or local_rank != 0,
                  disable_tqdm=disable_tqdm or local_rank != 0,
                  **kwargs)


    fire.Fire(main)
