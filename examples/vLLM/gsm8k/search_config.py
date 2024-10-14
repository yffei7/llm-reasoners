import io
import re
from typing import TypedDict, Optional
import numpy as np

from world_model import GSM8kState, GSM8kAction, GSM8kPromptDict
from reasoners import SearchConfig, LanguageModel
from reasoners.base import Example

class GSM8kUsefulPrompt(TypedDict):
    input: str
    question_prefix: str
    subquestion_prefix: str
    new_subquestion_prefix: str
    useful_prefix: str

class GSM8kConfig(SearchConfig):
    def __init__(self,
                 base_model: LanguageModel,
                 useful_prompt: GSM8kUsefulPrompt,
                 n_actions=4,
                 batch_size=1,
                 temperature=0.8,
                 top_k=50,
                 top_p=0.95,
                 reward_alpha=0.5,
                 reward_confidence_default=0.8,
                 depth_limit=5,
                 force_terminating_on_depth_limit=True,
                 force_overall_prompt_on_overall_question=True,
                 force_overall_question_on_overall_prompt=True,
                 output_extractor=None,
                 answer_extractor=None) -> None:
        super().__init__()
        self.base_model = base_model
        self.useful_prompt = useful_prompt
        self.example = ''
        self.batch_size = batch_size
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.n_actions = n_actions
        self.force_terminating_on_depth_limit = force_terminating_on_depth_limit
        self.depth_limit = depth_limit
        self.reward_alpha = reward_alpha
        self.reward_confidence_default = reward_confidence_default
        self.force_overall_prompt_on_overall_question = force_overall_prompt_on_overall_question
        self.force_overall_question_on_overall_prompt = force_overall_question_on_overall_prompt
        self.overall_question: Optional[str] = None
        self.prompt_examples = ""
        self.n_shots = 0
        self.answer = ""
        self.output_extractor = output_extractor
        self.answer_extractor = answer_extractor

    def update_example(self, example: Example, prompt: GSM8kPromptDict = None) -> None:
        super().update_example(example["question"], prompt)
        self.answer = example["answer"]

        assert prompt is not None
        self.prompt = prompt
        with io.StringIO() as f:
            f.write(self.prompt['instruction'] + '\n\n')
            for idx, example in enumerate(self.prompt['interactive_examples']):
                f.write(example.format(idx=idx + 1) + '\n\n')
            self.n_shots = len(self.prompt['interactive_examples'])
            self.prompt_examples = f.getvalue()

        if self.force_overall_prompt_on_overall_question or self.force_overall_question_on_overall_prompt:
            print(self.example)
            # self.overall_question = re.match('.*((Calculate|calculate|how|How|what|What|Find|find|True or false).*)$',
            #                                  self.example, flags=re.DOTALL)[1]
            self.overall_question = re.match('.*((([A-Z].* (calculate|how|what|find|true or false))|((Calculate|How|What|Find|True or false))).*)$', self.example, flags=re.DOTALL)[1]

    def get_actions(self, state: GSM8kState, ) -> list[GSM8kAction]:
        with io.StringIO() as f:
            f.write(self.prompt_examples)
            f.write(self.prompt["question_prefix"].format(idx=self.n_shots + 1, question=self.example) + "\n")
            for idx, (q, a, _) in enumerate(state):
                f.write(
                    self.prompt["subquestion_prefix"].format(idx=self.n_shots + 1, sub_idx=idx + 1) + " " + q + "\n")
                f.write(self.prompt["answer_prefix"].format(idx=self.n_shots + 1, sub_idx=idx + 1) + " " + a + "\n")
            # f.write(self.prompt["subquestion_prefix"].format(idx=self.n_shots + 1, sub_idx=len(state) + 1))
            if at_depth_limit := self.force_terminating_on_depth_limit and len(state) + 1 >= self.depth_limit:
                f.write(" " + self.prompt["overall_question_prefix"])
            model_input = f.getvalue()

        n_actions = 1 if at_depth_limit else self.n_actions
        temperature = 0 if at_depth_limit else self.temperature
        outputs = []
        for idx in range(0, n_actions, self.batch_size):
            n_samples = min(n_actions - idx, self.batch_size)
            outputs += self.base_model.generate([model_input] * n_samples,
                                                hide_input=True,
                                                do_sample=True,
                                                temperature=temperature,
                                                top_k=self.top_k,
                                                top_p=self.top_p,
                                                eos_token_id='\n').text

        outputs = [output.strip() for output in outputs]
        if at_depth_limit:
            outputs = [self.prompt["overall_question_prefix"] + ' ' + output for output in outputs]
        if self.force_overall_question_on_overall_prompt:
            for i, output in enumerate(outputs):
                if self.prompt["overall_question_prefix"] in output:
                    outputs[i] = self.prompt["overall_question_prefix"] + ' ' + self.overall_question
        if self.force_overall_prompt_on_overall_question:
            for i, output in enumerate(outputs):
                if self.overall_question.lower() == output.lower():
                    outputs[i] = self.prompt["overall_question_prefix"] + ' ' + self.overall_question

        outputs = list(dict.fromkeys(outputs))
        return outputs

    def fast_reward(self, state: GSM8kState, action: GSM8kAction) -> tuple[float, dict]:
        return 0, {'r_useful': 0}

    def calculate_reward(self, r_useful, r_conf=None):
        if r_conf is None:
            r_conf = self.reward_confidence_default
        return r_useful ** self.reward_alpha * r_conf ** (1 - self.reward_alpha), {'r_useful': r_useful,
                                                                                   'r_conf': r_conf}

    def reward(self, state: GSM8kState, action: GSM8kAction,
               r_useful: float = None,
               confidence: float = None) -> tuple[float, dict]:
        if len(state) > 0 and "Now we can answer" in state[-1].sub_question:
            output = self.output_extractor(action)
            answer = self.answer_extractor(self.answer)
            if output is None or answer is None:
                return self.calculate_reward(0)
            return self.calculate_reward(int(output == answer))
        else:
            return self.calculate_reward(0)
