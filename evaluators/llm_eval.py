from logging import Logger

import openai
from pydantic import BaseModel

from .abs_evaluator import AbsApiEvaluator

# Simple LLM evaluator based on OpenAI's Models


class evaluation_result(BaseModel):
    Accuracy: int
    Completeness: int
    Explanation: str


EVALUATION_PROMPT = """
You are grading the translation based on following input:
{input}
if the input is "", that means there is no input sentence.
you should grade the translation based on the reference translation:
Here is the real answer(reference):
{ref}
You are grading the following translation:
{mtg}
based on the following criteria: 
Accuracy, Completeness
Give two grades, accuracy and completeness rate them from a scale of 0 to 100, where 0 is the lowest (very low accuracy/completeness) and 100 is the highest (very high accuracy/completeness)? 
Give explanations for your scores in any cases. However punish the scores for answers that are 
numerically incorrect this also includes values that have the $ in front
You should return one json object WITHOUT any markdown or format text with the following format:
{
    "accuracy": YOUR_ACCURACY_SCORE,
    "completeness": YOUR_COMPLETENESS_SCORE,
    "explanation": "Your explanation here"
}
"""

EVALUATION_EXAMPLE_INPUT = {
    "input": "His opponent in the opposite corner with the red protoss pieces widely considered to be the strongest defensive player on the planet",
    "reference": "在对面角落使用红色神族的对手是公认的世界上最强的防守型玩家。",
    "output": "他对角位置的对手使用红色星灵棋子，被广泛认为是地球上最强的防守型选手。",
    "criteria": "Accuracy, Completeness",
}

EVALUATION_EXAMPLE_OUTPUT = {
    "accuracy": 100,
    "completeness": 100,
    "explanation": "The translation is good and complete but some words have better synonyms.",
}


class LLM_Evaluator(AbsApiEvaluator):
    def __init__(
        self, logger: Logger, model_name: str = "gpt-4o", batch_size: int = 8
    ) -> None:
        super().__init__()
        self.model_name = model_name
        self.logger = logger
        self.batch_size = batch_size
        self.model_name = model_name

    def load_model(self) -> None:
        pass

    def evaluate(self, input: list[dict]) -> list:
        evaluation_results = []

        for i in input:
            if "reference" not in i or "output" not in i:
                raise ValueError(
                    "The input should contain 'reference', and 'output' keys"
                )

            prompt = EVALUATION_PROMPT.format(
                input=i["input"], ref=i["reference"], mtg=i["output"]
            )
            response = openai.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert in translation and your task is to evaluate the following translation.",
                    },
                    {"role": "user", "content": prompt},
                    {
                        "role": "system",
                        "content": "Here is one example of the evaluation:",
                    },
                    {"role": "user", "content": EVALUATION_EXAMPLE_INPUT},
                    {"role": "system", "content": "Here is the evaluation result:"},
                    {"role": "assistant", "content": EVALUATION_EXAMPLE_OUTPUT},
                    {"role": "user", "content": "Please give your evaluation result:"},
                ],
                response_format={
                    "name": "evaluation_result",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "accuracy": {
                                "type": "number",
                                "description": "The accuracy of the evaluation, represented as an int between 0-100 (the higher the better).",
                            },
                            "completeness": {
                                "type": "number",
                                "description": "The completeness of the evaluation, represented as an int between 0-100 (the higher the better).",
                            },
                            "explanation": {
                                "type": "string",
                                "description": "A detailed explanation of the evaluation results.",
                            },
                        },
                        "required": ["accuracy", "completeness", "explanation"],
                        "additionalProperties": False,
                    },
                    "strict": True,
                },
            )

            evaluation_results.append(response.choices[0].message.parsed)

        return evaluation_results
        # return model_output
