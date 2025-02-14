from sacrebleu.metrics import BLEU
from .abs_evaluator import AbsApiEvaluator
from logging import Logger


class ScarBLEUEvaluator(AbsApiEvaluator):
    def __init__(self, logger: Logger, tokenize: str = "zh") -> None:
        super().__init__()
        self.logger = logger
        self.tokenize = tokenize
        self.bleu_model = BLEU(tokenize=tokenize)
        self.is_loaded = False

    def load_model(self) -> None:
        self.is_loaded = True

    def evaluate(self, input: dict) -> str:
        if not self.is_loaded:
            self.load_model()

        bleu_score = self.bleu_model.corpus_score(input['mt'], input['ref']).score
        return bleu_score