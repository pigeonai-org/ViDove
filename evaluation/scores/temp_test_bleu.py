from sacrebleu.metrics import BLEU, CHRF, TER

refs = [ # First set of references
          ['The dog bit the man.', 'It was not unexpected.', 'The man bit him first.'],
          # Second set of references
          ['The dog had bit the man.', 'No one was surprised.', 'The man had bitten the dog.'],
        ]
sys = ['The dog bit the man.', "It wasn't surprising.", 'The man had just bitten him.']

bleu = BLEU()

print(bleu.corpus_score(sys, refs))
print(bleu.get_signature())

bleu_inputs = ["你好世界，我认为BLEU是一个非常糟糕的评价指标。", "你好世界，我是好人"]

bleu_model = BLEU(tokenize="zh")
print(bleu_model.corpus_score(bleu_inputs, [bleu_inputs]))
print(bleu_model.get_signature())   


reference = ['我', '是', '好', '人']
hypothesis = ['我', '是', '善良的', '人']

bleu_model = BLEU(tokenize="zh")
print(bleu_model.corpus_score(hypothesis, reference))
print(bleu_model.get_signature())   

import jieba
bleu_input_seg = list(jieba.cut(bleu_inputs[1]))
print(bleu_input_seg)

bleu_model = BLEU(tokenize="zh")
print(bleu_model.corpus_score(bleu_input_seg, [bleu_input_seg]))
print(bleu_model.get_signature())   

