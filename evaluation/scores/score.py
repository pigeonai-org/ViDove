from comet import download_model, load_from_checkpoint
from sacrebleu.metrics import BLEU, CHRF, TER
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


# ===== COMET scoring =====
def COMETscore(src, mt, ref):
    data = []
    for i in range(len(src)):  # fixed small mistake: should be "for i in range", not enumerate(src)
        data.append({"src": src[i], "mt": mt[i], "ref": ref[i]})
    model_path = download_model("Unbabel/wmt22-comet-da")
    model = load_from_checkpoint(model_path)
    model_output = model.predict(data, batch_size=8, gpus=0)
    return model_output

# ===== BLEU scoring =====
def BLEUscore(sys, refs):
    bleu = BLEU()
    return bleu.corpus_score(sys, refs)

# ===== SubERscore =====
# code adapted from SubER-Main (https://github.com/apptek/SubER).
# Thanks to the original authors for their contributions.
def SubERscore(hypothesis_file: str, reference_file: str) -> float:
    
    from scores.SubER_main.suber.metrics.suber import calculate_SubER
    from scores.SubER_main.suber.file_readers import read_input_file


    hypo_segments = read_input_file(hypothesis_file, file_format="SRT")
    ref_segments = read_input_file(reference_file, file_format="SRT")

    score = calculate_SubER(hypo_segments, ref_segments)

    return score

