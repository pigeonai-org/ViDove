import os
import sys

import torch
from comet import download_model, load_from_checkpoint
from sacrebleu.metrics import BLEU
from tqdm import tqdm

from scores.SubER_main.suber.file_readers import read_input_file
from scores.SubER_main.suber.metrics.suber import calculate_SubER
from scores.subsonar.src.subsonar.sonar_metric import SonarAudioTextMetric
from scores.subsonar.src.subsonar.srt_reader import SrtReader

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


# ===== COMET scoring =====
def COMETscore(src, mt, ref):
    data = []
    for i in range(
        len(src)
    ):  # fixed small mistake: should be "for i in range", not enumerate(src)
        data.append({"src": src[i], "mt": mt[i], "ref": ref[i]})
    model_path = download_model("Unbabel/wmt22-comet-da")
    model = load_from_checkpoint(model_path)
    model_output = model.predict(data, batch_size=8, gpus=0)
    return model_output


# ===== BLEU scoring =====
def BLEUscore(sys, refs, lang="zh"):
    # Use Chinese tokenizer for better Chinese text processing
    bleu = BLEU(tokenize=lang)
    return bleu.corpus_score(sys, refs)


# ===== SubERscore =====
# code adapted from SubER-Main (https://github.com/apptek/SubER).
# Thanks to the original authors for their contributions.
def SubERscore(hypothesis_file: str, reference_file: str) -> float:
    hypo_segments = read_input_file(hypothesis_file, file_format="SRT")
    ref_segments = read_input_file(reference_file, file_format="SRT")

    score = calculate_SubER(hypo_segments, ref_segments)

    return score


# ===== SubSONAR score =====
# code adapted from SubSONAR (https://github.com/apptek/SubSONAR).
# Thanks to the original authors for their contributions.
def SubSONARscore(
    hypothesis_file: str, audio_file: str, audio_lang: str, text_lang: str
) -> float:
    """
    Calculate SubSONAR score between hypothesis and reference SRT files.
    Args:
        hypothesis_file: Path to hypothesis SRT file
        audio_file: Path to the audio file corresponding to the SRT
        audio_lang: Language of the speech in the audio file (e.g. 'eng')
        text_lang: Language of the text in SRT file in Flores 200 format (e.g. 'eng_Latn')

    Returns:
        float: SubSONAR score
    """

    # Initialize device and metric
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    metric = SonarAudioTextMetric(audio_lang, text_lang, device=device)

    # Process the SRT file with audio
    srt_reader = SrtReader(audio_file, hypothesis_file)
    scores = []
    batch = []
    batch_size = 10

    # Process blocks in batches
    for block in tqdm(srt_reader, desc="Processing SubSONAR blocks"):
        batch.append(block)
        if len(batch) >= batch_size:
            scores.extend(
                metric.batch_score([b.text for b in batch], [b.audio for b in batch])
            )
            batch = []

    # Process remaining blocks
    if len(batch) > 0:
        scores.extend(
            metric.batch_score([b.text for b in batch], [b.audio for b in batch])
        )

    # Calculate overall score
    score = metric.merge_scores(scores)
    return score
