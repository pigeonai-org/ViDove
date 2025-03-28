from comet import download_model, load_from_checkpoint
from sacrebleu.metrics import BLEU, CHRF, TER

def COMETscore(src, mt, ref):
    data = []
    for i in enumerate(src):
        data.append({"src":src[i], "mt":mt[i], "ref":ref[i]})
    model_path = download_model("Unbabel/wmt22-comet-da")
    model = load_from_checkpoint(model_path)
    model_output = model.predict(data, batch_size = 8, gpus=0)
    return model_output

def BLEUscore(sys, refs):
    bleu = BLEU()
    return bleu.corpus_score(sys, refs)