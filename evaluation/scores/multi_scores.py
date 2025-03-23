from comet import download_model, load_from_checkpoint
from sacrebleu.metrics import BLEU, CHRF, TER
# from scores import LLM_eval
import LLM_eval

class multi_scores:
    def __init__(self, source_lang="en", target_lang="zh", domain="starcraft 2") -> None:
        # 22 is not supported by comet
        # self.comet_model = load_from_checkpoint(download_model("Unbabel/wmt22-comet-da"))
        self.comet_model = load_from_checkpoint(download_model("Unbabel/wmt20-comet-da"))
        self.bleu_model = BLEU(tokenize=target_lang)
        self.LLM_model = LLM_eval.init_evaluator(source_lang=source_lang, target_lang=target_lang, domain=domain)
        # self.score = {}

    def __preprocess(self, src:str, mt:str, ref:str) -> dict:
        # remove the space in the beginning and end of the sentence\
        src = src.strip()
        mt = mt.strip()
        ref = ref.strip()
        print(src, mt, ref)
        return {'src':src, 'mt':mt, 'ref':ref}



    # The function to get the scores
    # src: orginal sentence
    # mt: machine translation
    # ref: reference translation
    def calculate_comet_llm(self, src:str, mt:str, ref:str) -> dict:
        # preprocess the input
        src, mt, ref = self.__preprocess(src, mt, ref)
        comet_score = self.comet_model.predict([{"src":src, "mt":mt, "ref":ref}], batch_size=8, gpus=0).scores[0]
        # bleu_score = self.bleu_model.corpus_score([mt], [ref]).score
        llm_acc, llm_completeness = LLM_eval.evaluate_prediction(src, ref, mt, self.LLM_model)
        return {'comet_score':comet_score, 'llm_score':llm_acc[0], 'llm_explanation': llm_acc[1]}
        # self.score['bleu_score'] = bleu_score
        # self.score['comet_score'] = comet_score
        # self.score['llm_score'] = llm_score
        # self.score['llm_explanation'] = llm_explanation
        
    def calculate_bleu(self, mts:list, refs:list) -> dict:
        # src, mt, ref = self.__preprocess(src, mt, ref)
        # remove the space in the beginning and end of the sentence for each sentence
        # mts = [mt.strip() for mt in mts]
        # refs = [ref.strip() for ref in refs]
        # print(mts, refs)
        # mt and ref are list of sentences
        bleu_score = self.bleu_model.corpus_score(mts, refs).score
        return {'bleu_score':bleu_score}
    
    def get_scores(self, src:str, mt:str, ref:str) -> dict:
        comet_score = self.comet_model.predict([{"src":src, "mt":mt, "ref":ref}], batch_size=8, gpus=0).scores[0]
        bleu_score = self.bleu_model.corpus_score([mt], [[ref]]).score
        llm_acc, llm_completeness = LLM_eval.evaluate_prediction(src, ref, mt, self.LLM_model)
        return {'bleu_score':bleu_score ,'comet_score':comet_score, 'llm_score':llm_acc[0], 'llm_explanation': llm_acc[1], 'llm_completeness':llm_completeness[0], 'llm_completeness_explanation':llm_completeness[1]}

    def calculate_comet_llm_batch(self, srcs: list, mts: list, refs: list) -> list:
        """
        Batch evaluate multiple translation results by processing them individually
        
        Parameters:
            srcs: List of source texts
            mts: List of machine translations
            refs: List of reference translations
        
        Returns:
            List of dictionaries containing scores for each translation
        """
        # Ensure input lists have the same length
        assert len(srcs) == len(mts) == len(refs), "Input lists must have the same length"
        
        results = []
        # Process each item individually to avoid COMET batch issue
        for src, mt, ref in zip(srcs, mts, refs):
            # Preprocess inputs
            src = src.strip()
            mt = mt.strip()
            ref = ref.strip()
            
            # Get COMET score individually
            comet_score = self.comet_model.predict([{"src": src, "mt": mt, "ref": ref}], batch_size=1, gpus=0).scores[0]
            
            # Get LLM evaluation
            llm_acc, llm_completeness = LLM_eval.evaluate_prediction(src, ref, mt, self.LLM_model)
            
            # Store results
            results.append({
                'comet_score': comet_score,
                'llm_score': llm_acc[0], 
                'llm_explanation': llm_acc[1]
            })
        
        return results

if __name__ == "__main__":
    src = "South Korea playing with the Blue Proto's Probes"
    src_list = [src,"你好"]
    mt = "位于对角线的另一个角落  使用蓝色的Proto's Probes"
    mt_list = [mt,"世界"]
    ref = " 在对角落里使用蓝色神族探机 他的名字是..."
    ref_list = [ref,"世界"]
    # print(multi_scores().get_scores(src, mt, ref))
    # print(multi_scores().calculate_comet_llm(src, mt, ref))
    # print(multi_scores().calculate_bleu([mt], [[ref]]))
    print("start")
    print(multi_scores().calculate_comet_llm_batch(src_list, mt_list, ref_list))
    print("end")
        
