from comet import download_model, load_from_checkpoint
from sacrebleu.metrics import BLEU, CHRF, TER
# from scores import LLM_eval
import LLM_eval

class multi_scores:
    def __init__(self, source_lang="en", target_lang="zh", domain="starcraft 2") -> None:
        # 22 is not supported by comet
        
        # self.comet_model = load_from_checkpoint(download_model("Unbabel/wmt20-comet-da"))
        self.comet_model = load_from_checkpoint(download_model("Unbabel/wmt22-comet-da"))
        
        self.bleu_model = BLEU(tokenize=target_lang)
        # self.bleu_model = BLEU()
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
        comet_output = self.comet_model.predict([{"src":src, "mt":mt, "ref":ref}], batch_size=1, gpus=0)
        comet_score = comet_output.scores[0]
        # comet_score = self.comet_model.predict([{"src":src, "mt":mt, "ref":ref}], batch_size=8, gpus=0).scores[0]
        
        # print(f"\n\n\n\n---------------------\n{comet_output}\n----------------------\n\n\n\n\n")
        
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
        bleu_results = self.bleu_model.corpus_score(mts, refs)
        print(bleu_results)
        bleu_score = bleu_results.score
        return {'bleu_score':bleu_score, 'bleu_results':bleu_results}
    
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

def cal_all_scores(src_list, mt_list, ref_list):
    print("\n\n\n BLUE ------------------------------------:")
    print(multi_scores().calculate_bleu(mt_list, [ref_list]))
    print("\n\n\n COMET ----------------- LLM ------------------------------------:")
    for result in multi_scores().calculate_comet_llm_batch(src_list, mt_list, ref_list):
        print(result)

if __name__ == "__main__":
    src = "The South Korea player is encountering with the Blue Terran's SCV"
    src_list = [src,"Hello, I am a player from China. I think BLEU is a very bad evaluation metric.","Second, I recommend greasing the groove, which means several times a day hanging on the bar for about 50 percent of your max hold time. It's all about doing submaximal sets. You want to practice frequently while keeping yourself feeling as fresh as possible. Grease the groove every single day."]
    mt = "位于对角线的另一个角落  使用蓝色的Terran Probes"
    mt_list = [mt,"你好，我是来自中国的玩家。我认为BLEU是一个非常糟糕的评价指标。","其次,我建议使用“润滑槽”方法,这意味着每天多次在单杠上悬挂,时间约为你最大悬挂时间的50%,这主要是进行低于最大强度的训练,你需要频繁练习,同时尽量保持身体的清新感,每天都要进行“润滑槽”训练."]
    ref = " 来自南韩的玩家遇到了来自蓝色人族的SCV"
    ref_list = [ref,"你好，我是来自中国的玩家。我认为BLEU是一个非常糟糕的评价指标。","第二，我推荐磨合训练法，单杠训练一天多次，锻炼时间保持在你最长记录的50%，也就是做次强度训练。勤加练习，同时保持精力充沛。每天如是磨合训练。"]
    
    cal_all_scores(src_list, mt_list, ref_list)
    
    #  print(multi_scores().get_scores(src, mt, ref))
    # print(multi_scores().calculate_comet_llm(src, mt, ref))
    # print("\n\n\n BLUE ------------------------------------:")
    # bleu_inputs = ["你好世界，我是来自中国的玩家。我认为BLEU是一个非常糟糕的评价指标。", "你好世界，我是来自中国的玩家。", "Hello, I am a player from China. I think BLEU is a very bad evaluation metric."]
    # refs = [ # First set of references
    #       ['The dog bit the man.', 'It was not unexpected.', 'The man bit him first.'],
    #       # Second set of references
    #       ['The dog had bit the man.', 'No one was surprised.', 'The man had bitten the dog.'],
    #     ]
    # sys = ['The dog bit the man.', "It wasn't surprising.", 'The man had just bitten him.']
    # print(multi_scores().calculate_bleu(mt_list, [ref_list]))
    # print("\n\n\n COMET ----------------- LLM ------------------------------------:")
    # for result in multi_scores().calculate_comet_llm_batch(src_list, mt_list, ref_list):
    #     print(result)
    # print(multi_scores().calculate_comet_llm_batch(src_list, mt_list, ref_list))
        
