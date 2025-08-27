from comet import download_model, load_from_checkpoint
from sacrebleu.metrics import BLEU, CHRF, TER

try:
    # 跑./evaluation/的文件的时候用这个
    from scores import LLM_eval
except ModuleNotFoundError:
    # 直接跑这个文件的时候用这个
    import LLM_eval

class multi_scores:
    def __init__(self, source_lang="en", target_lang="zh", domain="starcraft 2") -> None:
        # 22 is not supported by comet
        
        # self.comet_model = load_from_checkpoint(download_model("Unbabel/wmt20-comet-da"))
        self.comet_model = load_from_checkpoint(download_model("Unbabel/wmt22-comet-da"))
        
        # Initialize directional COMET (dCOMET) model
        # The model used by DELTA： https://unbabel-experimental-models.s3.amazonaws.com/comet/wmt21/wmt21-comet-qe-mqm. tar.gz
        self.dcomet_model = load_from_checkpoint(download_model("NataliaKhaidanova/wmt21-comet-qe-mqm"))
        # self.dcomet_model = load_from_checkpoint(download_model("Unbabel/wmt21-comet-qe-da"))
        
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
    def calculate_comet_llm(self, src:str, mt:str, ref:str, scomet_evaluated:bool,dcomet_evaluated:bool, llm_evaluated:bool, bleu_evaluated:bool) -> dict:
        # preprocess the input
        src, mt, ref = self.__preprocess(src, mt, ref)
        # 这里将batch_size设置为1，gpus设置为0，表示使用CPU进行预测
        # batch_size=1意味着每次只处理一个样本，这可能会影响处理速度，但对于单个样本评估是合适的
        # gpus=0表示不使用GPU加速，而是使用CPU进行计算，这会影响计算速度，但在没有GPU的环境中是必要的
        
        self.llm_evaluated = llm_evaluated
        self.bleu_evaluated = bleu_evaluated
        self.scomet_evaluated = scomet_evaluated
        self.dcomet_evaluated = dcomet_evaluated
        
        comet_output = self.comet_model.predict([{"src":src, "mt":mt, "ref":ref}], batch_size=1, gpus=0)
        comet_score = comet_output.scores[0]
        
        # Get directional COMET score (reference-free evaluation)
        dcomet_output = self.dcomet_model.predict([{"src":src, "mt":mt}], batch_size=1, gpus=0)
        # dcomet_output = self.dcomet_model.predict([{"src":src, "mt":mt , "ref":ref}], batch_size=1, gpus=0)
        dcomet_score = dcomet_output.scores[0]
        
        # print(f"\n\n\n\n---------------------\n{comet_output}\n----------------------\n\n\n\n\n")
        
        # bleu_score = self.bleu_model.corpus_score([mt], [ref]).score
        llm_acc, llm_completeness = LLM_eval.evaluate_prediction(src, ref, mt, self.LLM_model)
        return {
            'comet_score': comet_score, 
            'dcomet_score': dcomet_score,
            'llm_score': llm_acc[0], 
            'llm_explanation': llm_acc[1]
        }
        
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
        dcomet_score = self.dcomet_model.predict([{"src":src, "mt":mt}], batch_size=8, gpus=0).scores[0]
        bleu_score = self.bleu_model.corpus_score([mt], [[ref]]).score
        llm_acc, llm_completeness = LLM_eval.evaluate_prediction(src, ref, mt, self.LLM_model)
        return {
            'bleu_score': bleu_score,
            'comet_score': comet_score,
            'dcomet_score': dcomet_score,
            'llm_score': llm_acc[0],
            'llm_explanation': llm_acc[1],
            'llm_completeness': llm_completeness[0],
            'llm_completeness_explanation': llm_completeness[1]
        }

    def calculate_comet_llm_batch(self, srcs: list, mts: list, refs: list, csv_path="./evaluation/test_data/result.csv") -> list:
        """
        Batch evaluate multiple translation results by processing them individually
        
        Parameters:
            srcs: List of source texts
            mts: List of machine translations
            refs: List of reference translations
            csv_path: Path to save results to CSV
        
        Returns:
            List of dictionaries containing scores for each translation
        """
        
        # 删除重复的 results 初始化
        results = []
        
        # for testing
        # result = {
        # 'comet_score': "111",
        # 'llm_score': "222" , 
        # 'llm_explanation': "333"
        # }
        # results.append(result)
        # return results
        
        
        # Ensure input lists have the same length
        assert len(srcs) == len(mts) == len(refs), "Input lists must have the same length"
        
        # 用于跟踪上次保存的索引位置
        last_saved_index = -1
        
        # Process each item individually to avoid COMET batch issue
        for i, (src, mt, ref) in enumerate(zip(srcs, mts, refs)):
            # Preprocess inputs
            src = src.strip()
            mt = mt.strip()
            ref = ref.strip()
            
            
            # bleu_score = multi_scores().calculate_bleu(mt, [ref])
            
            # Get COMET score individually
            comet_score = self.comet_model.predict([{"src": src, "mt": mt, "ref": ref}], batch_size=1, gpus=0).scores[0]
            
            # Get directional COMET score (no reference needed)
            dcomet_score = self.dcomet_model.predict([{"src": src, "mt": mt}], batch_size=1, gpus=0).scores[0]
            
            result = {
                # 'bleu_score': bleu_score,
                'comet_score': comet_score,
                'dcomet_score': dcomet_score,
                'llm_score': "" , 
                'llm_explanation': ""
            }
            
            # Get LLM evaluation
            # llm_acc, llm_completeness = LLM_eval.evaluate_prediction(src, ref, mt, self.LLM_model)
            
            # # Store results
            # result = {
            #     # 'comet_score': comet_score,
            #     'llm_score': llm_acc[0], 
            #     'llm_explanation': llm_acc[1]
            # }
            
            results.append(result)
            
            # Save to CSV every 10 items - 只保存新处理的部分
            if (i + 1) % 10 == 0:
                # 只保存新增的结果，而不是全部重新保存
                new_results = results[last_saved_index+1:i+1]
                new_srcs = srcs[last_saved_index+1:i+1]
                new_mts = mts[last_saved_index+1:i+1]
                new_refs = refs[last_saved_index+1:i+1]
                
                self._save_batch_to_csv(new_srcs, new_mts, new_refs, new_results, csv_path)
                print(f"Saved results {last_saved_index+1} to {i+1} to {csv_path}")
                last_saved_index = i
        
        # Save any remaining results
        if last_saved_index < len(results) - 1:
            new_results = results[last_saved_index+1:]
            new_srcs = srcs[last_saved_index+1:]
            new_mts = mts[last_saved_index+1:]
            new_refs = refs[last_saved_index+1:]
            
            self._save_batch_to_csv(new_srcs, new_mts, new_refs, new_results, csv_path)
            print(f"Saved remaining results {last_saved_index+1} to {len(results)} to {csv_path}")
            
        return results
    
    def _save_batch_to_csv(self, srcs, mts, refs, results, csv_path):
        """Helper method to save batch results to CSV"""
        import csv
        import os
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        
        # Check if file exists to determine if we need to write headers
        file_exists = os.path.isfile(csv_path)
        
        with open(csv_path, "a", encoding="utf-8", newline='') as f:
            csv_writer = csv.writer(f)
            
            # Write headers if file doesn't exist
            if not file_exists:
                csv_writer.writerow(["Source", "MT", "Reference", "COMET", "dCOMET", "LLM Score", "LLM Explanation"])
            
            # Write results
            for i, result in enumerate(results):
                row = [
                    srcs[i],                          # Source
                    mts[i],                           # MT
                    refs[i],                          # Reference
                    result.get("bleu_score", ""),     # BLEU
                    result.get("comet_score", ""),    # COMET
                    result.get("dcomet_score", ""),   # dCOMET
                    result.get("llm_score", ""),      # LLM Score
                    result.get("llm_explanation", "") # LLM Explanation
                ]
                csv_writer.writerow(row)

def cal_all_scores(src_list, mt_list, ref_list, csv_path="./evaluation/test_data/result.csv"):
    """Calculate all scores for a list of src, mt, and ref."""
    import csv
    import json
    
    # Set up the CSV file with headers
    # with open("result.csv", "w", encoding="utf-8") as f:
    #     csv_writer = csv.writer(f)
    #     csv_writer.writerow(["Source", "MT", "Reference", "BLEU", "COMET", "dCOMET", "LLM Score", "LLM Explanation"])
    
    # Calculate BLEU score
    
    # TODO 已经处理完了BLEU，因此注释
    # 在这个地方bleu只会处理一整个batch，不是单句的，但是貌似一个list就可以包括一个batch的所有单句了
    bleu_score = multi_scores().calculate_bleu(mt_list, [ref_list])
    
    # # 将bleu_score写入csv文件
    with open("./evaluation/test_data/bleu_result.csv", "a", encoding="utf-8") as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(["BLEU", bleu_score])
    print(f"BLEU: {bleu_score}")
    
    # Calculate scores for each example and write to CSV
    results = multi_scores().calculate_comet_llm_batch(src_list, mt_list, ref_list, csv_path=csv_path)
    for i, result in enumerate(results):
        print(result)
        
    return results  # 返回结果以便其他函数可以使用

if __name__ == "__main__":
    src = "The South Korea player is encountering with the Blue Terran's SCV"
    src_list = [src,"Hello,  I think BLEU is a very bad evaluation metric.","Second, I recommend greasing the groove, which means several times a day hanging on the bar for about 50 percent of your max hold time. It's all about doing submaximal sets. You want to practice frequently while keeping yourself feeling as fresh as possible. Grease the groove every single day."]
    mt = "位于对角线的另一个角落  使用蓝色的Terran Probes"
    mt_list = [mt,"我认为BLEU是一个很好的东西。","其次,我建议使用润滑槽方法,这意味着每天多次在单杠上悬挂,时间约为你最大悬挂时间的50%,这主要是进行低于最大强度的训练,你需要频繁练习,同时尽量保持身体的清新感,每天都要进行润滑槽训练."]
    ref = " 来自南韩的玩家遇到了来自蓝色人族的SCV"
    ref_list = [ref,"你好，我认为BLEU是一个非常糟糕的评价指标。","第二，我推荐磨合训练法，单杠训练一天多次，锻炼时间保持在你最长记录的50%，也就是做次强度训练。勤加练习，同时保持精力充沛。每天如是磨合训练。"]
    
    cal_all_scores(src_list, mt_list, ref_list, csv_path="./evaluation/test_data/gemini_result.csv")
    
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
        
