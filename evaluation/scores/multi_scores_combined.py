"""
Combined evaluation module that integrates all metrics
"""

import os
import csv
from pathlib import Path

# Import evaluation modules
from scomet_dcomet_eval import CometEvaluator
from bleu_eval import BleuEvaluator

try:
    # When running from evaluation directory
    from scores import LLM_eval
except ModuleNotFoundError:
    # When running directly
    import LLM_eval

class MultiScores:
    def __init__(self, source_lang="en", target_lang="zh", domain="starcraft 2", 
                 scomet_enabled=True, dcomet_enabled=True, bleu_enabled=True, llm_enabled=True):
        """
        Initialize combined evaluator with all metrics
        
        Args:
            source_lang: Source language code
            target_lang: Target language code
            domain: Domain for LLM evaluation
            scomet_enabled: Whether to enable standard COMET evaluation
            dcomet_enabled: Whether to enable directional COMET evaluation
            bleu_enabled: Whether to enable BLEU evaluation
            llm_enabled: Whether to enable LLM-based evaluation
        """
        # Store evaluation flags
        self.scomet_enabled = scomet_enabled
        self.dcomet_enabled = dcomet_enabled
        self.bleu_enabled = bleu_enabled
        self.llm_enabled = llm_enabled
        
        # Initialize evaluators based on enabled flags
        if scomet_enabled or dcomet_enabled:
            self.comet_evaluator = CometEvaluator()
        
        if bleu_enabled:
            self.bleu_evaluator = BleuEvaluator(tokenize=target_lang)
        
        if llm_enabled:
            self.llm_model = LLM_eval.init_evaluator(source_lang=source_lang, target_lang=target_lang, domain=domain)
        
        # Store language settings
        self.source_lang = source_lang
        self.target_lang = target_lang
        self.domain = domain

    def _preprocess(self, src, mt, ref):
        """
        Preprocess inputs by stripping whitespace
        """
        return src.strip(), mt.strip(), ref.strip()
    
    def evaluate_all(self, src, mt, ref):
        """
        Evaluate all enabled metrics for a single translation
        
        Args:
            src: Source text
            mt: Machine translation
            ref: Reference translation
            
        Returns:
            Dictionary containing evaluation scores for enabled metrics
        """
        # Preprocess inputs
        src, mt, ref = self._preprocess(src, mt, ref)
        
        # Initialize results dictionary
        results = {}
        
        # Calculate COMET scores if enabled
        if self.scomet_enabled:
            results['scomet_score'] = self.comet_evaluator.evaluate_scomet(src, mt, ref)
            
        if self.dcomet_enabled:
            results['dcomet_score'] = self.comet_evaluator.evaluate_dcomet(src, mt)
        
        # Calculate BLEU score if enabled
        if self.bleu_enabled:
            results['bleu_score'] = self.bleu_evaluator.evaluate_sentence(mt, [ref])
        
        # Calculate LLM scores if enabled
        if self.llm_enabled:
            llm_acc, llm_completeness = LLM_eval.evaluate_prediction(src, ref, mt, self.llm_model)
            results['llm_score'] = llm_acc[0]
            results['llm_explanation'] = llm_acc[1]
            results['llm_completeness'] = llm_completeness[0]
            results['llm_completeness_explanation'] = llm_completeness[1]
        
        return results
    
    def evaluate_batch(self, srcs, mts, refs, csv_path="./evaluation/test_data/result.csv", save_every=10):
        """
        Batch evaluate multiple translation results using enabled metrics
        
        Args:
            srcs: List of source texts
            mts: List of machine translations
            refs: List of reference translations
            csv_path: Path to save results CSV
            save_every: Save results to CSV after processing this many items
            
        Returns:
            List of dictionaries containing evaluation results
        """
        # Ensure input lists have the same length
        assert len(srcs) == len(mts) == len(refs), "Input lists must have the same length"
        
        results = []
        last_saved_index = -1
        
        # Process each item individually
        for i, (src, mt, ref) in enumerate(zip(srcs, mts, refs)):
            # Preprocess inputs
            src, mt, ref = self._preprocess(src, mt, ref)
            
            # Initialize result for this item
            result = {}
            
            # Get COMET scores if enabled
            if self.scomet_enabled:
                result['scomet_score'] = self.comet_evaluator.evaluate_scomet(src, mt, ref)
                
            if self.dcomet_enabled:
                result['dcomet_score'] = self.comet_evaluator.evaluate_dcomet(src, mt)
            
            # Get LLM evaluation if enabled
            if self.llm_enabled:
                llm_acc, llm_completeness = LLM_eval.evaluate_prediction(src, ref, mt, self.llm_model)
                result['llm_score'] = llm_acc[0]
                result['llm_explanation'] = llm_acc[1]
            
            results.append(result)
            
            # Save to CSV periodically
            if (i + 1) % save_every == 0:
                # Only save new results
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
        # Create directory if it doesn't exist
        Path(os.path.dirname(csv_path)).mkdir(parents=True, exist_ok=True)
        
        # Check if file exists to determine if we need to write headers
        file_exists = os.path.isfile(csv_path)
        
        with open(csv_path, "a", encoding="utf-8", newline='') as f:
            csv_writer = csv.writer(f)
            
            # Write headers if file doesn't exist
            if not file_exists:
                # Create headers based on enabled metrics
                headers = ["Source", "MT", "Reference"]
                if self.scomet_enabled:
                    headers.append("sCOMET")
                if self.dcomet_enabled:
                    headers.append("dCOMET")
                if self.llm_enabled:
                    headers.extend(["LLM Score", "LLM Explanation"])
                
                csv_writer.writerow(headers)
            
            # Write results
            for i, result in enumerate(results):
                # Start with required fields
                row = [srcs[i], mts[i], refs[i]]
                
                # Add metric results based on what's enabled
                if self.scomet_enabled:
                    row.append(result.get("scomet_score", ""))
                if self.dcomet_enabled:
                    row.append(result.get("dcomet_score", ""))
                if self.llm_enabled:
                    row.append(result.get("llm_score", ""))
                    row.append(result.get("llm_explanation", ""))
                
                csv_writer.writerow(row)

# Modified convenience function for backward compatibility
def cal_all_scores(src_list, mt_list, ref_list, csv_path="./evaluation/test_data/result.csv", 
                   scomet_enabled=True, dcomet_enabled=True, bleu_enabled=True, llm_enabled=False):
    """
    Calculate scores for a list of src, mt, and ref with configurable metrics.
    
    Args:
        src_list: List of source texts
        mt_list: List of machine translations
        ref_list: List of reference translations
        csv_path: Path to save results CSV
        scomet_enabled: Whether to calculate standard COMET scores
        dcomet_enabled: Whether to calculate directional COMET scores
        bleu_enabled: Whether to calculate BLEU scores
        llm_enabled: Whether to calculate LLM-based scores
    
    Returns:
        List of dictionaries containing evaluation results
    """
    import csv
    
    # Create evaluator with specified enabled metrics
    evaluator = MultiScores(
        scomet_enabled=scomet_enabled,
        dcomet_enabled=dcomet_enabled,
        bleu_enabled=bleu_enabled,
        llm_enabled=llm_enabled
    )
    
    # Calculate BLEU score if enabled
    if bleu_enabled:
        bleu_result = evaluator.bleu_evaluator.evaluate_corpus(mt_list, [ref_list])
        
        # Write BLEU score to CSV file
        with open("./evaluation/test_data/bleu_result.csv", "a", encoding="utf-8") as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow(["BLEU", bleu_result['bleu_score']])
        print(f"BLEU: {bleu_result['bleu_score']}")
    
    # Calculate scores for each example and write to CSV
    results = evaluator.evaluate_batch(src_list, mt_list, ref_list, csv_path=csv_path)
    for i, result in enumerate(results):
        print(result)
    
    return results

if __name__ == "__main__":
    # Test examples
    src = "The South Korea player is encountering with the Blue Terran's SCV"
    src_list = [
        src,
        "Hello, I think BLEU is a very bad evaluation metric.",
        "Second, I recommend greasing the groove, which means several times a day hanging on the bar for about 50 percent of your max hold time. It's all about doing submaximal sets. You want to practice frequently while keeping yourself feeling as fresh as possible. Grease the groove every single day."
    ]
    
    mt = "位于对角线的另一个角落  使用蓝色的Terran Probes"
    mt_list = [
        mt,
        "我认为BLEU是一个很好的东西。",
        "其次,我建议使用润滑槽方法,这意味着每天多次在单杠上悬挂,时间约为你最大悬挂时间的50%,这主要是进行低于最大强度的训练,你需要频繁练习,同时尽量保持身体的清新感,每天都要进行润滑槽训练."
    ]
    
    ref = "来自南韩的玩家遇到了来自蓝色人族的SCV"
    ref_list = [
        ref,
        "你好，我认为BLEU是一个非常糟糕的评价指标。",
        "第二，我推荐磨合训练法，单杠训练一天多次，锻炼时间保持在你最长记录的50%，也就是做次强度训练。勤加练习，同时保持精力充沛。每天如是磨合训练。"
    ]
    
    # Example: Only run COMET metrics (sCOMET and dCOMET), skip BLEU and LLM
    cal_all_scores(
        src_list, 
        mt_list, 
        ref_list, 
        csv_path="./evaluation/test_data/gemini_result.csv",
        scomet_enabled=True,
        dcomet_enabled=True,
        bleu_enabled=True,
        llm_enabled=False  # LLM evaluation is more time-consuming, so default to off
    ) 