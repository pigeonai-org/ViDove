import argparse
import pandas as pd
from alignment import alignment
from scores.multi_scores import multi_scores

class Evaluator:
    def __init__(self, pred_path, gt_path, eval_path, res_path):
        self.pred_path = pred_path
        self.gt_path = gt_path
        self.eval_path = eval_path
        self.res_path = res_path

    def eval(self):
        # Align two SRT files
        aligned_srt = alignment(self.pred_path, self.gt_path)

        # Get sentence scores
        scorer = multi_scores()
        result_data = []
        for (pred_s, gt_s) in aligned_srt:
            print("pred_s.source_text: ", pred_s.source_text)
            print("pred_s.translation: ", pred_s.translation)
            print("gt_s.source_text: ", gt_s.source_text)
            print("gt_s.translation: ", gt_s.translation)

            # Check if the gt_s.translation is not empty
            if gt_s.translation != "":
                # gt_s.translation = " "
                scores_dict = scorer.get_scores(pred_s.source_text, pred_s.translation, gt_s.translation)
            else:
                scores_dict = scorer.get_scores(pred_s.source_text, pred_s.translation, gt_s.source_text)

            print("scores_dict: ", scores_dict)

            scores_dict['Source'] = pred_s.source_text
            scores_dict['Prediction'] = pred_s.translation
            scores_dict['Ground Truth'] = gt_s.source_text
            result_data.append(scores_dict)

        eval_df = pd.DataFrame(result_data)
        eval_df.to_csv(self.eval_path, index=False, columns=['Source', 'Prediction', 'Ground Truth', 'bleu_score', 'comet_score', 'llm_score', 'llm_explanation'])

        # Get average scores
        avg_llm = eval_df['llm_score'].mean()
        avg_bleu = eval_df['bleu_score'].mean()
        avg_comet = eval_df['comet_score'].mean()

        res_data = {
            'Metric': ['Avg LLM', 'Avg BLEU', 'Avg COMET'],
            'Score': [avg_llm, avg_bleu, avg_comet]
        }
        res_df = pd.DataFrame(res_data)
        res_df.to_csv(self.res_path, index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate SRT files.')
    parser.add_argument('-bi_path', default='evaluation/test5_tiny/test5_bi.srt', help='Path to predicted SRT file')
    parser.add_argument('-zh_path', default='evaluation/test5_tiny/test5_gt.srt', help='Path to ground truth SRT file')
    parser.add_argument('-eval_output', default='evaluation/test5_tiny/eval.csv', help='Path to eval CSV file')
    parser.add_argument('-res_output', default='evaluation/test5_tiny/res.csv', help='Path to result CSV file')
    args = parser.parse_args()

    evaluator = Evaluator(args.bi_path, args.zh_path, args.eval_output, args.res_output)
    evaluator.eval()


# python evaluation.py -bi_path /home/jiaenliu/project-t/results/test1/test1_bi.srt -zh_path test5_tiny/test1_gt.srt -eval_output /home/jiaenliu/project-t/evaluation/results/test1_large/eval.csv -res_output /home/jiaenliu/project-t/evaluation/results/test1_large/res.csv

