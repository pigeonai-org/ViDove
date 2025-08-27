import pandas as pd

class CalAvgScoresInCsv:
    def __init__(self, csv_path):
        self.csv_path = csv_path
        
    def cal_avg_scores(self):
        df = pd.read_csv(self.csv_path)

        # 过滤【MT】栏位不为 "。" 的行（跳过【Translation的结果】不为空的情况）
        filtered_df = df[df['MT'] != '。']
        
        print(f'过滤后的行数: {len(filtered_df)}')
        # 计算 COMET / LLM Score 栏位的平均值
        average_comet = filtered_df['COMET'].mean()
        average_llm_score = filtered_df['LLM Score'].mean()

        print(f'COMET 栏位的平均值为: {average_comet}')
        print(f'LLM Score 栏位的平均值为: {average_llm_score}')

if __name__ == "__main__":
    cal_avg_scores_in_csv = CalAvgScoresInCsv('./evaluation/test_data/gemini_result.csv')
    cal_avg_scores_in_csv.cal_avg_scores()
