import pandas as pd

# 读取 CSV 文件
df = pd.read_csv('./evaluation/test_data/result.csv')

# 过滤 MT 栏位不为 "。" 的行
filtered_df = df[df['MT'] != '。']

# 计算 COMET 栏位的平均值
average_comet = filtered_df['COMET'].mean()

# 计算 LLM Score 栏位的平均值
average_llm_score = filtered_df['LLM Score'].mean()



print(f'COMET 栏位的平均值为: {average_comet}')
print(f'LLM Score 栏位的平均值为: {average_llm_score}')
