merge.py: to merge the csv files, just run it and the merged files will be saved in the /data/output folder.

check_empty.py: 检查是否存在空值
🧠 总结建议表：
特征名	建议处理方式
sales	log1p → 预测后再 expm1
date	用于排序，生成派生时间特征
store_nbr	Embedding
family	Embedding
city	Embedding（类别较多）或 One-Hot
state	One-Hot
type	One-Hot
cluster	Z-score
onpromotion	直接使用
dcoilwtico	Z-score
is_working_day	直接使用
quake_severe	直接使用
quake_moderate	直接使用
payday	直接使用
sample_weight	Z-score 或归一化
时间特征（派生）	Z-score

