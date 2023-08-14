import pandas as pd
import matplotlib.pyplot as plt

# 从 Excel 文件读取数据
data = pd.read_excel('./results/ffhq1k-013-DiT_Uncondition-S-4/eval_scores.xlsx')

# 假设你的 Excel 文件中有 'x'、'y1'、'y2' 列
x = data['ckpt_name']
y1 = data['fid']
y2 = data['kid']

# 创建第一个图表，对应 y1 数据
plt.figure(figsize=(8, 5))
plt.plot(x, y1, label='y1', marker='o')
plt.title('ckpt - fid data')
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True)
plt.legend()
plt.savefig('line_chart_fid.png')  # 保存图表为图片文件
plt.show()

# 创建第二个图表，对应 y2 数据
plt.figure(figsize=(8, 5))
plt.plot(x, y2, label='y2', marker='s')
plt.title('ckpt - kid data')
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True)
plt.legend()
plt.savefig('line_chart_kid.png')
plt.show()
