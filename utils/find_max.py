import argparse
import os

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--folder_path", "-fp", help="The path to the folder containing the log files.")
	args = parser.parse_args()
	
	# 定义文件夹路径
	folder_path = args.folder_path

	# 初始化存储每个文件最大 AUC 和 F1 的字典
	max_scores = {}

	# 遍历文件夹中的所有文件
	for file_name in os.listdir(folder_path):
		# 确保文件是文本文件
		if file_name.endswith(".txt"):
			file_path = os.path.join(folder_path, file_name)
			# 初始化当前文件的最大 AUC 和 F1
			max_auc = 0
			max_f1 = 0
			# 读取文件内容
			with open(file_path, "r") as file:
				lines = file.readlines()
				# 查找文件中的最大 AUC 和 F1
				for line in lines:
					if "Validation AUC:" in line:
						auc = float(line.split(":")[-1].strip())
						max_auc = max(max_auc, auc)
					elif "F1 score" in line:
						f1 = float(line.split(":")[-1].strip())
						max_f1 = max(max_f1, f1)
			# 存储当前文件的最大 AUC 和 F1
			max_scores[file_name] = {"Max AUC": max_auc, "Max F1 score": max_f1}

 	# 定义排序顺序，只考虑模型名字
	order = ["MLP", "GraphSAGE", "GeneralConv", "GINE", "EdgeConv", "GraphTransformer"]

	# 按照顺序排序文件名，只考虑模型名字部分
	filtered_keys = [key for key in max_scores.keys() if key != 'summary.txt']
	sorted_files = sorted(filtered_keys, key=lambda x: order.index(x.split('_')[0]))

	# 打印每个文件的最大 AUC 和 F1
	for file_name in sorted_files:
		print("File:", file_name)
		print(max_scores[file_name]["Max AUC"], max_scores[file_name]["Max F1 score"])