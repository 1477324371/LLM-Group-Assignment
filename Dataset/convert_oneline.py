# import json

# input_file = "contradiction_hallucination.json"
# output_file = "contradiction_hallucination.jsonl"

# # 读取整个 JSON 文件
# with open(input_file, "r", encoding="utf-8") as f:
#     data = json.load(f)

# # 写成 JSONL（一行一个 JSON）
# with open(output_file, "w", encoding="utf-8") as f:
#     for item in data:
#         f.write(json.dumps(item, ensure_ascii=False) + "\n")

# print(f"Converted to JSONL: {output_file}")

import json

# 输入文件路径
input_files = [
    "/trinity/home/senbao.zhang/project/TA/LLM_homework/Final_Project/Dataset/type1_evident_conflict.json",
    "/trinity/home/senbao.zhang/project/TA/LLM_homework/Final_Project/Dataset/type2_baseless_info.json",
    "/trinity/home/senbao.zhang/project/TA/LLM_homework/Final_Project/Dataset/type3_missing_tool.json"
]

# 输出文件
output_file = "merged.jsonl"

merged_data = []
new_id = 0

for file_path in input_files:
    with open(file_path, "r", encoding="utf-8") as f:

        for line in f:
            line = line.strip()

            # 跳过空行
            if not line:
                continue

            item = json.loads(line)

            # 重置 id
            item["id"] = str(new_id)

            merged_data.append(item)

            new_id += 1

# 保存为 jsonl
with open(output_file, "w", encoding="utf-8") as f:
    for item in merged_data:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

print(f"合并完成，共 {len(merged_data)} 条数据")