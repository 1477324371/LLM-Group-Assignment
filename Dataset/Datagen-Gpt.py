import json
import re
import random
import time
from datasets import load_dataset, Dataset, DatasetDict
from openai import OpenAI
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "3"

client = OpenAI(
    api_key="your deepseek api",          # 建议设为环境变量
    base_url="https://api.deepseek.com",
)

def llm_chat(prompt, temperature=0.7, max_tokens=2000):
    """封装 DeepSeek 调用，带重试和 JSON 提取"""
    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model="deepseek-chat",
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            content = resp.choices[0].message.content
            # 提取 JSON（去除可能的包裹）
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            return json_match.group() if json_match else content
        except Exception as e:
            print(f"DeepSeek 错误 (尝试 {attempt+1}): {e}")
            time.sleep(2 ** attempt)
    return None


# ============================================================
# 1. 加载数据集
# ============================================================
dataset = load_dataset("Team-ACE/ToolACE", split="train")
# dataset = dataset.select(range(10))
print(f"数据集大小: {len(dataset)}")


# ============================================================
# 2. 数据预处理：提取关键字段
# ============================================================
def extract_fields(sample):

    conversations = sample["conversations"]

    user_msgs = []
    tool_msgs = []
    assistant_msgs = []

    for conv in conversations:
        role = conv["from"]
        value = conv["value"]

        if role == "user":
            user_msgs.append(value)

        elif role == "tool":
            tool_msgs.append(value)

        elif role == "assistant":
            assistant_msgs.append(value)

    # 拼接成单条文本（推荐加 role tag，避免语义混乱）
    query = "\n".join(user_msgs) if user_msgs else None

    tool_output = "\n".join(tool_msgs) if tool_msgs else None

    model_response = "\n".join(assistant_msgs) if assistant_msgs else None

    # # 从 system 中解析可用工具名称
    # system = sample["system"]
    # tool_names = re.findall(r'"name":\s*"([^"]+)"', system)

    return {
        "query": query,
        "tool_output": tool_output,
        "model_response": model_response,
    }


# ============================================================
# 3. 原有规则方法（作为后备）
# ============================================================

# ============================================================
# 4. LLM 增强注入函数（优先使用，失败回退规则）
# ============================================================

def inject_contradiction_llm(sample):
    fields = extract_fields(sample)

    prompt = f"""你是数据编写员：给出你多轮对话中，所有用户的提问，所有工具输出和所有助手的回复。
    User Query: {fields['query']}
    Tool Output: {fields['tool_output']}
    Original Response: {fields['model_response']}
    使用这些信息里的部分信息，撰写一条模型响应与输出工具输出之间的幻觉数据,生成以下格式的Json文件,只返回Json,不要其他额外文本
    {{
        "query":"",
        "context":"",
        "output":"",
        "hallucination_labels":[
            {{
                "end":,
                "label_type":"",
                "start":,
                "text":""
            }}
        ]
    }}
    其中query, context和output之间的关系如下:
    query:用户提的一个问题,
    context:针对这个问题,给出Tool Output相关的答案,
    output:必须基于你编写的context内容回答这个问题,不要给context中没有提到的,回答的时候要修改一下，地点,数字等值使得与context中的值不同
    hallucination_labels中的text就是output中不正确的数字单词或者文本,hallucination_labels中的start和end是hallucination_labels中text在output中的开始和结束索引(整数)
    """
    raw = llm_chat(prompt, temperature=0.5)  # 降低温度提高稳定性
    if not raw:
        return None
    try:
        result = json.loads(raw)

        # 基础字段校验
        required_keys = ["query", "context", "output", "hallucination_labels"]
        for key in required_keys:
            if key not in result:
                return None

        # 校验 hallucination_labels
        if not isinstance(result["hallucination_labels"], list):
            return None

        for item in result["hallucination_labels"]:
            for k in ["start", "end", "text", "label_type"]:
                if k not in item:
                    return None

            # 自动校验索引是否正确
            text = item["text"]
            start = item["start"]
            end = item["end"]

            if result["output"][start:end] != text:
                corrected_start = result["output"].find(text)
                if corrected_start == -1:
                    return None

                item["start"] = corrected_start
                item["end"] = corrected_start + len(text)
        result["label_type"]="Evident Conflict"
        return result
    
    except Exception:
        return None



def inject_overgeneration_llm(sample):
    fields = extract_fields(sample)

    prompt = f"""你是数据编写员：给出你多轮对话中，所有用户的提问，所有工具输出和所有助手的回复。
    User Query: {fields['query']}
    Tool Output: {fields['tool_output']}
    Original Response: {fields['model_response']}
    使用这些信息里的部分信息，撰写一条模型过度生成的幻觉数据,
    生成以下格式的Json文件,只返回Json,不要其他额外文本
    {{
        "query":"",
        "context":"",
        "output":"",
        "hallucination_labels":[
            {{
                "end":,
                "label_type":"",
                "start":,
                "text":""
            }}
        ]
    }}
    其中query, context和output之间的关系如下:
    query:用户提的一个问题,
    context:针对这个问题,给出Tool Output相关的答案,
    output:基于你编写的context内容回答这个问题的时候, 增加一句工具输出中不存在的信息且不与工具输出中的事实相矛盾的内容
    hallucination_labels中的text就是output中凭空增加的不与工具输出中的事实相矛盾的内容,hallucination_labels中的start和end是hallucination_labels中text在output中的开始和结束索引(整数)
    """
    raw = llm_chat(prompt, temperature=0.5)  # 降低温度提高稳定性
    if not raw:
        return None
    try:
        result = json.loads(raw)

        # 基础字段校验
        required_keys = ["query", "context", "output", "hallucination_labels"]
        for key in required_keys:
            if key not in result:
                return None

        # 校验 hallucination_labels
        if not isinstance(result["hallucination_labels"], list):
            return None

        for item in result["hallucination_labels"]:
            for k in ["start", "end", "text", "label_type"]:
                if k not in item:
                    return None

            # 自动校验索引是否正确
            text = item["text"]
            start = item["start"]
            end = item["end"]

            if result["output"][start:end] != text:
                corrected_start = result["output"].find(text)
                if corrected_start == -1:
                    return None

                item["start"] = corrected_start
                item["end"] = corrected_start + len(text)
        result["label_type"]="Overgeneration"
        return result
    
    except Exception:
        return None


def inject_missing_tool_llm(sample):
    fields = extract_fields(sample)

    prompt = f"""你是数据编写员：给出你多轮对话中，所有用户的提问，所有工具输出和所有助手的回复。
    User Query: {fields['query']}
    Tool Output: {fields['tool_output']}
    Original Response: {fields['model_response']}
    使用这些信息里的部分信息，撰写一条工具缺失(当响应建议执行的操作需要调用当前不可用的工具时)的幻觉数据,生成以下格式的Json文件,只返回Json,不要其他额外文本
    {{
        "query":"",
        "context":"",
        "output":"",
        "hallucination_labels":[
            {{
                "end":,
                "label_type":"",
                "start":,
                "text":""
            }}
        ]
    }}
    其中query, context和output之间的关系如下:
    query:用户提的一个问题,
    context:针对这个问题,给出Tool Output相关的和不想关且不重复的答案,
    output:基于你编写的context内容回答这个问题的时候, 增加一句利用context中所有工具无法完成的句子
    hallucination_labels中的text就是output中利用context中所有工具无法完成的句子,hallucination_labels中的start和end是hallucination_labels中text在output中的开始和结束索引(整数)
    """
    raw = llm_chat(prompt, temperature=0.5)  # 降低温度提高稳定性
    if not raw:
        return None
    try:
        result = json.loads(raw)

        # 基础字段校验
        required_keys = ["query", "context", "output", "hallucination_labels"]
        for key in required_keys:
            if key not in result:
                return None

        # 校验 hallucination_labels
        if not isinstance(result["hallucination_labels"], list):
            return None

        for item in result["hallucination_labels"]:
            for k in ["start", "end", "text", "label_type"]:
                if k not in item:
                    return None

            # 自动校验索引是否正确
            text = item["text"]
            start = item["start"]
            end = item["end"]

            if result["output"][start:end] != text:
                corrected_start = result["output"].find(text)
                if corrected_start == -1:
                    return None

                item["start"] = corrected_start
                item["end"] = corrected_start + len(text)
        result["label_type"]="Missing Tool"
        return result
    
    except Exception:
        return None


# 混合函数：LLM 优先，失败回退规则
def inject_contradiction(sample):
    res = inject_contradiction_llm(sample)
    return res 

def inject_overgeneration(sample):
    res = inject_overgeneration_llm(sample)
    return res

def inject_missing_tool(sample):
    res = inject_missing_tool_llm(sample)
    return res


# ============================================================
# 5. 构建输出格式
# ============================================================
def build_output(fields, modified_response, hallucination_labels):
    return {
        "id": str(random.randint(100000, 999999)),
        "query": fields["query"] or "",
        "context": fields["tool_output"] or "",
        "output": modified_response,
        "hallucination_labels": hallucination_labels
    }


# ============================================================
# 6. 批量处理
# ============================================================
# def process_dataset(dataset, inject_func, dataset_name):
#     results = []
#     skipped = 0
#     for i, sample in enumerate(dataset):
#         try:
#             result = inject_func(sample)
#             if result is not None:
#                 result["id"] = str(i)
#                 results.append(result)
#             else:
#                 skipped += 1
#         except Exception as e:
#             skipped += 1
#         if (i + 1) % 500 == 0:
#             print(f"[{dataset_name}] 处理进度: {i+1}/{len(dataset)}")
#     print(f"[{dataset_name}] 成功处理: {len(results)}, 跳过: {skipped}")
#     return results

# import json
# from datasets import Dataset

def process_dataset(dataset, inject_func, dataset_name, save_every=1000, save_dir="./checkpoints"):
    import os
    os.makedirs(save_dir, exist_ok=True)

    results = []
    skipped = 0

    for i, sample in enumerate(dataset):
        try:
            result = inject_func(sample)
            if result is not None:
                result["id"] = str(i)
                results.append(result)
            else:
                skipped += 1
        except Exception:
            skipped += 1

        # ==============================
        # checkpoint 保存
        # ==============================
        if (i + 1) % save_every == 0:
            ckpt_json = os.path.join(save_dir, f"{dataset_name}_ckpt_{i+1}.json")
            # 保存 json
            with open(ckpt_json, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            print(f"[Checkpoint] 已保存 {i+1} 条 -> {ckpt_json}")

        if (i + 1) % 500 == 0:
            print(f"[{dataset_name}] 处理进度: {i+1}/{len(dataset)}")

    print(f"[{dataset_name}] 成功处理: {len(results)}, 跳过: {skipped}")
    return results
# ============================================================
# 7. 执行注入
# ============================================================
print("\n开始注入幻觉（LLM 优先，规则回退）...")
dataset_type1 = process_dataset(dataset, inject_contradiction, "Type1_Evident_Conflict", save_every=1000)
# dataset_type2 = process_dataset(dataset, inject_overgeneration, "Type2_Baseless_Info", save_every=1000)
# dataset_type3 = process_dataset(dataset, inject_missing_tool, "Type3_Missing_Tool", save_every=1000)

# 转为 HuggingFace Dataset
hf_dataset_type1 = Dataset.from_list(dataset_type1)
# hf_dataset_type2 = Dataset.from_list(dataset_type2)
# hf_dataset_type3 = Dataset.from_list(dataset_type3)

hallucinated_datasets = DatasetDict({
    "type1_evident_conflict": hf_dataset_type1,
    # "type2_baseless_info": hf_dataset_type2,
    # "type3_missing_tool": hf_dataset_type3,
})

print("\n数据集构建完成！")
print(hallucinated_datasets)

# ============================================================
# 8. 保存为 JSON 文件
# ============================================================
hf_dataset_type1.to_json("type1_evident_conflict.json")
# hf_dataset_type2.to_json("type2_baseless_info.json")
# hf_dataset_type3.to_json("type3_missing_tool.json")
