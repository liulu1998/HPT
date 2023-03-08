from collections import defaultdict
from typing import List

import torch

id2label = {
    0: "校园欺凌", 1: "网络暴力", 2: "低效感",
    3: "自杀", 4: "容貌焦虑", 5: "负性情绪",
    6: "心理危机", 7: "精神障碍",
    8: "个人成长",
    9: "受创经历",
    10: "原生家庭",
    11: "学业压力",
    12: "亲密关系",
    13: "人际障碍",
    14: "问题行为",
    15: "行为矫正",
    16: "性",
    17: "追星",
    18: "公共事件",
    19: "安全感",
    20: "自我接纳",
    21: "人生意义",
    22: "家庭压力",
    23: "家长期许",
    24: "恋爱经营",
    25: "依赖依恋",
    26: "失恋",
    27: "人际边界",
    28: "倾诉倾听",
    29: "社交恐惧",
    30: "社交障碍",
    31: "自虐",
    32: "攻击行为",
    33: "暴食节食",
    34: "懒惰",
    35: "手机依赖",
    36: "性行为",
    37: "性取向",
    38: "性别认同"
}

label2id = {v: k for k, v in id2label.items()}


# label_cnt = {'低效感': 484, '心理危机': 1259, '负性情绪': 976, '问题行为': 428, '个人成长': 911, '自杀': 187,
#              '校园欺凌': 128, '学业压力': 622, '行为矫正': 402, '人际障碍': 815, '原生家庭': 1004, '受创经历': 365,
#              '亲密关系': 387, '性': 60, '容貌焦虑': 86, '家庭压力': 158, '精神障碍': 491, '公共事件': 6, '追星': 9,
#              '网络暴力': 7, '社交障碍': 3, '家长期许': 18, '性别认同': 24, '倾诉倾听': 0, '手机依赖': 0, '暴食节食': 0,
#              '懒惰': 0, '人生意义': 0, '自虐': 0, '性取向': 0, '失恋': 0, '社交恐惧': 0, '安全感': 0, '人际边界': 0,
#              '依赖依恋': 0, '性行为': 0, '恋爱经营': 0, '攻击行为': 0, '自我接纳': 0}


def build_label_hierarchy(hierarchy_file) -> (dict, dict):
    # int -> Set of int, parent to childs, both label id
    tree = defaultdict(set)

    # int -> int, child to parent, both label id
    tree_r = dict()

    def parse_line(line):
        parent, childs = line.split()[0], line.split()[1:]

        parent: int = label2id[parent] if parent.upper() != "ROOT" else -1
        childs: List[int] = [label2id[c] for c in childs]

        if parent != -1:
            for c in childs:
                tree[parent].add(c)

        if parent != -1:
            for c in childs:
                tree_r[c] = parent

    # << inner method

    with open(hierarchy_file, "r", encoding="utf-8") as f:
        for line in f.readlines():
            parse_line(line)

    for id_ in id2label:
        if id_ not in tree:
            pass

    return tree, tree_r


def to_pt(data, save_file_name: str = "./data.pt"):
    torch.save(data, save_file_name)


if __name__ == '__main__':
    to_pt(id2label, "value_dict.pt")
    tree, tree_r = build_label_hierarchy("./yixinli.taxnomy")

    print(tree)
    to_pt(tree, "slot.pt")
