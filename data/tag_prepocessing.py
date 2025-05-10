from config.configs import ExperimentConfig
import json
from pathlib import Path

cfg = ExperimentConfig()

def load_character_tags(json_path: str) -> dict[str, set[str]]:
    """
    从 data_min.json 加载角色-tag映射
    """
    file_path = Path(json_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"JSON文件不存在: {json_path}")
    
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    char_index = data["char_index"]
    char2attr = data["char2attr"]
    attrs = data["attr_index"]
    
    char_tags = {}
    for char_name, tags in zip(char_index, char2attr):
        try:
            char_tags[char_name] = set(attrs[int(tag_id)] for tag_id in tags)
        except (IndexError, ValueError) as e:
            print(f"处理角色 f{char_name} 错误: {e}")
            continue

    print(f"Current characters num: {len(char_tags)}")

    # 过滤小标签频率
    tag_freq = {}
    for tags in char_tags.values():
        for tag in tags:
            tag_freq[tag] = tag_freq.get(tag, 0) + 1

    print(f"Current tags num: {len(tag_freq)}")
    tag_freq_filter = {tag: freq for tag, freq in tag_freq.items() if freq >= cfg.dataset.tag_freq_threshold}
    print(f"Filtered tags num: {len(tag_freq_filter)}")

    # 删除低频标签，全低频标签角色
    for char_name, tags in char_tags.items():
        char_tags[char_name] = {tag for tag in tags if tag in tag_freq_filter}
    char_tags = {char_name: tags for char_name, tags in char_tags.items() if len(tags) > 1}

    print(f"Filtered characters num: {len(char_tags)}")
    
    return char_tags
