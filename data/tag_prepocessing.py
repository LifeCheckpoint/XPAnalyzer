import json
from pathlib import Path

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
    
    return char_tags
