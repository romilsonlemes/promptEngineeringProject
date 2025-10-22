import yaml

def load_llm_models_yaml(file_path: str, flatten: bool = False):
    with open(file_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if flatten:
        result = []
        for model, info in data.items():
            result.append({
                "Model": info.get("Model"),
                "Platform": info.get("Platform"),
                "Platform_API_KEY": info.get("Platform_API_KEY")
            })
        return result
    return data
