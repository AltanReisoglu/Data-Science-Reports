import os
import json
import subprocess

def main():
    print("🚀 RandOpt YT Tutorial Hazırlık Scripti 🚀\n")
    
    # 1. RandOpt reposunu klonla
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_dir = os.path.join(script_dir, "RandOpt_repo")
    if not os.path.exists(repo_dir):
        print("[1/4] RandOpt GitHub reposu klonlanıyor...")
        subprocess.run(["git", "clone", "https://github.com/sunrainyg/RandOpt.git", repo_dir], check=True)
    else:
        print("[1/4] RandOpt klasörü zaten var, klonlama atlandı.")

    # 2. Örnek Veri Seti Oluştur
    print("[2/4] 'tutorial_math' veri seti oluşturuluyor...")
    data_dir = os.path.join(repo_dir, "data", "tutorial_math")
    os.makedirs(data_dir, exist_ok=True)
    
    dummy_data = [
        {"question": "What is 15 + 27?", "answer": "42"},
        {"question": "If x = 5, what is 3x + 2?", "answer": "17"},
        {"question": "How many sides does a hexagon have?", "answer": "6"},
        {"question": "What is 120 / 4?", "answer": "30"},
        {"question": "Is 7 a prime number? Answer Yes or No.", "answer": "Yes"}
    ]
    
    with open(os.path.join(data_dir, "data.json"), "w", encoding="utf-8") as f:
        json.dump(dummy_data, f, indent=4)

    # 3. Reward Fonksiyonunu Yaz
    print("[3/4] Reward fonksiyonu yazılıyor...")
    reward_code = """\"\"\"Reward scoring for tutorial_math dataset.\"\"\"
import re

def extract_answer(response: str) -> str:
    m = re.search(r"<answer>(.*?)</answer>", response, re.DOTALL)
    # Eğer <answer> tag'i yoksa son kelimeyi al.
    return m.group(1).strip() if m else response.strip().split()[-1].strip()

def compute_score(response: str, ground_truth: str) -> float:
    ans = extract_answer(response)
    if ans.lower() == str(ground_truth).strip().lower():
        return 1.0
    # Noktalama işaretini atıp tekrar dene
    if ans.strip('.').lower() == str(ground_truth).strip().lower():
        return 1.0
    return 0.0
"""
    with open(os.path.join(repo_dir, "utils", "reward_score", "tutorial_math.py"), "w", encoding="utf-8") as f:
        f.write(reward_code)

    # 4. Data Handler Oluştur ve Kaydet
    print("[4/4] Data handler oluşturuluyor ve register ediliyor...")
    handler_code = """\"\"\"Tutorial Math dataset handler.\"\"\"
import json
from typing import Dict, List, Optional
from utils.reward_score import tutorial_math as math_reward
from .base import DatasetHandler

class TutorialMathHandler(DatasetHandler):
    name = "tutorial_math"
    default_train_path = "data/tutorial_math/data.json"
    default_test_path = "data/tutorial_math/data.json"
    default_max_tokens = 256

    def load_data(self, path, split="train", max_samples=None) -> List[Dict]:
        with open(path) as f:
            raw = json.load(f)
        out = []
        for item in raw:
            out.append({
                "messages": [{"role": "user", "content": item["question"]}],
                "ground_truth": item["answer"],
            })
            if max_samples and len(out) >= max_samples:
                break
        return out
"""
    with open(os.path.join(repo_dir, "data_handlers", "tutorial_math.py"), "w", encoding="utf-8") as f:
        f.write(handler_code)

    # __init__.py patch (sadece import'u ekle)
    init_file = os.path.join(repo_dir, "data_handlers", "__init__.py")
    if os.path.exists(init_file):
        with open(init_file, "r", encoding="utf-8") as f:
            content = f.read()
            
        if "from .tutorial_math import TutorialMathHandler" not in content:
            # import statement'i ekle
            import_str = "from .tutorial_math import TutorialMathHandler\n"
            # DATASET_HANDLERS sözlüğüne ekle
            dict_str = "    \"tutorial_math\": TutorialMathHandler,"
            
            # Dictionary içine inject edelim
            if "DATASET_HANDLERS = {" in content:
                content = import_str + "\n" + content
                content = content.replace("DATASET_HANDLERS = {", "DATASET_HANDLERS = {\n" + dict_str)
                with open(init_file, "w", encoding="utf-8") as f:
                    f.write(content)

    print("\n✅ Kurulum tamamlandı! RandOpt şimdi kendi 'tutorial_math' veri setimizle çalışmaya hazır.")
    print("▶️ Öğreticide göstermek için 'run_demo.sh' dosyasını çalıştırın.")

if __name__ == "__main__":
    main()
