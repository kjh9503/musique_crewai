import json
from pathlib import Path

num=1
def main() -> None:
    input_path = Path("/home/sr2/kjh9503/musique/data/musique_full_v1.0_train.jsonl")
    items = []
    with input_path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                items.append(json.loads(line))
    
    for i in range(100):
        output_path = Path(f"/home/sr2/kjh9503/qa_task/data/musique_train_{i}.json")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(items[num*i:num*(i+1)], f, ensure_ascii=True, indent=2)


if __name__ == "__main__":
    main()
