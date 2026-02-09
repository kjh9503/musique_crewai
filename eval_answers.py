#!/usr/bin/env python3
import argparse
import glob
import json
import os
import re
from typing import Any, Dict, List

def load_samples(json_path: str) -> List[Dict[str, Any]]:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    samples = []
    for sample in data:
        samples.append(
            {
                "id": sample.get("id"),
                "question": sample.get("question"),
                "answer": sample.get("answer"),
            }
        )
    return samples


def load_answer_text(md_path: str) -> str:
    with open(md_path, "r", encoding="utf-8") as f:
        return f.read().strip()


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def judge_samples(
    samples: List[Dict[str, Any]],
    candidate_answer: str,
) -> List[Dict[str, Any]]:
    normalized_answer = normalize_text(candidate_answer)
    judged = []
    for sample in samples:
        gold = (sample.get("answer") or "").strip()
        gold_norm = normalize_text(gold)
        is_correct = gold_norm != "" and gold_norm in normalized_answer
        judged.append(
            {
                "id": sample.get("id"),
                "question": sample.get("question"),
                "gold_answer": gold,
                "extracted_answer": gold if is_correct else "",
                "verdict": "correct" if is_correct else "incorrect",
                "reason": "gold answer appears in answer text"
                if is_correct
                else "gold answer not found in answer text",
            }
        )
    return judged


def score_items(items: List[Dict[str, Any]]) -> Dict[str, int]:
    counts = {"correct": 0, "incorrect": 0}
    for item in items:
        verdict = item.get("verdict")
        if verdict not in counts:
            verdict = "incorrect"
            item["verdict"] = verdict
        counts[verdict] += 1
    return counts


def render_markdown(results: List[Dict[str, Any]], summary: Dict[str, int]) -> str:
    lines = []
    total = sum(summary.values())
    lines.append("# Evaluation Report")
    lines.append("")
    lines.append(
        f"- Total questions: {total} | "
        f"correct: {summary['correct']} | "
        f"incorrect: {summary['incorrect']}"
    )
    lines.append("")
    for file_result in results:
        lines.append(f"## {file_result['file_id']}")
        lines.append("")
        for item in file_result["items"]:
            lines.append(
                f"- {item['question']} => {item['verdict']} "
                f"(gold: {item['gold_answer']}; extracted: {item['extracted_answer']})"
            )
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Substring evaluator for musique_train_x.json answers."
    )
    parser.add_argument(
        "--data_dir",
        default="./data",
        help="Directory containing musique_train_*.json files.",
    )
    parser.add_argument(
        "--answers_dir",
        default="./output",
        help="Directory containing answer_*.md files.",
    )
    parser.add_argument(
        "--output_json",
        default="./eval_report.json",
        help="Path to write JSON report.",
    )
    parser.add_argument(
        "--output_md",
        default="./eval_report.md",
        help="Path to write Markdown report.",
    )
    args = parser.parse_args()

    data_files = sorted(glob.glob(os.path.join(args.data_dir, "musique_train_*.json")))
    results = []
    total_counts = {"correct": 0, "incorrect": 0}

    for data_path in data_files:
        match = re.search(r"musique_train_(\d+)\.json", os.path.basename(data_path))
        if not match:
            continue
        idx = match.group(1)
        answer_path = os.path.join(args.answers_dir, f"answer_{idx}.md")
        if not os.path.exists(answer_path):
            continue
        samples = load_samples(data_path)
        candidate_answer = load_answer_text(answer_path)
        judged_items = judge_samples(samples, candidate_answer)
        counts = score_items(judged_items)
        for k in total_counts:
            total_counts[k] += counts[k]
        results.append(
            {
                "file_id": f"answer_{idx}.md",
                "data_file": os.path.basename(data_path),
                "counts": counts,
                "items": judged_items,
            }
        )

    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(
            {"summary": total_counts, "results": results},
            f,
            ensure_ascii=False,
            indent=2,
        )

    with open(args.output_md, "w", encoding="utf-8") as f:
        f.write(render_markdown(results, total_counts))

    print("Done.")
    print(f"JSON: {args.output_json}")
    print(f"Markdown: {args.output_md}")


if __name__ == "__main__":
    main()
