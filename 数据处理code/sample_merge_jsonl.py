import argparse
import json
import random
from pathlib import Path
from typing import Dict, List


# =========================
# 关键参数：按需修改这里即可
# =========================
INPUT_PATHS = [
    "data/maze_dataset9.jsonl",
    "data/maze_dataset10.jsonl",
    "data/maze_dataset11.jsonl",
]
SAMPLES_PER_FILE = 200
OUTPUT_PATH = "data/maze_dataset_merged_600.jsonl"
RANDOM_SEED = 42
SHUFFLE_OUTPUT = True


def load_jsonl(input_path: str | Path) -> List[Dict]:
    """
    读取 JSONL 文件。

    参数：
        input_path: 输入 JSONL 文件路径。

    返回：
        JSON 对象列表。
    """
    input_path = Path(input_path)
    with input_path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def sample_records(records: List[Dict], sample_size: int, rng: random.Random) -> List[Dict]:
    """
    从记录列表中无放回随机抽样。

    参数：
        records: 候选样本列表。
        sample_size: 需要抽取的样本数量。
        rng: 随机数生成器。

    返回：
        抽样得到的样本列表。
    """
    if len(records) < sample_size:
        raise ValueError(f"样本数量不足：需要 {sample_size} 条，但只有 {len(records)} 条。")
    return rng.sample(records, sample_size)


def save_jsonl(records: List[Dict], output_path: str | Path) -> None:
    """
    将样本列表保存为 JSONL 文件。

    参数：
        records: 要保存的样本列表。
        output_path: 输出 JSONL 文件路径。
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def build_merged_dataset(
    input_paths: List[str],
    samples_per_file: int,
    output_path: str,
    seed: int,
    shuffle_output: bool,
) -> List[Dict]:
    """
    从多个 JSONL 文件中各抽取固定数量样本，并合并保存。

    参数：
        input_paths: 输入 JSONL 文件路径列表。
        samples_per_file: 每个文件抽取的样本数量。
        output_path: 合并后的 JSONL 输出路径。
        seed: 随机种子。
        shuffle_output: 是否打乱合并后的样本顺序。

    返回：
        合并后的样本列表。
    """
    rng = random.Random(seed)
    merged_records: List[Dict] = []

    for input_path in input_paths:
        records = load_jsonl(input_path)
        sampled = sample_records(records, samples_per_file, rng)
        source_name = Path(input_path).name

        for source_index, record in enumerate(sampled):
            record["source_file"] = source_name
            record["source_sample_index"] = source_index

        merged_records.extend(sampled)
        print(f"{input_path}: loaded={len(records)}, sampled={len(sampled)}")

    if shuffle_output:
        rng.shuffle(merged_records)

    save_jsonl(merged_records, output_path)
    return merged_records


def parse_args() -> argparse.Namespace:
    """
    解析命令行参数。

    返回：
        argparse.Namespace 参数对象。
    """
    parser = argparse.ArgumentParser(description="Sample records from multiple JSONL files and merge them.")
    parser.add_argument("--inputs", nargs="+", default=INPUT_PATHS, help="输入 JSONL 文件路径列表。")
    parser.add_argument("--samples-per-file", type=int, default=SAMPLES_PER_FILE, help="每个文件抽取的样本数量。")
    parser.add_argument("--output", default=OUTPUT_PATH, help="合并后的 JSONL 输出路径。")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED, help="随机种子。")
    parser.add_argument("--no-shuffle", action="store_true", help="不打乱合并后的输出顺序。")
    return parser.parse_args()


def main() -> None:
    """
    脚本入口：从三个数据文件中各抽样并合并为总 JSONL。
    """
    args = parse_args()
    merged_records = build_merged_dataset(
        input_paths=args.inputs,
        samples_per_file=args.samples_per_file,
        output_path=args.output,
        seed=args.seed,
        shuffle_output=not args.no_shuffle,
    )
    print(f"Saved {len(merged_records)} records to: {Path(args.output).resolve()}")


if __name__ == "__main__":
    main()
