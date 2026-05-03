import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple


# =========================
# Key parameters: edit here if needed
# =========================
# 输入的合并后迷宫样本 JSONL 路径（每行一个样本）
INPUT_JSONL = "data/maze_dataset_merged_600.jsonl"
# 渲染好的迷宫轨迹图像根目录，脚本会按 sample_xxxxxx 子目录去匹配图片
IMAGE_ROOT = "data/600image"
# 输出的 SFT/transition 样本 JSONL 路径
OUTPUT_JSONL = "data/maze_训练样本1.jsonl"
# 是否强制要求图像文件存在：
# True: 缺图就报错终止；False: 允许缺图路径写入（通常配合 --allow-missing-images）
REQUIRE_IMAGES = True
# 多模态占位符 token，会写入 conversations 的 value 中
IMAGE_TOKEN = "<image>"
# 从 merged 数据集的第几个样本开始处理（包含该下标）
START_INDEX = 0
# 处理到第几个样本结束（不包含该下标）；None 表示处理到结尾
END_INDEX = 1


ACTION_TEXT = {
    "up": "go up",
    "down": "go down",
    "left": "go left",
    "right": "go right",
}
SYSTEM_PROMPT = "You are a deterministic visual world model. Map the given observation and action to the next visual state."


Coord = Tuple[int, int]


def load_jsonl(input_path: str | Path) -> List[Dict[str, Any]]:
    """
    Load all records from a JSONL file.

    Args:
        input_path: Path to the input JSONL file.

    Returns:
        A list of JSON objects.
    """
    input_path = Path(input_path)
    with input_path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def save_jsonl(records: List[Dict[str, Any]], output_path: str | Path) -> None:
    """
    Save records to a JSONL file.

    Args:
        records: Records to save.
        output_path: Path to the output JSONL file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def coord_to_text(coord: Sequence[int]) -> str:
    """
    Format a coordinate as [row, col].

    Args:
        coord: Coordinate in [row, col] form.

    Returns:
        A formatted coordinate string.
    """
    return f"[{int(coord[0])}, {int(coord[1])}]"


def action_to_text(action: str) -> str:
    """
    Convert an action keyword into English instruction text.

    Args:
        action: One of up, down, left, or right.

    Returns:
        English action phrase.
    """
    if action not in ACTION_TEXT:
        raise ValueError(f"Unknown action: {action}")
    return ACTION_TEXT[action]


def find_frame_path(sample_dir: Path, frame_index: int) -> Path:
    """
    Find the rendered image path for a given frame index.

    Args:
        sample_dir: Directory for one rendered sample.
        frame_index: Frame index, where 0 is the initial observation.

    Returns:
        Path to the PNG frame.
    """
    matches = sorted(sample_dir.glob(f"*_move_{frame_index:04d}.png"))
    if not matches:
        return sample_dir / f"missing_move_{frame_index:04d}.png"
    if len(matches) > 1:
        raise ValueError(f"Multiple images found for frame {frame_index} in {sample_dir}")
    return matches[0]


def collect_image_paths(sample_index: int, frame_count: int, image_root: str | Path) -> List[str]:
    """
    Collect image paths for one sample.

    Args:
        sample_index: Index of the sample in the merged JSONL file.
        frame_count: Number of frames required. This equals len(moves) + 1.
        image_root: Root directory containing sample_000000 style subdirectories.

    Returns:
        Ordered image paths. The first path is the initial observation.
    """
    sample_dir = Path(image_root) / f"sample_{sample_index:06d}"
    return [str(find_frame_path(sample_dir, frame_index)) for frame_index in range(frame_count)]


def build_prompt(sample: Dict[str, Any]) -> str:
    """
    Build the English instruction prompt for one maze sample.

    Args:
        sample: One maze sample from the merged JSONL file.

    Returns:
        Prompt text.
    """
    label_points = sample["label_points"]
    moves = sample["moves"]

    destination_lines = [
        f"{label} coordinate: {coord_to_text(label_points[label])}."
        for label in ["A", "B", "C", "D"]
    ]
    action_sequence = " ".join(f"{action_to_text(action)}." for action in moves)

    return "\n".join(
        [
            "Task: Maze Navigation Simulation",
            "Given the action sequence, start from the initial position and determine the final destination (A, B, C, or D).",
            f"You must simulate the action sequence step by step. After each action, you must output an {IMAGE_TOKEN} tag to visualize the updated path on the maze before continuing to the next action.",
            "",
            "Action definitions:",
            "* Move up / move down / move left / move right: move one cell in the absolute up / down / left / right direction.",
            "",
            "Destination coordinates:",
            *destination_lines,
            "",
            f"Complete action sequence: {action_sequence}",
            "",
            f"Initial Observation: {IMAGE_TOKEN}",
            "Response:",
        ]
    )


def build_response(sample: Dict[str, Any]) -> str:
    """
    Build the target assistant response for one maze sample.

    Args:
        sample: One maze sample from the merged JSONL file.

    Returns:
        Response text.
    """
    path = sample["longest_path"]
    moves = sample["moves"]
    end_label = sample["end_label"]

    if len(path) != len(moves) + 1:
        raise ValueError(f"Path length and moves length do not match for sample id={sample.get('id')}")

    lines = [f"Initial agent coordinate: {coord_to_text(path[0])}."]

    for step_index, action in enumerate(moves, start=1):
        lines.append(f"Action: {action_to_text(action)}.")
        lines.append(IMAGE_TOKEN)
        lines.append("")
        lines.append(f"Agent coordinate: {coord_to_text(path[step_index])}.")

    lines.append(f"All actions have been completed. The final coordinate is {coord_to_text(path[-1])}.")
    lines.append(f"The answer is {end_label}.")
    return "\n".join(lines)


def validate_image_paths(image_paths: List[str]) -> None:
    """
    Validate that all image paths exist.

    Args:
        image_paths: Ordered image paths used by a training record.
    """
    missing = [path for path in image_paths if not Path(path).exists()]
    if missing:
        preview = ", ".join(missing[:3])
        raise FileNotFoundError(f"Missing rendered images. First missing paths: {preview}")


def build_sft_records(
    samples: List[Dict[str, Any]],
    image_root: str | Path,
    require_images: bool,
    start_index: int = START_INDEX,
    end_index: int | None = END_INDEX,
) -> List[Dict[str, Any]]:
    """
    Build transition-style SFT records from maze samples and rendered image paths.

    Args:
        samples: Maze samples from the merged JSONL file.
        image_root: Root directory containing rendered image folders.
        require_images: Whether to require all rendered images to exist.
        start_index: Inclusive start index in the merged JSONL file.
        end_index: Exclusive end index in the merged JSONL file. None means the end of the file.

    Returns:
        Transition records ready to save as JSONL.
    """
    sft_records: List[Dict[str, Any]] = []
    selected_samples = samples[start_index:end_index]

    for local_index, sample in enumerate(selected_samples):
        sample_index = start_index + local_index
        moves = sample["moves"]
        image_paths = collect_image_paths(sample_index, len(moves) + 1, image_root)
        if require_images:
            validate_image_paths(image_paths)

        for step_idx, action in enumerate(moves):
            sft_records.append(
                {
                    "id": f"maze_env_transition_{sample_index:06d}_{step_idx:04d}",
                    "image": [image_paths[step_idx], image_paths[step_idx + 1]],
                    "conversations": [
                        {
                            "from": "system",
                            "value": SYSTEM_PROMPT,
                        },
                        {
                            "from": "user",
                            "value": f"{IMAGE_TOKEN}\nAction: {action}",
                        },
                        {
                            "from": "assistant",
                            "value": IMAGE_TOKEN,
                        },
                    ],
                    "metadata": {
                        "sample_index": sample_index,
                        "step_index": step_idx,
                        "source_file": sample.get("source_file"),
                        "source_sample_index": sample.get("source_sample_index"),
                        "maze_id": sample.get("id"),
                        "action": action,
                    },
                }
            )

    return sft_records


def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments.

    Returns:
        Parsed command line arguments.
    """
    parser = argparse.ArgumentParser(description="Build English maze-navigation SFT JSONL records.")
    parser.add_argument(
        "--input",
        default=INPUT_JSONL,
        help="输入 merged 迷宫 JSONL 文件路径（默认使用文件头 INPUT_JSONL）。",
    )
    parser.add_argument(
        "--image-root",
        default=IMAGE_ROOT,
        help="渲染图像根目录（默认 data/600image，会在其下查找 sample_xxxxxx 子目录）。",
    )
    parser.add_argument(
        "--output",
        default=OUTPUT_JSONL,
        help="输出 JSONL 文件路径。",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=START_INDEX,
        help="起始样本下标（包含）。例如 0 表示从第一条样本开始。",
    )
    parser.add_argument(
        "--end-index",
        type=int,
        default=END_INDEX,
        help="结束样本下标（不包含）。例如 5 表示处理 [0, 5) 共 5 条；省略可处理到末尾。",
    )
    parser.add_argument(
        "--allow-missing-images",
        action="store_true",
        help="允许缺失图像文件时继续生成（开启后不会因缺图报错终止）。",
    )
    return parser.parse_args()


def main() -> None:
    """
    Script entry point: build SFT prompt/response records from maze JSONL and images.
    """
    args = parse_args()
    samples = load_jsonl(args.input)
    records = build_sft_records(
        samples=samples,
        image_root=args.image_root,
        require_images=not args.allow_missing_images,
        start_index=args.start_index,
        end_index=args.end_index,
    )
    save_jsonl(records, args.output)
    print(f"Saved {len(records)} records to: {Path(args.output).resolve()}")


if __name__ == "__main__":
    main()
