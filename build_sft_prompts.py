import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple


# =========================
# Key parameters: edit here if needed
# =========================
INPUT_JSONL = "data/maze_dataset_merged_600.jsonl"
IMAGE_ROOT = "data/600image"
OUTPUT_JSONL = "data/maze_navigation_sft_600.jsonl"
REQUIRE_IMAGES = True
IMAGE_TOKEN = "<image>"
START_INDEX = 0
END_INDEX = 5


ACTION_TEXT = {
    "up": "go up",
    "down": "go down",
    "left": "go left",
    "right": "go right",
}


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
    Build SFT records from maze samples and rendered image paths.

    Args:
        samples: Maze samples from the merged JSONL file.
        image_root: Root directory containing rendered image folders.
        require_images: Whether to require all rendered images to exist.
        start_index: Inclusive start index in the merged JSONL file.
        end_index: Exclusive end index in the merged JSONL file. None means the end of the file.

    Returns:
        SFT records ready to save as JSONL.
    """
    sft_records: List[Dict[str, Any]] = []
    selected_samples = samples[start_index:end_index]

    for local_index, sample in enumerate(selected_samples):
        sample_index = start_index + local_index
        moves = sample["moves"]
        image_paths = collect_image_paths(sample_index, len(moves) + 1, image_root)
        if require_images:
            validate_image_paths(image_paths)

        sft_records.append(
            {
                "id": f"maze_nav_{sample_index:06d}",
                "images": image_paths,
                "messages": [
                    {
                        "role": "user",
                        "content": build_prompt(sample),
                    },
                    {
                        "role": "assistant",
                        "content": build_response(sample),
                    },
                ],
                "metadata": {
                    "sample_index": sample_index,
                    "source_file": sample.get("source_file"),
                    "source_sample_index": sample.get("source_sample_index"),
                    "maze_id": sample.get("id"),
                    "answer": sample["end_label"],
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
    parser.add_argument("--input", default=INPUT_JSONL, help="Input merged maze JSONL file.")
    parser.add_argument("--image-root", default=IMAGE_ROOT, help="Root directory of rendered maze images.")
    parser.add_argument("--output", default=OUTPUT_JSONL, help="Output SFT JSONL file.")
    parser.add_argument("--start-index", type=int, default=START_INDEX, help="Inclusive start sample index.")
    parser.add_argument("--end-index", type=int, default=END_INDEX, help="Exclusive end sample index. Omit to use all remaining samples.")
    parser.add_argument("--allow-missing-images", action="store_true", help="Do not fail if rendered images are missing.")
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
