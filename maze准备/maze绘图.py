import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple


os.environ.setdefault("MPLCONFIGDIR", str(Path(".matplotlib_cache").resolve()))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(".cache").resolve()))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


# =========================
# 关键参数：按需修改这里即可
# =========================
INPUT_PATH = "data/maze_dataset_merged_600.jsonl"
OUTPUT_DIR = "600image"
RENDER_ALL_SAMPLES = True #是否应用于整个jsonl文件
SAMPLE_INDEX = 10 #如果上面是False，就会只应用于这个样本
FRAME_STRIDE = 1 #表示每一步都+1连续画图，2的话就表示当前步数+2画图
DPI = 180 #输出的分辨率
DRAW_LABELS = True #是否画ABCD的位置
WALL_LINEWIDTH = 3.0 #迷宫黑色墙线的粗细
PATH_LINEWIDTH = 3.0 #红色路径线的粗细


Coord = Tuple[int, int]
DELTA_BY_ACTION = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}


def load_json_or_jsonl(input_path: str | Path, sample_index: int = 0) -> Dict[str, Any]:
    """
    读取 JSON 或 JSONL 格式的迷宫数据。

    参数：
        input_path: 输入文件路径，支持 .json 和 .jsonl。
        sample_index: 当输入是 JSONL 或 JSON 列表时，要读取的样本序号。

    返回：
        单条迷宫样本字典。
    """
    input_path = Path(input_path)

    if input_path.suffix.lower() == ".jsonl":
        with input_path.open("r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                if idx == sample_index:
                    return json.loads(line)
        raise IndexError(f"JSONL 中不存在 sample_index={sample_index} 的样本。")

    with input_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        return data[sample_index]
    return data


def load_all_jsonl(input_path: str | Path) -> List[Dict[str, Any]]:
    """
    读取整个 JSONL 文件。

    参数：
        input_path: 输入 JSONL 文件路径。

    返回：
        JSON 对象列表。
    """
    input_path = Path(input_path)
    if input_path.suffix.lower() != ".jsonl":
        with input_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
        return [data]

    with input_path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def to_coord(value: Sequence[int]) -> Coord:
    """
    将 JSON 中的 [row, col] 坐标转换为 Python 元组。

    参数：
        value: 形如 [row, col] 的坐标。

    返回：
        形如 (row, col) 的坐标元组。
    """
    return int(value[0]), int(value[1])


def actions_to_path(start: Coord, moves: List[str]) -> List[Coord]:
    """
    将上下左右动作序列还原为逐步坐标路径。

    参数：
        start: 起点坐标。
        moves: 动作列表，每个元素为 up、down、left 或 right。

    返回：
        包含起点和每一步移动后坐标的路径。
    """
    path = [start]
    current = start

    for move in moves:
        if move not in DELTA_BY_ACTION:
            raise ValueError(f"未知 move 动作: {move}")
        dr, dc = DELTA_BY_ACTION[move]
        current = (current[0] + dr, current[1] + dc)
        path.append(current)

    return path


def get_move_path(sample: Dict[str, Any]) -> List[Coord]:
    """
    从样本中提取每一步 move 对应的路径。

    参数：
        sample: 单条迷宫样本。

    返回：
        坐标元组列表，每个坐标代表一步移动后的位置。
    """
    raw_path = sample.get("path") or sample.get("longest_path")
    if raw_path:
        return [to_coord(coord) for coord in raw_path]

    raw_moves = sample.get("moves")
    if raw_moves:
        start = to_coord(sample["start"])
        return actions_to_path(start, raw_moves)

    raise ValueError("输入数据中没有找到 path、longest_path 或 moves 字段。")


def coord_to_plot(coord: Coord) -> Tuple[float, float]:
    """
    将迷宫矩阵坐标映射到细线迷宫图的绘图坐标。

    参数：
        coord: 迷宫矩阵中的 (row, col) 坐标。

    返回：
        matplotlib 使用的 (x, y) 坐标。
    """
    row, col = coord
    return col / 2, row / 2


def draw_base_maze(ax: plt.Axes, maze: List[List[int]]) -> None:
    """
    绘制白底细黑线墙的迷宫底图。

    参数：
        ax: matplotlib 坐标轴。
        maze: 迷宫矩阵，0 表示通道，1 表示墙壁。
    """
    rows = len(maze)
    cols = len(maze[0])
    logical_rows = (rows - 1) // 2
    logical_cols = (cols - 1) // 2

    ax.set_facecolor("white")
    ax.set_xlim(0, logical_cols)
    ax.set_ylim(logical_rows, 0)
    ax.set_aspect("equal")

    ax.plot([0, logical_cols], [0, 0], color="black", linewidth=WALL_LINEWIDTH)
    ax.plot([0, logical_cols], [logical_rows, logical_rows], color="black", linewidth=WALL_LINEWIDTH)
    ax.plot([0, 0], [0, logical_rows], color="black", linewidth=WALL_LINEWIDTH)
    ax.plot([logical_cols, logical_cols], [0, logical_rows], color="black", linewidth=WALL_LINEWIDTH)

    for r in range(logical_rows):
        for c in range(logical_cols - 1):
            wall_row = 2 * r + 1
            wall_col = 2 * c + 2
            if maze[wall_row][wall_col] == 1:
                x = c + 1
                ax.plot([x, x], [r, r + 1], color="black", linewidth=WALL_LINEWIDTH)

    for r in range(logical_rows - 1):
        for c in range(logical_cols):
            wall_row = 2 * r + 2
            wall_col = 2 * c + 1
            if maze[wall_row][wall_col] == 1:
                y = r + 1
                ax.plot([c, c + 1], [y, y], color="black", linewidth=WALL_LINEWIDTH)

    ax.axis("off")


def draw_move_frame(
    maze: List[List[int]],
    path_so_far: List[Coord],
    start: Coord,
    end: Coord,
    output_path: str | Path,
    label_points: Dict[str, Coord] | None = None,
    dpi: int = DPI,
    draw_labels: bool = DRAW_LABELS,
) -> None:
    """
    渲染单步 move 的中间态截图。

    参数：
        maze: 迷宫矩阵，0 表示通道，1 表示墙壁。
        path_so_far: 截至当前帧已经走过的路径。
        start: 起点坐标。
        end: 终点坐标。
        output_path: 输出 PNG 图片路径。
        label_points: A/B/C/D 标注点坐标；若为空，则回退绘制 S/E。
        dpi: 图片分辨率。
        draw_labels: 是否绘制 Start、End 和当前位置标签。
    """
    rows = len(maze)
    cols = len(maze[0])
    logical_rows = (rows - 1) // 2
    logical_cols = (cols - 1) // 2
    fig_size = max(3.0, logical_cols * 0.9), max(3.0, logical_rows * 0.9)
    fig, ax = plt.subplots(figsize=fig_size)

    draw_base_maze(ax, maze)

    sx, sy = coord_to_plot(start)
    ax.plot(sx, sy, marker="o", color="red", markersize=8)

    if path_so_far:
        points = [coord_to_plot(coord) for coord in path_so_far]
        xs = [x for x, y in points]
        ys = [y for x, y in points]
        ax.plot(xs, ys, color="red", linewidth=PATH_LINEWIDTH, solid_capstyle="round", solid_joinstyle="round")

        cur_x, cur_y = coord_to_plot(path_so_far[-1])
        ax.plot(cur_x, cur_y, marker="x", color="red", markersize=8, markeredgewidth=1.8)

    if draw_labels:
        if label_points:
            for label in sorted(label_points):
                x, y = coord_to_plot(label_points[label])
                ax.text(x, y, label, ha="center", va="center", fontsize=13, color="#222222")
        else:
            ex, ey = coord_to_plot(end)
            ax.text(ex, ey, "E", ha="center", va="center", fontsize=13, color="#222222")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.02)
    fig.savefig(output_path, dpi=dpi, facecolor="white")
    plt.close(fig)


def render_move_frames(
    sample: Dict[str, Any],
    output_dir: str | Path,
    frame_stride: int = FRAME_STRIDE,
    dpi: int = DPI,
    draw_labels: bool = DRAW_LABELS,
) -> List[Path]:
    """
    将迷宫路径渲染为每一步 move 的截图序列。

    参数：
        sample: 单条迷宫样本。
        output_dir: PNG 截图输出目录。
        frame_stride: 帧间隔，1 表示每一步 move 都保存。
        dpi: 图片分辨率。
        draw_labels: 是否绘制文字标签。

    返回：
        生成的 PNG 文件路径列表。
    """
    maze = sample["maze"]
    start = to_coord(sample["start"])
    end = to_coord(sample["end"])
    move_path = get_move_path(sample)
    label_points = None
    if sample.get("label_points"):
        label_points = {label: to_coord(coord) for label, coord in sample["label_points"].items()}

    output_dir = Path(output_dir)
    sample_id = sample.get("id", 0)
    saved_paths: List[Path] = []

    frame_indices = list(range(0, len(move_path), max(1, frame_stride)))
    if frame_indices[-1] != len(move_path) - 1:
        frame_indices.append(len(move_path) - 1)

    for out_idx, path_idx in enumerate(frame_indices):
        path_so_far = move_path[: path_idx + 1]
        output_path = output_dir / f"maze_{sample_id}_move_{out_idx:04d}.png"
        draw_move_frame(
            maze,
            path_so_far,
            start,
            end,
            output_path,
            label_points=label_points,
            dpi=dpi,
            draw_labels=draw_labels,
        )
        saved_paths.append(output_path)

    return saved_paths


def render_dataset_frames(
    samples: List[Dict[str, Any]],
    output_dir: str | Path,
    frame_stride: int = FRAME_STRIDE,
    dpi: int = DPI,
    draw_labels: bool = DRAW_LABELS,
) -> List[Path]:
    """
    批量渲染多个迷宫样本，每个样本保存到独立子目录。

    参数：
        samples: 多条迷宫样本。
        output_dir: 总输出目录。
        frame_stride: 帧间隔，1 表示每一步 move 都保存。
        dpi: 图片分辨率。
        draw_labels: 是否绘制文字标签。

    返回：
        所有生成 PNG 的路径列表。
    """
    output_dir = Path(output_dir)
    all_saved_paths: List[Path] = []

    for merged_index, sample in enumerate(samples):
        sample_dir = output_dir / f"sample_{merged_index:06d}"
        saved_paths = render_move_frames(
            sample,
            output_dir=sample_dir,
            frame_stride=frame_stride,
            dpi=dpi,
            draw_labels=draw_labels,
        )
        all_saved_paths.extend(saved_paths)

        if (merged_index + 1) % 25 == 0 or merged_index == len(samples) - 1:
            print(f"Rendered {merged_index + 1}/{len(samples)} samples, frames={len(all_saved_paths)}")

    return all_saved_paths


def parse_args() -> argparse.Namespace:
    """
    解析命令行参数。

    返回：
        argparse.Namespace 参数对象。
    """
    parser = argparse.ArgumentParser(description="Render maze move-by-move PNG frames.")
    parser.add_argument("--input", default=INPUT_PATH, help="输入 JSON 或 JSONL 文件路径。")
    parser.add_argument("--output-dir", default=OUTPUT_DIR, help="输出截图目录。")
    parser.add_argument("--sample-index", type=int, default=SAMPLE_INDEX, help="JSONL 或 JSON 列表中的样本序号。")
    parser.add_argument("--render-all", action="store_true", default=RENDER_ALL_SAMPLES, help="渲染输入文件中的全部样本。")
    parser.add_argument("--single", action="store_true", help="只渲染 sample-index 指定的单个样本。")
    parser.add_argument("--frame-stride", type=int, default=FRAME_STRIDE, help="帧采样间隔，1 表示每一步都保存。")
    parser.add_argument("--dpi", type=int, default=DPI, help="输出图片 DPI。")
    parser.add_argument("--no-labels", action="store_true", help="不绘制 S、E、C 文字标签。")
    return parser.parse_args()


def main() -> None:
    """
    脚本入口：读取迷宫数据，并逐步渲染每一步 move 的截图。
    """
    args = parse_args()
    render_all = args.render_all and not args.single

    if render_all:
        samples = load_all_jsonl(args.input)
        saved_paths = render_dataset_frames(
            samples,
            output_dir=args.output_dir,
            frame_stride=args.frame_stride,
            dpi=args.dpi,
            draw_labels=not args.no_labels,
        )

        print(f"Rendered {len(samples)} samples and {len(saved_paths)} frames to: {Path(args.output_dir).resolve()}")
        return

    sample = load_json_or_jsonl(args.input, sample_index=args.sample_index)
    saved_paths = render_move_frames(
        sample,
        output_dir=args.output_dir,
        frame_stride=args.frame_stride,
        dpi=args.dpi,
        draw_labels=not args.no_labels,
    )

    print(f"Rendered {len(saved_paths)} move frames to: {Path(args.output_dir).resolve()}")
    if saved_paths:
        print(f"First frame: {saved_paths[0]}")
        print(f"Last frame: {saved_paths[-1]}")


if __name__ == "__main__":
    main()
