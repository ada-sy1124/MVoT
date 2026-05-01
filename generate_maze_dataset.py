import json
import random
from collections import deque
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


# =========================
# 关键参数：按需修改这里即可
# =========================
NUM_MAZES = 1000
MAZE_SIZE = 10
RANDOM_SEED = 42
OUTPUT_PATH = "maze_dataset10.jsonl"


Coord = Tuple[int, int]
LABELS = ("A", "B", "C", "D")
ACTION_BY_DELTA = {
    (-1, 0): "up",
    (1, 0): "down",
    (0, -1): "left",
    (0, 1): "right",
}


def _normalize_maze_size(maze_size: int) -> int:
    """规范化迷宫尺寸，确保尺寸至少为 5 且为奇数。"""
    if maze_size < 5:
        maze_size = 5
    if maze_size % 2 == 0:
        maze_size += 1
    return maze_size


def generate_perfect_maze_dfs(maze_size: int, rng: random.Random | None = None) -> np.ndarray:
    """
    使用 DFS 回溯算法生成一个完美迷宫。

    参数：
        maze_size: 迷宫尺寸。若传入偶数，会自动调整为下一个奇数。
        rng: 随机数生成器，用于控制随机性和复现实验结果。

    返回：
        numpy.ndarray 类型的迷宫矩阵，其中 0 表示通道，1 表示墙壁。
    """
    maze_size = _normalize_maze_size(maze_size)
    rng = rng or random.Random()

    maze = np.ones((maze_size, maze_size), dtype=np.uint8)

    # Start carving from (1,1), stepping by 2 to keep walls between cells.
    start = (1, 1)
    maze[start] = 0
    stack: List[Coord] = [start]

    directions = [(-2, 0), (2, 0), (0, -2), (0, 2)]

    while stack:
        r, c = stack[-1]
        neighbors: List[Coord] = []

        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            if 1 <= nr < maze_size - 1 and 1 <= nc < maze_size - 1 and maze[nr, nc] == 1:
                neighbors.append((nr, nc))

        if neighbors:
            nr, nc = rng.choice(neighbors)
            # Carve wall between current cell and next cell.
            maze[(r + nr) // 2, (c + nc) // 2] = 0
            maze[nr, nc] = 0
            stack.append((nr, nc))
        else:
            stack.pop()

    return maze


def _bfs_farthest(maze: np.ndarray, src: Coord) -> Tuple[Coord, Dict[Coord, Coord | None], Dict[Coord, int]]:
    """
    从指定通道坐标开始执行 BFS，找到距离最远的通道节点。

    参数：
        maze: 迷宫矩阵，0 表示通道，1 表示墙壁。
        src: BFS 起始坐标。

    返回：
        farthest: 从 src 出发可达的最远坐标。
        parent: BFS 搜索树中的父节点映射，用于还原路径。
        dist: 从 src 到各个可达坐标的距离映射。
    """
    h, w = maze.shape
    q = deque([src])
    visited = {src}
    parent: Dict[Coord, Coord | None] = {src: None}
    dist: Dict[Coord, int] = {src: 0}

    farthest = src

    while q:
        r, c = q.popleft()

        # Track farthest by distance; tie-break by coordinate for determinism.
        if dist[(r, c)] > dist[farthest] or (dist[(r, c)] == dist[farthest] and (r, c) < farthest):
            farthest = (r, c)

        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and maze[nr, nc] == 0:
                nxt = (nr, nc)
                if nxt not in visited:
                    visited.add(nxt)
                    parent[nxt] = (r, c)
                    dist[nxt] = dist[(r, c)] + 1
                    q.append(nxt)

    return farthest, parent, dist


def longest_path_by_two_bfs(maze: np.ndarray) -> Tuple[Coord, Coord, List[Coord]]:
    """
    使用“两次 BFS 算法”求迷宫连通图的直径路径。

    算法步骤：
        1. 从任意通道节点 A 出发 BFS，找到最远节点 B。
        2. 从节点 B 再次 BFS，找到最远节点 C。
        3. B 到 C 的路径即作为该迷宫的最长连通路径。

    参数：
        maze: 迷宫矩阵，0 表示通道，1 表示墙壁。

    返回：
        start: 最长路径起点坐标。
        end: 最长路径终点坐标。
        path: 从 start 到 end 的完整最长路径坐标列表。
    """
    passages = np.argwhere(maze == 0)
    if len(passages) == 0:
        raise ValueError("Maze has no passage cells.")

    a = tuple(int(x) for x in passages[0])
    b, _, _ = _bfs_farthest(maze, a)
    c, parent, _ = _bfs_farthest(maze, b)

    # Reconstruct diameter path c -> b via parent, then reverse to b -> c.
    path_rev: List[Coord] = []
    cur: Coord | None = c
    while cur is not None:
        path_rev.append(cur)
        cur = parent[cur]

    path = list(reversed(path_rev))
    start, end = path[0], path[-1]
    return start, end, path


def generate_label_points(
    maze: np.ndarray,
    end: Coord,
    rng: random.Random,
) -> Tuple[Dict[str, Coord], str, List[Coord]]:
    """
    从全图通道中随机生成 3 个额外点，并与原本终点一起随机分配为 A/B/C/D。

    参数：
        maze: 迷宫矩阵，0 表示通道，1 表示墙壁。
        end: 两次 BFS 得到的真实终点坐标。
        rng: 随机数生成器，用于控制标注点采样和标签分配。

    返回：
        label_points: A/B/C/D 到坐标的映射。
        end_label: 原本真实终点对应的标签。
        random_points: 除真实终点外额外随机生成的 3 个点。
    """
    passages = [tuple(int(x) for x in coord) for coord in np.argwhere(maze == 0)]
    candidates = [coord for coord in passages if coord != end]

    if len(candidates) < 3:
        raise ValueError("迷宫通道点不足，无法额外随机生成 3 个不同标注点。")

    random_points = rng.sample(candidates, 3)
    points = [end] + random_points
    shuffled_labels = list(LABELS)
    rng.shuffle(shuffled_labels)

    label_points = {label: point for label, point in zip(shuffled_labels, points)}
    end_label = shuffled_labels[0]

    return label_points, end_label, random_points


def path_to_moves(path: List[Coord]) -> List[str]:
    """
    将相邻坐标路径转换为上下左右动作序列。

    参数：
        path: 从起点到终点的坐标路径，相邻坐标必须只差一步。

    返回：
        moves 动作列表，每个元素为 up、down、left 或 right。
    """
    moves: List[str] = []
    for current, nxt in zip(path, path[1:]):
        dr = nxt[0] - current[0]
        dc = nxt[1] - current[1]
        action = ACTION_BY_DELTA.get((dr, dc))
        if action is None:
            raise ValueError(f"路径中存在非上下左右的一步移动: {current} -> {nxt}")
        moves.append(action)
    return moves


def generate_maze_dataset(n: int, maze_size: int, seed: int | None = None) -> List[Dict]:
    """
    批量生成迷宫数据集，并为每个迷宫计算真实起点、终点和 A/B/C/D 标注点。

    参数：
        n: 要生成的迷宫数量。
        maze_size: 每个迷宫的尺寸。
        seed: 随机种子，用于保证生成结果可复现。

    返回：
        字典列表。每个字典包含迷宫矩阵、起点、终点、最长路径、A/B/C/D 标注点。
    """
    if n <= 0:
        raise ValueError("n must be > 0")

    rng = random.Random(seed)
    dataset: List[Dict] = []

    for i in range(n):
        maze = generate_perfect_maze_dfs(maze_size, rng=rng)
        start, end, longest_path = longest_path_by_two_bfs(maze)
        moves = path_to_moves(longest_path)
        label_points, end_label, random_points = generate_label_points(maze, end, rng)

        sample = {
            "id": i,
            "maze_size": int(maze.shape[0]),
            "maze": maze,
            "start": start,
            "end": end,
            "label_points": label_points,
            "end_label": end_label,
            "random_points": random_points,
            "longest_path_length": len(longest_path),
            "longest_path": longest_path,
            "moves": moves,
        }
        dataset.append(sample)

    return dataset


def _to_jsonl_record(sample: Dict) -> Dict:
    """
    将单个迷宫样本转换为可写入 JSONL 的普通 Python 数据结构。

    参数：
        sample: generate_maze_dataset 生成的单个迷宫样本。

    返回：
        可被 json.dumps 序列化的字典。
    """
    return {
        "id": sample["id"],
        "maze_size": sample["maze_size"],
        "maze": sample["maze"].tolist(),
        "start": list(sample["start"]),
        "end": list(sample["end"]),
        "label_points": {label: list(coord) for label, coord in sample["label_points"].items()},
        "end_label": sample["end_label"],
        "random_points": [list(coord) for coord in sample["random_points"]],
        "longest_path_length": sample["longest_path_length"],
        "longest_path": [list(coord) for coord in sample["longest_path"]],
        "moves": sample["moves"],
    }


def save_dataset_jsonl(dataset: List[Dict], output_path: str | Path) -> None:
    """
    将迷宫数据集保存为 JSONL 文件。

    JSONL 格式中每一行都是一个完整 JSON 对象，适合大规模数据集逐行读取。

    参数：
        dataset: generate_maze_dataset 返回的迷宫样本列表。
        output_path: 输出文件路径。
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        for sample in dataset:
            record = _to_jsonl_record(sample)
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    """运行示例：生成迷宫数据集，保存为 JSONL，并打印第一个迷宫的信息。"""
    dataset = generate_maze_dataset(n=NUM_MAZES, maze_size=MAZE_SIZE, seed=RANDOM_SEED)
    save_dataset_jsonl(dataset, OUTPUT_PATH)

    print("Generated samples:", len(dataset))
    print("Saved JSONL to:", OUTPUT_PATH)
    print("\nFirst maze matrix (0=passage, 1=wall):")
    print(dataset[0]["maze"])
    print("First maze start:", dataset[0]["start"])
    print("First maze end:", dataset[0]["end"])
    print("First maze label points:", dataset[0]["label_points"])
    print("First maze true end label:", dataset[0]["end_label"])
    print("First maze longest path length:", dataset[0]["longest_path_length"])
    print("First maze moves:", dataset[0]["moves"])


if __name__ == "__main__":
    main()
