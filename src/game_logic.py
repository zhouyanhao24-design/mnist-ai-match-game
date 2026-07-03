from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np


@dataclass
class Tile:
    tile_id: int
    true_label: int
    image: np.ndarray
    row: int
    col: int
    predicted_label: int | None = None
    removed: bool = False
    in_tray: bool = False


class MatchGameLogic:
    def __init__(self, tray_size: int = 7) -> None:
        self.tray_size = tray_size
        self.tray: List[Tile] = []
        self.moves = 0
        self.matches = 0
        self.game_over = False
        self.win = False
        self.message = "Click a tile to classify it."

    def can_select(self, tile: Tile) -> bool:
        return not tile.removed and not tile.in_tray and not self.game_over

    def select_tile(self, tile: Tile, predicted_label: int, remaining_tiles: int) -> None:
        if not self.can_select(tile):
            return
        tile.predicted_label = int(predicted_label)
        tile.in_tray = True
        self.tray.append(tile)
        self.moves += 1
        self.message = f"Predicted digit: {predicted_label}"
        self.remove_triples()
        self.update_state(remaining_tiles)

    def remove_triples(self) -> None:
        counts: dict[int, list[Tile]] = {}
        for tile in self.tray:
            if tile.predicted_label is None:
                continue
            counts.setdefault(tile.predicted_label, []).append(tile)

        for label, group in counts.items():
            while len(group) >= 3:
                removed_group = group[:3]
                group = group[3:]
                for tile in removed_group:
                    tile.removed = True
                    tile.in_tray = False
                    if tile in self.tray:
                        self.tray.remove(tile)
                self.matches += 1
                self.message = f"Matched three digit {label} tiles."
            counts[label] = group

    def update_state(self, remaining_tiles: int) -> None:
        if remaining_tiles == 0 and len(self.tray) == 0:
            self.game_over = True
            self.win = True
            self.message = "You win! All tiles are cleared."
        elif len(self.tray) >= self.tray_size:
            self.game_over = True
            self.win = False
            self.message = "Game over! The tray is full."

    def reset(self) -> None:
        self.tray.clear()
        self.moves = 0
        self.matches = 0
        self.game_over = False
        self.win = False
        self.message = "Click a tile to classify it."
