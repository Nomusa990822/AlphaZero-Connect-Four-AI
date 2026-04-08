"""
Board utilities for Connect Four.

The board is represented as a NumPy array of shape (ROWS, COLS).

Values:
    0  -> empty
    1  -> player one
   -1  -> player two

Rows are indexed from top (0) to bottom (ROWS - 1).
When a piece is dropped into a column, it occupies the lowest available row.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from src.core.constants import COLS, EMPTY, PLAYER_ONE, PLAYER_TWO, ROWS, VALID_PLAYERS


@dataclass
class Board:
    """
    Encapsulates the Connect Four board state and piece placement logic.
    """

    grid: np.ndarray = field(default_factory=lambda: np.zeros((ROWS, COLS), dtype=np.int8))

    def __post_init__(self) -> None:
        """
        Validate the provided board shape and values.
        """
        if not isinstance(self.grid, np.ndarray):
            raise TypeError("grid must be a NumPy array.")

        if self.grid.shape != (ROWS, COLS):
            raise ValueError(f"grid must have shape ({ROWS}, {COLS}).")

        valid_values = {EMPTY, PLAYER_ONE, PLAYER_TWO}
        unique_values = set(np.unique(self.grid))
        if not unique_values.issubset(valid_values):
            raise ValueError(
                f"grid contains invalid values: {unique_values - valid_values}. "
                f"Allowed values are {valid_values}."
            )

        self.grid = self.grid.astype(np.int8)

    def copy(self) -> "Board":
        """
        Return a deep copy of the board.
        """
        return Board(self.grid.copy())

    def reset(self) -> None:
        """
        Clear the board to all-empty cells.
        """
        self.grid.fill(EMPTY)

    def is_valid_column(self, col: int) -> bool:
        """
        Return True if the column index is within range.
        """
        return 0 <= col < COLS

    def is_column_full(self, col: int) -> bool:
        """
        Return True if a piece cannot be dropped into the column.
        """
        if not self.is_valid_column(col):
            raise ValueError(f"Column {col} is out of bounds.")
        return self.grid[0, col] != EMPTY

    def get_valid_moves(self) -> List[int]:
        """
        Return a list of columns that can still accept a piece.
        """
        return [col for col in range(COLS) if not self.is_column_full(col)]

    def get_next_open_row(self, col: int) -> Optional[int]:
        """
        Return the row index where the next piece would land in the given column.
        Return None if the column is full.
        """
        if not self.is_valid_column(col):
            raise ValueError(f"Column {col} is out of bounds.")

        for row in range(ROWS - 1, -1, -1):
            if self.grid[row, col] == EMPTY:
                return row
        return None

    def drop_piece(self, col: int, player: int) -> int:
        """
        Drop a player's piece into the specified column.

        Args:
            col: Target column.
            player: PLAYER_ONE or PLAYER_TWO.

        Returns:
            The row index where the piece landed.

        Raises:
            ValueError: If player is invalid, column is invalid, or column is full.
        """
        if player not in VALID_PLAYERS:
            raise ValueError(f"Invalid player value: {player}.")

        if not self.is_valid_column(col):
            raise ValueError(f"Column {col} is out of bounds.")

        row = self.get_next_open_row(col)
        if row is None:
            raise ValueError(f"Column {col} is full.")

        self.grid[row, col] = player
        return row

    def is_full(self) -> bool:
        """
        Return True if the board has no remaining legal moves.
        """
        return not np.any(self.grid == EMPTY)

    def to_list(self) -> List[List[int]]:
        """
        Return the board as a nested Python list.
        """
        return self.grid.tolist()

    def __getitem__(self, key):
        """
        Allow direct indexing into the underlying NumPy grid.
        """
        return self.grid[key]

    def __str__(self) -> str:
        """
        Pretty string view of the board for debugging.
        """
        symbol_map = {
            PLAYER_ONE: "X",
            PLAYER_TWO: "O",
            EMPTY: ".",
        }
        rows = []
        for row in self.grid:
            rows.append(" ".join(symbol_map[int(cell)] for cell in row))
        return "\n".join(rows)