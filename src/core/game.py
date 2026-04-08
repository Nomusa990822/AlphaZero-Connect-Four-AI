"""
High-level Connect Four game environment.

This class manages:
- the board
- current player
- move application
- terminal checks
- winner tracking
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from src.core.board import Board
from src.core.constants import DRAW, PLAYER_ONE, PLAYER_TWO, VALID_PLAYERS
from src.core.rules import check_draw, check_winner, get_game_result, is_terminal_state


@dataclass
class ConnectFourGame:
    """
    Main game environment for Connect Four.
    """

    board: Board = field(default_factory=Board)
    current_player: int = PLAYER_ONE
    winner: Optional[int] = None
    done: bool = False
    move_count: int = 0
    last_move: Optional[Dict[str, int]] = None

    def __post_init__(self) -> None:
        """
        Validate the starting player.
        """
        if self.current_player not in VALID_PLAYERS:
            raise ValueError(f"Invalid current_player: {self.current_player}.")

        self._refresh_game_state()

    def reset(self) -> None:
        """
        Reset the game to its initial state.
        """
        self.board.reset()
        self.current_player = PLAYER_ONE
        self.winner = None
        self.done = False
        self.move_count = 0
        self.last_move = None

    def copy(self) -> "ConnectFourGame":
        """
        Return a deep copy of the game state.
        """
        copied_game = ConnectFourGame(
            board=self.board.copy(),
            current_player=self.current_player,
            winner=self.winner,
            done=self.done,
            move_count=self.move_count,
            last_move=None if self.last_move is None else dict(self.last_move),
        )
        return copied_game

    def get_valid_moves(self) -> List[int]:
        """
        Return the list of currently legal columns.
        """
        return self.board.get_valid_moves()

    def is_valid_move(self, col: int) -> bool:
        """
        Return True if the move is legal in the current state.
        """
        return col in self.get_valid_moves()

    def switch_player(self) -> None:
        """
        Switch the current player.
        """
        self.current_player = PLAYER_TWO if self.current_player == PLAYER_ONE else PLAYER_ONE

    def apply_move(self, col: int) -> Dict[str, Optional[int]]:
        """
        Apply a move for the current player.

        Args:
            col: Column to play.

        Returns:
            A dictionary containing move information:
            {
                "row": landed row,
                "col": played column,
                "player": player who moved,
                "winner": winner or None,
                "is_draw": bool,
                "done": bool
            }

        Raises:
            ValueError: If the game is already over or the move is invalid.
        """
        if self.done:
            raise ValueError("Cannot apply a move: the game is already over.")

        if not self.is_valid_move(col):
            raise ValueError(f"Invalid move: column {col} is not playable.")

        player = self.current_player
        row = self.board.drop_piece(col, player)
        self.move_count += 1
        self.last_move = {"row": row, "col": col, "player": player}

        if check_winner(self.board, player):
            self.winner = player
            self.done = True
        elif check_draw(self.board):
            self.winner = DRAW
            self.done = True
        else:
            self.switch_player()

        return {
            "row": row,
            "col": col,
            "player": player,
            "winner": self.winner if self.winner in VALID_PLAYERS else None,
            "is_draw": self.winner == DRAW,
            "done": self.done,
        }

    def _refresh_game_state(self) -> None:
        """
        Recalculate terminal information from the board state.
        Useful when a board is passed into the constructor.
        """
        result = get_game_result(self.board)
        if result is None:
            self.winner = None
            self.done = False
        else:
            self.winner = result
            self.done = True

    def is_terminal(self) -> bool:
        """
        Return True if the game is finished.
        """
        return is_terminal_state(self.board)

    def get_result(self) -> Optional[int]:
        """
        Return:
            PLAYER_ONE / PLAYER_TWO / DRAW / None
        """
        return get_game_result(self.board)

    def get_state(self) -> dict:
        """
        Return a serializable snapshot of the game state.
        """
        return {
            "board": self.board.to_list(),
            "current_player": self.current_player,
            "winner": self.winner,
            "done": self.done,
            "move_count": self.move_count,
            "last_move": self.last_move,
            "valid_moves": self.get_valid_moves(),
        }

    def render(self) -> None:
        """
        Print a simple text view of the board.
        """
        print(self.board)

    def __str__(self) -> str:
        return str(self.board)