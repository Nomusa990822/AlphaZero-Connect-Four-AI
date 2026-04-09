from src.agents.alphazero_agent import AlphaZeroAgent
from src.agents.heuristic_agent import HeuristicAgent
from src.agents.minimax_agent import MinimaxAgent
from src.agents.random_agent import RandomAgent
from src.evaluation.arena import Arena
from src.evaluation.baseline_matches import (
    evaluate_against_heuristic,
    evaluate_against_minimax,
    evaluate_against_random,
)
from src.evaluation.metrics import summarize_results
from src.evaluation.tournament import Tournament
from src.neural.network import AlphaZeroNet


def test_arena_play_game_returns_match_result():
    arena = Arena()
    agent_a = RandomAgent(seed=42, name="RandomA")
    agent_b = RandomAgent(seed=43, name="RandomB")

    result = arena.play_game(agent_a, agent_b)

    assert result.winner in (1, -1, 0)
    assert result.num_moves > 0
    assert result.agent_player_one_name == "RandomA"
    assert result.agent_player_two_name == "RandomB"


def test_arena_play_games_returns_correct_count():
    arena = Arena()
    agent_a = RandomAgent(seed=42, name="RandomA")
    agent_b = HeuristicAgent(name="HeuristicB")

    results = arena.play_games(agent_a, agent_b, num_games=4, alternate_starts=True)

    assert len(results) == 4


def test_summarize_results_keys():
    arena = Arena()
    agent_a = RandomAgent(seed=42, name="RandomA")
    agent_b = RandomAgent(seed=43, name="RandomB")

    results = arena.play_games(agent_a, agent_b, num_games=4)
    summary = summarize_results(results, agent_a.name, agent_b.name)

    expected_keys = {
        "games",
        "wins",
        "losses",
        "draws",
        "win_rate",
        "loss_rate",
        "draw_rate",
        "average_game_length",
    }

    assert set(summary.keys()) == expected_keys
    assert summary["games"] == 4


def test_summary_counts_add_up():
    arena = Arena()
    agent_a = RandomAgent(seed=42, name="RandomA")
    agent_b = RandomAgent(seed=43, name="RandomB")

    results = arena.play_games(agent_a, agent_b, num_games=6)
    summary = summarize_results(results, agent_a.name, agent_b.name)

    assert summary["wins"] + summary["losses"] + summary["draws"] == summary["games"]


def test_tournament_runs():
    arena = Arena()
    tournament = Tournament(arena=arena)

    agents = [
        RandomAgent(seed=1, name="Random1"),
        HeuristicAgent(name="Heuristic"),
        MinimaxAgent(depth=2, seed=2, name="Minimax2"),
    ]

    standings = tournament.run(agents, games_per_pairing=2)

    assert isinstance(standings, list)
    assert len(standings) > 0
    assert all("agent" in row for row in standings)
    assert all("opponent" in row for row in standings)


def test_alphazero_agent_selects_valid_move():
    model = AlphaZeroNet()
    agent = AlphaZeroAgent(model=model, simulations=10, name="AZ")
    from src.core.game import ConnectFourGame

    game = ConnectFourGame()
    move = agent.select_move(game)

    assert move in game.get_valid_moves()


def test_baseline_evaluation_helpers_run():
    model = AlphaZeroNet()
    agent = AlphaZeroAgent(model=model, simulations=8, name="AZ")

    random_summary = evaluate_against_random(agent, num_games=2, seed=42)
    heuristic_summary = evaluate_against_heuristic(agent, num_games=2)
    minimax_summary = evaluate_against_minimax(agent, num_games=2, depth=2, seed=42)

    for summary in (random_summary, heuristic_summary, minimax_summary):
        assert "games" in summary
        assert "wins" in summary
        assert "losses" in summary
        assert "draws" in summary
        assert summary["games"] == 2
