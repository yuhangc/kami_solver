from solver import PuzzleSolver


if __name__ == "__main__":
    solver = PuzzleSolver(13)
    solver.load_puzzle(106, puzzle_suf="", show=True)
    solver.solve_puzzle()
