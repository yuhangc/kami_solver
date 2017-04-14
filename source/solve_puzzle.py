from solver import PuzzleSolver


if __name__ == "__main__":
    solver = PuzzleSolver(6)
    solver.load_puzzle(99, show=True)
    solver.solve_puzzle()
