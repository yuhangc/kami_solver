from solver import PuzzleSolver


if __name__ == "__main__":
    solver = PuzzleSolver(13, 5)
    solver.load_puzzle(106, down_sample=True, show=True)
