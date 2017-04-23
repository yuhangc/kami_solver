from solver import PuzzleSolver


if __name__ == "__main__":
    solver = PuzzleSolver(13)
    solver.load_puzzle(106, puzzle_suf="", show=True)

    # set parameters
    solver.set_solver_param("stick_to_one_patch", True)
    solver.set_solver_param("STOP_start_step", 2)

    solver.solve_puzzle()
