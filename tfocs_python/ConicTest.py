import unittest
import ConicSolver as CS
import Functions

def main():
    solver = CS.ConicSolver()
    print(solver.test_method())
    print(solver)

class TestConicSolver(unittest.TestCase):
    def test_init(self):
        pass

    def test_lasso(self):

        smooth_quad = None # TODO
        solver = CS.ConicSolver(smooth_quad)



main()