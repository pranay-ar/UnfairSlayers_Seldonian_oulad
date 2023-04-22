# loan fairness
import os

from seldonian.seldonian_algorithm import SeldonianAlgorithm
from seldonian.utils.io_utils import load_pickle

import sys

if __name__ == '__main__':
    # Load loan spec file
    attr_arg = sys.argv[1]
    constraint_arg = sys.argv[2]

    specfile = f'./specs/spec_{attr_arg}_{constraint_arg}.pkl'
    spec = load_pickle(specfile)
    SA = SeldonianAlgorithm(spec)
    passed_safety, solution = SA.run(write_cs_logfile=True)
    if passed_safety:
        print("Passed safety test!")
    else:
        print("Failed safety test")
    print()
    print("Primary objective (log loss) evaluated on safety dataset:")
    print(SA.evaluate_primary_objective(branch='safety_test',theta=solution))

