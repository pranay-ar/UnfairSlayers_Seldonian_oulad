# createSpec.py
import autograd.numpy as np
import os
from seldonian.parse_tree.parse_tree import (ParseTree,
                                             make_parse_trees_from_constraints)

from seldonian.dataset import DataSetLoader
from seldonian.utils.io_utils import (load_json, save_pickle)
from seldonian.spec import SupervisedSpec
from seldonian.models.models import (
    BinaryLogisticRegressionModel as LogisticRegressionModel)
from seldonian.models import objectives

import sys

if __name__ == '__main__':
    attr_vals = {'gender':('M','F'),'disability':('N','Y'),'higher_education':('he0','he1')}
    data_pth = "studentInfoconverted.csv"
    metadata_pth = "metadata_studentInfo.json"
    save_dir = '.'
    os.makedirs(save_dir, exist_ok=True)
    # Create dataset from data and metadata file
    regime = 'supervised_learning'
    sub_regime = 'classification'

    attr_arg = sys.argv[1]
    constraint_arg = sys.argv[2]

    loader = DataSetLoader(
        regime=regime)

    dataset = loader.load_supervised_dataset(
        filename=data_pth,
        metadata_filename=metadata_pth,
        file_type='csv')
    sensitive_col_names = dataset.meta_information['sensitive_col_names']

    # Use logistic regression model
    model = LogisticRegressionModel()

    # Set the primary objective to be log loss
    primary_objective = objectives.binary_logistic_loss

    # Define behavioral constraints
    # Change this string to implement the constraint

    if constraint_arg == 'disp':
        constraint_strs = [
            f'min((PR | [{attr_vals[attr_arg][0]}])/(PR | [{attr_vals[attr_arg][1]}]),(PR | [{attr_vals[attr_arg][1]}])/(PR | [{attr_vals[attr_arg][0]}])) >= 0.9']
    elif constraint_arg == 'eq':
        constraint_strs = [
            f'abs((FNR | [{attr_vals[attr_arg][0]}]) - (FNR | [{attr_vals[attr_arg][1]}])) + abs((FPR | [{attr_vals[attr_arg][0]}]) - (FPR | [{attr_vals[attr_arg][1]}])) <= 0.1']
    deltas = [0.05]

    # For each constraint (in this case only one), make a parse tree
    parse_trees = make_parse_trees_from_constraints(
        constraint_strs, deltas, regime=regime,
        sub_regime=sub_regime, columns=sensitive_col_names)

    # Save spec object, using defaults where necessary
    spec = SupervisedSpec(
        dataset=dataset,
        model=model,
        parse_trees=parse_trees,
        sub_regime=sub_regime,
        frac_data_in_safety=0.6,
        primary_objective=primary_objective,
        initial_solution_fn=model.fit,
        use_builtin_primary_gradient_fn=True,
        optimization_technique='gradient_descent',
        optimizer='adam',
        optimization_hyperparams={
            'lambda_init': np.array([0.5]),
            'alpha_theta': 0.01,
            'alpha_lamb': 0.01,
            'beta_velocity': 0.9,
            'beta_rmsprop': 0.95,
            'use_batches': False,
            'num_iters': 1500,
            'gradient_library': "autograd",
            'hyper_search': None,
            'verbose': True,
        }
    )

    spec_save_name = os.path.join(save_dir, f'./specs/spec_{attr_arg}_{constraint_arg}.pkl')
    save_pickle(spec_save_name, spec)
    print(f"Saved Spec object to: {spec_save_name}")
