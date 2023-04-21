import os
import numpy as np 

from experiments.generate_plots import SupervisedPlotGenerator
from seldonian.utils.io_utils import load_pickle
from sklearn.metrics import log_loss

if __name__ == "__main__":
    # Parameter setup
    run_experiments = True
    make_plots = True
    save_plot = False
    constraint_name = 'disparate_impact'
    performance_metric = 'log_loss'
    n_trials = 50
    data_fracs = np.logspace(-3,0,15)
    n_workers = 8
    verbose=True
    results_dir = f'results/oulad_{constraint_name}_seldo_log_loss_0.85'
    os.makedirs(results_dir,exist_ok=True)

    plot_savename = os.path.join(results_dir,f'{constraint_name}_{performance_metric}.png')

    specfile = f'./spec.pkl'
    spec = load_pickle(specfile)

    dataset = spec.dataset
    test_features = dataset.features
    test_labels = dataset.labels 

    def perf_eval_fn(y_pred,y,**kwargs):
        if performance_metric == 'log_loss':
            return log_loss(y,y_pred)

    perf_eval_kwargs = {
        'X':test_features,
        'y':test_labels,
        }
    
    plot_generator = SupervisedPlotGenerator(
        spec=spec,
        n_trials=n_trials,
        data_fracs=data_fracs,
        n_workers=n_workers,
        datagen_method='resample',
        perf_eval_fn=perf_eval_fn,
        constraint_eval_fns=[],
        results_dir=results_dir,
        perf_eval_kwargs=perf_eval_kwargs,
        )
    
    if run_experiments:
        plot_generator.run_baseline_experiment(
            model_name='random_classifier',verbose=verbose)

        plot_generator.run_baseline_experiment(
            model_name='logistic_regression',verbose=verbose)

    plot_generator.run_seldonian_experiment(verbose=verbose)

    if make_plots:
        if save_plot:
            plot_generator.make_plots(fontsize=12,legend_fontsize=8,
                performance_label=performance_metric,
                performance_yscale='log',
                savename=plot_savename)
        else:
            plot_generator.make_plots(fontsize=12,legend_fontsize=8,
                performance_label=performance_metric,
                performance_yscale='log')