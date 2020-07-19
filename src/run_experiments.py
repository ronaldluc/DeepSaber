import os

os.environ['AUTOGRAPH_VERBOSITY'] = '5'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import experiments

if __name__ == '__main__':
    experiments.custom_model.main()
    experiments.baseline_model.main()
    experiments.ddc_model.main()
    experiments.best_model_comparison.main()
    experiments.hypersearch_model.main()
    experiments.information_comparison.main()
    experiments.information_comparison.mainly_id()
    experiments.information_comparison.mainly_vec()
    experiments.temperature_search.main()
