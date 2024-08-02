import numpy as np
import pandas as pd

import splitting_utils as su

from aif360.datasets import GermanDataset

dataset_m = GermanDataset()

privileged_groups = [{'sex': 1}]
unprivileged_groups = [{'sex': 0}]


sens_ind = 0

#su.double_splitting(dataset_m=dataset_m,
#                    unprivileged_groups=unprivileged_groups,
#                    privileged_groups=privileged_groups,
#                    sens_ind=sens_ind)