import numpy as np
import pandas as pd

class TestSpec:
    '''Class that defines specifications for inter-model tests'''

    def __init__(self, indices, start_dates, window_size=400, margin=7):
        # Test sets
        if not isinstance(indices,pd.DatetimeIndex):
            raise TypeError('indices param must be of type pandas.DatetimeIndex')

        if not isinstance(start_dates, list):
            raise TypeError('start_dates param must be a list of strings')

        if not isinstance(window_size, int):
            raise TypeError('window_size param must be an integer')

        # Sort values of start/end and indices
        start_dates.sort()
        indices = indices.sort_values()

        # Save start dates
        self.start_dates = start_dates

        # Margin of test
        self.margin = margin

        # List of test instances (test set + CV sets + FV sets)
        self.instance = []
        n_tests = len(start_dates)
        for i in range(n_tests-1):
            self.instance.append(self.TestInstance(indices, start_dates[i], start_dates[i+1],
                                                   window_size, margin))
        self.instance.append(self.TestInstance(indices, start_dates[-1], None, window_size, margin))

    class TestInstance:
        def __init__(self, indices, start_date, end_date, window_size, margin):
            test_indices = indices[indices >= start_date]
            if end_date is not None:
                test_indices = test_indices[test_indices < end_date]

            train_val_indices_expanding = indices[indices < start_date]
            train_val_indices_sliding = train_val_indices_expanding[-window_size:]

            self.train_set = indices[indices < start_date]
            self.test_set = test_indices
            self.expanding_window_fv =  self.ForwardValidation(train_val_indices_expanding)
            self.expanding_window_cv =  self.CrossValidation(train_val_indices_expanding, margin)
            self.sliding_window_fv =  self.ForwardValidation(train_val_indices_sliding)
            self.sliding_window_cv =  self.CrossValidation(train_val_indices_sliding, margin)

        class CrossValidation:
            '''Class that defines the folds of Blocked Cross Validation for a test instance'''
            n_folds = 5
            n_blocks_per_fold = 2

            def __init__(self, train_val_inds, margin):
                k = self.n_folds
                n_blocks_per_k = self.n_blocks_per_fold
                n_purge = margin

                n_blocks = k*n_blocks_per_k
                inds = train_val_inds
                n_inds = len(inds)

                # Split into consecutive n_blocks
                p = 0
                block_inds = []
                for i in range(n_blocks):
                    nb = int(np.ceil((n_inds-p)/(n_blocks-i)))
                    block_inds.append(np.arange(p,min(p+nb, n_inds)))
                    p += nb

                # Random order
                block_ord = np.random.permutation(n_blocks).reshape(5,2)

                # Select n_blocks_per_k
                self.train_sets = []
                self.val_sets = []
                proportion = float(1/n_blocks)
                val_purge = int(np.ceil(proportion*n_purge)) # margin to be removed from val set
                for i in range(k):
                    select_array = np.full(n_inds, fill_value=1) # 0: purge / 1: train / 2:val
                    if abs(block_ord[i,0]-block_ord[i,1]) == 1: # no need to purge between the sets
                        purge = False
                        if block_ord[i,0]-block_ord[i,1] == -1: # which block comes first
                            first = 0
                        else:
                            first = 1
                    else:
                        purge = True

                    for j in range(n_blocks_per_k):
                        # Select validation and purge indices
                        val_inds = block_inds[block_ord[i,j]]
                        purge_inds_before = np.arange(0)
                        purge_inds_after = np.arange(0)
                        if (purge==True or j==first) and val_inds[0] != 0:
                            val_inds = val_inds[val_purge:]  # purge before block
                            purge_inds_before = np.arange(max(0,val_inds[0]-n_purge), val_inds[0])
                        if (purge==True or j!=first) and val_inds[-1] != (n_inds-1):
                            val_inds = val_inds[:-val_purge] # purge after block
                            purge_inds_after= np.arange(val_inds[-1]+1, min(n_inds,val_inds[-1]+n_purge+1))

                        select_array[val_inds] = 2
                        select_array[purge_inds_before] = 0
                        select_array[purge_inds_after] = 0

                    self.train_sets.append(inds[select_array==1])
                    self.val_sets.append(inds[select_array==2])

                for i in range(k):
                    assert len(np.intersect1d(self.train_sets[i],self.val_sets[i])) is 0
                    for j in range(i+1, k):
                        assert len(np.intersect1d(self.val_sets[i],self.val_sets[j])) is 0


        class ForwardValidation:
            '''Class that defines the Forward Validation sets for a test instance'''
            val_test_ratio = 0.2
            n_validation_sets = 5
            purge_size = 0

            def __init__(self, train_val_inds):
                n = self.n_validation_sets
                ratio = self.val_test_ratio
                n_purge = self.purge_size

                inds = train_val_inds
                n_inds = len(inds)

                # Vector with the ratios (train+val)/train for each validation (e.g., [1, 1.2, ..., 1.2])
                ratio_vec = np.r_[1, np.full(n-1, fill_value = 1+ratio)]

                # Percentage of samples to be used in first train set so that the ratio can be kept
                perc_samples = 1 / (1 + ratio * np.sum(np.cumprod(ratio_vec)))

                # Final indices of each training set
                train_inds = np.round(n_inds * perc_samples * np.cumprod(ratio_vec)).astype(int)
                train_inds = np.r_[train_inds, n_inds]

                # Defines each training set
                self.train_sets = []
                self.val_sets = []
                val_purge = int(np.ceil(ratio*n_purge)) # margin to be removed from val set
                train_purge = n_purge - val_purge
                for i in range(n):
                    self.train_sets.append(inds[:train_inds[i]-train_purge])
                    self.val_sets.append(inds[train_inds[i]+val_purge:train_inds[i+1]])

                return
