import numpy as np
import pandas as pd
import os
import pickle
import matplotlib.pyplot as plt
from utils import *

rep = 100 # Replication number
round_to = 2 # Round results to decimal
scenario_list = ['S1', 'S2', 'S3'] # Scenario list
samplesize_list = ['N50', 'N500'] # Sample size list

Table_S1_Bias = np.zeros((23, 6)) # Summary table for the bias of the estimated causal effects
Table_S1_SE = np.zeros((23, 6)) # Summary table for the standard error of the estimated causal effects

for scenario in scenario_list:
    for samplesize in samplesize_list:
        print('Processing results for ANOCE: Scenario: ' + scenario + '; Sample size: ' + samplesize)
        
        column_id = len(samplesize_list) * scenario_list.index(scenario) + samplesize_list.index(samplesize)
        
        # Load results
        with open(os.path.join('ANOCE_Results', 'ANOCE_Simu_' + scenario + '_Gauss_' + samplesize + '.data'), 'rb') as data:
            data = pickle.load(data)
        
        # Load true graph
        with open(os.path.join('True_Graphs', scenario + '_trueG.pkl'), 'rb') as true_G:
            true_G = pickle.load(true_G)
    
        # Calculate the true causal effects
        true_B = nx.to_numpy_array(true_G)
        d = true_B.shape[0]
        true_TE, true_DE, true_IE, true_DM, true_IM = calculate_effect(true_B)
        
        # Calculate the estimated causal effects for each replication
        TE_BS = np.zeros((rep))
        DE_BS = np.zeros((rep))
        IE_BS = np.zeros((rep))
        DM_BS = np.zeros((rep, d - 2))
        IM_BS = np.zeros((rep, d - 2))
        all_Bs = np.zeros((rep, d, d))
        
        for k in range(rep):
            all_Bs[k, :, :] = np.squeeze(np.asarray(data[k][0]))
            
            # Estimated causal effects
            TE_BS[k], DE_BS[k], IE_BS[k], DM_BS[k], IM_BS[k] = calculate_effect(all_Bs[k, :, :])
        
        print('Bias for TE:')
        Table_S1_Bias[0, column_id] = np.mean(TE_BS) - true_TE
        print(Table_S1_Bias[0, column_id])
        print('SE for TE:')
        Table_S1_SE[0, column_id] = np.std(TE_BS) / np.sqrt(rep - 1)
        print(Table_S1_SE[0, column_id])
        
        print('Bias for DE:')
        Table_S1_Bias[1, column_id] = np.mean(DE_BS) - true_DE
        print(Table_S1_Bias[1, column_id])
        print('SE for DE:')
        Table_S1_SE[1, column_id] = np.std(DE_BS) / np.sqrt(rep - 1)
        print(Table_S1_SE[1, column_id])
        
        print('Bias for IE:')
        Table_S1_Bias[2, column_id] = np.mean(IE_BS) - true_IE
        print(Table_S1_Bias[2, column_id])
        print('SE for IE:')
        Table_S1_SE[2, column_id] = np.std(IE_BS) / np.sqrt(rep - 1)
        print(Table_S1_SE[2, column_id])
        
        print('Bias for DM:')
        Table_S1_Bias[3:13, column_id] = np.mean(DM_BS, 0) - true_DM
        print(Table_S1_Bias[3:13, column_id])
        print('SE for DM:')
        Table_S1_SE[3:13, column_id] = np.std(DM_BS, 0) / np.sqrt(rep - 1)
        print(Table_S1_SE[3:13, column_id])
        
        print('Bias for IM:')
        Table_S1_Bias[13:, column_id] = np.mean(IM_BS, 0) - true_IM
        print(Table_S1_Bias[13:, column_id])
        print('SE for DM:')
        Table_S1_SE[13:, column_id] = np.std(IM_BS, 0) / np.sqrt(rep - 1)
        print(Table_S1_SE[13:, column_id])
        
        # Calculate the averaged estimated weighted matrix B
        average_B = np.mean(all_Bs, 0)
        
        plt.matshow(average_B.T, cmap = 'bwr', vmin = -1, vmax = 1)
        fig1 = plt.gcf()
        plt.colorbar()
        plt.show()
        fig1.savefig('Figures/MATplot_ANOCE_' + scenario + '_Gauss_' + samplesize + '.pdf')

np.savetxt("Table_S1_Bias.csv", Table_S1_Bias, delimiter = ",")
np.savetxt("Table_S1_SE.csv", Table_S1_SE, delimiter = ",")
