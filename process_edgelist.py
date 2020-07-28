import pandas as pd
from collections import Counter
import os
from scipy.stats import rankdata

# data = pd.read_csv(os.path.dirname(os.path.realpath(__file__))+"/melanoma40_groundtruth_edgelist_ES.txt", header=None)
# print(data)
# data = data[data[2] > .3]
# print(data)
# data.to_csv('melanoma40_groundtruth_edgelist_0.3.txt')

# data = pd.read_csv('TCGA_HiSeq2_10.csv')
# print(data)
# data = pd.read_csv('TCGA_Methylation_10.csv')
# c = data.iloc[:, -1]
# data = data.fillna(0)
# print(data)
# data.to_csv('TCGA_Methylation_10_temp.csv', index=0)
#
# data = pd.read_csv('TCGA_Methylation_10_temp.csv')
# print(data)

# data = pd.read_csv('HiSeq/Hiseq.csv')
# data = pd.read_csv(os.path.dirname(os.path.realpath(__file__))+"/yeast_original.csv")
# #
# print(data)
# data = data[data.p_precision < 1.13e-08]
# print(data)
#
# #
# # print(rankdata(data.p_precision))
# #
#
# data = pd.read_csv("Cytoscape_B_NW_SL.csv")
# print(data)

def fdr(p_vals):

    #ranked_p_values = rankdata(p_vals)
    drop_list = [i for i in range(len(p_vals)) if p_vals[i] > 0.1]
    #drop_list = [i for i in range(len(p_vals)) if (p_vals[i] * len(ranked_p_values) / ranked_p_values[i]) > 0.05]

    return drop_list

# drop_list = fdr(data.p_precision)
#
# # #
# dd = data.drop(index=drop_list)
# #
# print(dd)
# # # # dd = pd.read_csv('melanoma70_silggm_edgelist_P_0.05.csv', index_col=None)
# # # # print(dd)
# # #
# df = dd.iloc[:, 0:2]
#
#
# df.to_csv('yeast_silggm_p_0.1.csv', index=False, header=0)


# print(df)
# df['gene1'] = df['gene1'].str.split(r'\D').str.get(1)
#
# df['gene1'] = df['gene1'].astype(int)
#
#
# df['gene2'] = df['gene2'].str.split(r'\D').str.get(1)
# df['gene2'] = df['gene2'].astype(int)
# print(df)
# print(df['gene2'])
# #
# df = df - 1
# print(df)
#
# # # df = pd.read_csv('TCGA_HiSeq2_10_edgelist.txt', header=None)
# # # df = df.iloc[:, :2]
# # # print(df)
# matrix = df.to_numpy()
# print(matrix)
# # #
# import numpy as np
# np.savetxt("yeast_p_0.05_edgelist.txt", matrix.astype(int), fmt='%i', delimiter=' ')
# #

# d = dd.iloc[:, 1:-1]
# print(d)
# d.to_csv('OVA_Ovary_silggm.csv', header=0, index=0)
# dd = pd.read_csv('OVA_Ovary_silggm.csv')
# print(dd)
# dd.to_csv("melanoma70_silggm_edgelist_P_0.05.csv")

# df = pd.read_csv("edge_sillgm_FDR_wo.txt", names=None)
#
# print(df)

import numpy as np

rf = np.load('F1_yeast_nested1_random forest.npy')
cs = np.load('F1_yeast_nested1_cascade.npy')
window = np.load('F1_yeast_nested1_window.npy')
graph = np.load('F1_yeast_nested1_graph.npy')

rf2 = np.load('F1_yeast_nested2_random forest.npy')
cs2 = np.load('F1_yeast_nested2_cascade.npy')
window2 = np.load('F1_yeast_nested2_window.npy')
graph2 = np.load('F1_yeast_nested2_graph.npy')

rf3 = np.load('F1_yeast_nested3_random forest.npy')
cs3 = np.load('F1_yeast_nested3_cascade.npy')
window3 = np.load('F1_yeast_nested3_window.npy')
graph3 = np.load('F1_yeast_nested3_graph.npy')


rf4 = np.load('F1_yeast_nested4_random forest.npy')
cs4 = np.load('F1_yeast_nested4_cascade.npy')
window4 = np.load('F1_yeast_nested4_window.npy')
graph4 = np.load('F1_yeast_nested4_graph.npy')


rf5 = np.load('F1_yeast_nested5_random forest.npy')
cs5 = np.load('F1_yeast_nested5_cascade.npy')
window5 = np.load('F1_yeast_nested5_window.npy')
graph5 = np.load('F1_yeast_nested5_graph.npy')


rf_1 = np.mean(rf)
cs_1 = np.mean(cs)
window_1 = np.mean(window)
graph_1 = np.mean(graph)

rf_2 = np.mean(rf2)
cs_2 = np.mean(cs2)
window_2 = np.mean(window2)
graph_2 = np.mean(graph2)

rf_3 = np.mean(rf3)
cs_3 = np.mean(cs3)
window_3 = np.mean(window3)
graph_3 = np.mean(graph3)

rf_4 = np.mean(rf4)
cs_4 = np.mean(cs4)
window_4 = np.mean(window4)
graph_4 = np.mean(graph4)

rf_5 = np.mean(rf5)
cs_5 = np.mean(cs5)
window_5 = np.mean(window5)
graph_5 = np.mean(graph5)



rf = np.concatenate((rf, rf2, rf3, rf4, rf5))
cs = np.concatenate((cs, cs2, cs3, cs4, cs5))
window = np.concatenate((window, window2, window3, window4, window5))
graph = np.concatenate((graph, graph2, graph3, graph4, graph5))

np.save('f1_rf', rf)
np.save('f1_cs', cs)
np.save('f1_window', window)
np.save('f1_graph', graph)

print(np.mean(rf))

rf = (rf_1+rf_2+ rf_3+rf_4+rf_5)/5

cs = (cs_1+cs_2+cs_3+cs_4+cs_5)/5

window = (window_1+window_2+window_3+window_4+window_5)/5

graph = (graph_1+graph_2+graph_3+graph_4+graph_5)/5

print(rf)
print(cs)
print(window)
print(graph)

