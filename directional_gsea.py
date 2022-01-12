import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle, gzip
from sklearn.preprocessing import StandardScaler
import gseapy as gp

entrez_to_symbol = {}
symbol_to_entrez = {}
df = pd.read_csv('df', header = 0, index_col = 0, compression='gzip')
cms_cmap = {'CMS1':'tab:green', 'CMS2_1':'tab:red', 'CMS2_2':'tab:purple', 'CMS3':'tab:brown', \
            'CMS4_1':'tab:pink', 'CMS4_2':'tab:olive', 'CMS4_3':'tab:cyan'}
directional_pearson_correlation = pd.read_csv('directional_pearson_correlation.csv', header = 0, index_col = 0)
directional_pearson_correlation.columns = [float(x) for x in directional_pearson_correlation.columns]
directional_pearson_correlation.index = [str(x) for x in directional_pearson_correlation.index]

with gzip.open('biomart.txt.gz', 'rt') as fin:
    fin.readline()
 
    for line in fin.readlines():
        content = line.strip().split('\t')

        if len(content) > 3:
            if len(content[2]) * len(content[3]) > 0: ## entrez id exists
                entrez_to_symbol[content[3]] = content[2]
                symbol_to_entrez[content[2]] = content[3]
                
def plot_directional_gsea(gsea, term, fdr = 0.05, linewidth = 4, display = False):
    if display:
        print([entrez_to_symbol[x] for x in gsea[directional_pearson_correlation.columns.values[0]].res2d.loc[term, 'genes'].split(';')])
    
    plt.figure(figsize = (8, 4))
    plt.subplot(1, 2, 1, polar = True)    
    
    for cms in cms_cmap.keys():
        plt.scatter(df.loc[df['new_cms'] == cms, 'densMAP_62_angle'], \
                    df.loc[df['new_cms'] == cms, 'densMAP_62_radius'], c=cms_cmap[cms], s = 4, label = cms)
    
    plt.legend(markerscale = 2)
    index = 0
    
    while index < directional_pearson_correlation.shape[1]:
        angle = directional_pearson_correlation.columns.values[index]
        
        if gsea[angle].res2d.loc[term, 'fdr'] <= fdr and gsea[angle].res2d.loc[term, 'nes'] > 0:
            start_angle = angle
            angles = [angle]
            end = index + 1

            while end < directional_pearson_correlation.shape[1] and \
                  gsea[directional_pearson_correlation.columns.values[end]].res2d.loc[term, 'fdr'] <= fdr and \
                  gsea[directional_pearson_correlation.columns.values[end]].res2d.loc[term, 'nes'] > 0:
                angles.append(directional_pearson_correlation.columns.values[end])
                end += 1
    
            if len(angles) == 1:
                plt.plot([angles[0] - np.pi/25, angles[0] + np.pi/25], [10, 10], linewidth = linewidth, c = 'tab:blue')
            else:
                plt.plot(angles, [10] * len(angles), linewidth = linewidth, c = 'tab:blue')
            
            index = end
        
        else:
            index += 1
    
    plt.title(term)
    
    plt.subplot(1, 2, 2, polar = True)
    
    genes = gsea[directional_pearson_correlation.columns.values[0]].res2d.loc[term, 'genes'].split(';')
    mean = df[genes].mean(axis=1).to_numpy().reshape([-1, 1])
    values = StandardScaler().fit_transform(mean)
    cm = plt.cm.get_cmap('bwr')
    plt.scatter(df['densMAP_62_angle'], df['densMAP_62_radius'], c=values, \
                cmap=cm, vmin=-4, vmax=4, alpha=0.8, s = 2)

    plt.tight_layout()
    plt.show()
    
def directional_gsea(term , target_genes, fdr = 0.05, linewidth = 4, display = True):
    gsea_results = pickle.load(open('gsea_custom_genes_103121.pkl', 'rb'))
    if term in gsea_results[0.06283185307179529].res2d.index:
        print(term)
        print(f"Given term ({term}) is in precompute geneset. Using precompute data.")
    else:
        print(f"Compute {term}")
        gsea_results = dict()
        target_genes = [x for x in df.columns if x in entrez_to_symbol and entrez_to_symbol[x] in target_genes]
        gsea_genesets = {term:target_genes}
        gsea_genesets_symbols = {}
        for s in gsea_genesets:
            gsea_genesets_symbols[s] = [entrez_to_symbol[g] for g in gsea_genesets[s]]
        for angle in directional_pearson_correlation.columns.values:
            temp = pd.concat([pd.DataFrame(directional_pearson_correlation.index.values, index = directional_pearson_correlation.index), directional_pearson_correlation[angle]], axis = 1)
            gsea_results[angle] = gp.prerank(rnk = temp, gene_sets = gsea_genesets,
                                             processes = 4,
                                             permutation_num = 1000,
                                             outdir=None,
                                             format = 'png', seed = 4649)
    plot_directional_gsea(gsea_results, term, fdr = fdr, linewidth = linewidth, display = display)