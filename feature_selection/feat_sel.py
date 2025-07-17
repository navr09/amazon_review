# Drop the vine column

import pandas as pd
import numpy as np

'''
This method of Cliques uses similar logic to maximal cliques.
Can try the logic of maximal cliques if we don't see merit this way.
'''
class CliquesFeatureSelector:
    def __init__(self, df, target, corr_thr, corr_type='pearson'):
        self.df = df
        self.target = target
        self.corr_thr = corr_thr
        self.corr_type = corr_type
        self.corr_matrix = None
        self.cliques = []
        self.no_clique = set()
        self.chosenFS = []

    def calculate_correlation_matrix(self):
        if self.corr_type == 'pearson':
            self.corr_matrix = self.df.corr().abs()
        elif self.corr_type == 'spearman':
            self.corr_matrix = self.df.corr(method='spearman').abs()
        # elif self.corr_type == 'mutual_information':
        # self.corr_matrix = self.mutual_information_matrix()

    # def mutual_information_matrix(self):
    # mi = mutual_info_classif(self.df.drop(columns=[self.target]), self.df[self.target])
    # mi_matrix = pd.DataFrame(mi, index=self.df.columns.drop(self.target), columns=[self.target])
    # return mi_matrix.abs()
    # D8, D9 didnt make the threshold cut

    # Create the initial cliques
    def create_initial_cliques(self):
        corr_mask = self.corr_matrix > self.corr_thr
        self.cliques = [set([f1, f2]) for f1 in self.corr_matrix.columns for f2 in self.corr_matrix.columns 
            if f1 != f2 and corr_mask.loc[f1, f2]]
        # {[D1,D2],[D3,D4],[D4,D6]}

    # Merge cliques that have common features.
    # Creates a graph of highly correlated cliques.
    # Joins clique features whichever is joinable, and creates 
    # list of features from just 2 features in a list to n.
    def merge_cliques(self):
        merged = True
        while merged:
            merged = False
            for i in range(len(self.cliques)):
                for j in range(i + 1, len(self.cliques)):
                    if self.cliques[i].intersection(self.cliques[j]):
                        self.cliques[i] = self.cliques[i].union(self.cliques[j])
                        del self.cliques[j]
                        merged = True
                        break
                if merged:
                    break
    # {[D1,D2],[D3, D4, D6]} : Target [D4, D2]
    # Remove target from cliques and create a set for non-clique features
    # Doing this as these features didn't meet the threshold of high correlation.
    # Hence, we add these features as is. Better way is to check these features 
    # correlation with the output feature
    def remove_target_from_cliques(self):
        self.no_clique = set(self.corr_matrix.columns) - set([self.target])
        for clique in self.cliques:
            if self.target in clique:
                clique.remove(self.target)
            self.no_clique -= clique

    # self.no_clique = D8,D9
    # Choose the feature with highest correlation to target from each clique
    def select_features(self):
        self.chosenFS = list(self.no_clique)
        for clique in self.cliques:
            max_corr_feature = self.corr_matrix.loc[list(clique), self.target].idxmax()
            print('Max correlated feature:', max_corr_feature)
            self.chosenFS.append(max_corr_feature)
        self.chosenFS.append(self.target)
        # Remove duplicates
        self.chosenFS = list(set(self.chosenFS))

    # self.no_clique = D8,D9,D4,D2
    def get_selected_features(self):
        self.calculate_correlation_matrix()
        self.create_initial_cliques()
        self.merge_cliques()
        self.remove_target_from_cliques()
        self.select_features()
        chosenFS_df = pd.DataFrame(self.chosenFS, columns=["Features"])
        return chosenFS_df
    
# selector = CliquesFeatureSelector(X, 'helpfulness_ratio', corr_thr=0.8, corr_type='pearson')
# selected_features = selector.get_selected_features()