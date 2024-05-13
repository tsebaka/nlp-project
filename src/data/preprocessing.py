import sklearn
from sklearn.model_selection import StratifiedKFold


def make_folds(
    config,
    dataframe,
):
    kfold = StratifiedKFold(n_splits=config.main.n_folds, shuffle=True, random_state=config.main.seed)

    for n, (_, val_index) in enumerate(kfold.split(dataframe, dataframe[config.main.target_col])):
        dataframe.loc[val_index, "fold"] = int(n)
    dataframe["fold"] = dataframe["fold"].astype(int)
    
    return dataframe