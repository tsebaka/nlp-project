import sklearn
from sklearn.metrics import cohen_kappa_score


def quadratic_weighted_kappa(
    y_pred,
    y_true,
    task="regression"
):
    if task == "regression":
        return cohen_kappa_score(
            y_true.detach().cpu().numpy().astype(int),
            y_pred.detach().cpu().numpy().clip(0, 5).round(0),
            weights='quadratic'
        )
    else:
        return cohen_kappa_score(
            y_pred,
            y_true,
            weights="quadratic"
        )