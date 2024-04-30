from scipy.stats import spearmanr


def LDS(preds, outputs):
    """
    Computes the Linear Datamodeling Score (LDS) of a given
    context attribution method.

    Arguments:
        TBD

    Returns:
        lds (float): Linear Datamodeling Score
    """
    lds = spearmanr(preds, outputs).statistic
    return lds


def TopKRemoval():
    """
    Computes the drop in log-likelihood of the selected response when we remove
    the top-k most important (according to a given context attribution method)
    sources from the context.

    Arguments:
        TBD

    Returns:
        drop (float): Drop in log-likelihood
    """
    # TODO
    ...
