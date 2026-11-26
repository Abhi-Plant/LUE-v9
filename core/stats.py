"""Small statistics helpers: weighted quantiles, area-weighted means and validation metrics."""
import numpy as np


def wquantile(x, w, q):
    """Weighted quantile (inverse of the weighted empirical CDF, Hyndman & Fan type 1)."""
    x, w = np.asarray(x, float), np.asarray(w, float)
    o = np.argsort(x)
    x, cw = x[o], np.cumsum(w[o]) / w.sum()
    return np.array([x[np.searchsorted(cw, qq - 1e-12)] for qq in np.atleast_1d(q)])


def area_mean(values, area):
    v, a = np.asarray(values, float), np.asarray(area, float)
    return float((v * a).sum() / a.sum())


def metrics(obs, pred):
    """Validation statistics (e = predicted - observed): n, bias = mean(e), MAE, RMSE, NRMSE = RMSE / mean(observed)
    x 100, Willmott (1981) index of agreement d, and two R2:
      r2_pearson        squared Pearson correlation of observed and predicted (pattern only; ignores bias)
      r2_determination  1 - sum(e^2) / sum((o - mean(o))^2) (coefficient of determination of the predictions, 1:1
                        line; negative when the predictions are worse than the observed mean)"""
    o, p = np.asarray(obs, float), np.asarray(pred, float)
    e = p - o
    rmse = float(np.sqrt(np.mean(e ** 2)))
    om = o.mean()
    d = 1 - np.sum(e ** 2) / np.sum((np.abs(p - om) + np.abs(o - om)) ** 2)
    r2 = float(np.corrcoef(o, p)[0, 1] ** 2) if len(o) > 2 else np.nan
    sst = float(np.sum((o - om) ** 2))
    r2d = float(1 - np.sum(e ** 2) / sst) if sst > 0 else np.nan
    return {"n": len(o), "mean_observed": float(om), "mean_predicted": float(p.mean()), "bias": float(e.mean()),
            "mae": float(np.abs(e).mean()), "rmse": rmse, "nrmse_pct": 100 * rmse / om, "willmott_d": float(d),
            "r2_pearson": r2, "r2_determination": r2d}
