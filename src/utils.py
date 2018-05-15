def check_is_fitted(estimator, attribute):
    """Perform is_fitted validation for estimator.
    """

    if not hasattr(estimator, attribute):
        raise NotFittedError("Estimator not fitted, call `fit`"
                             " before making predictions`.")
