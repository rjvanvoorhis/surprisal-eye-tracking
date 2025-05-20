import typing

import numpy as np
from common.interfaces import CalculatesLikelihood, SizedIterable


def surprisal(
    likelihood_fn: CalculatesLikelihood,
    text: SizedIterable[str],
    bins: typing.Optional[SizedIterable[int]] = None,
) -> np.ndarray:
    likelihoods = list(likelihood_fn(text))
    token_surprisals = -1 * np.log(likelihoods)
    return (
        token_surprisals
        if bins is None
        else np.bincount(bins, weights=token_surprisals)
    )
