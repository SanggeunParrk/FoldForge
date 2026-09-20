"""The two seams every predictor has to pass through."""

import pytest
import torch

from foldforge.models import describe, get_model, known_models, registered_models
from foldforge.prediction import Prediction


def test_prediction_rejects_a_squeezed_sample_axis():
    # The mistake this guards is silent: (n_atoms, 3) is a valid tensor and
    # every downstream consumer would then read n_atoms as a sample count.
    with pytest.raises(ValueError, match="n_samples"):
        Prediction(coords=torch.zeros(37, 3))


def test_prediction_keeps_absent_heads_as_none():
    # None means "this model has no distogram head", which is information a
    # zero tensor would destroy.
    prediction = Prediction(coords=torch.zeros(1, 37, 3))
    assert prediction.distogram_logits is None
    assert prediction.n_samples == 1


def test_registry_lists_without_importing_any_model():
    # Listing must not need weights or an upstream package on the path.
    assert set(known_models()) >= {"af3", "boltz2", "chai1", "esmfold2", "protenix"}
    assert describe("esmfold2")


def test_registered_is_only_what_is_actually_ported():
    # known_models() is the plan, registered_models() is the truth. Conflating
    # them is how a CLI ends up offering a model it cannot build.
    assert set(registered_models()) <= set(known_models())


def test_planned_model_fails_at_lookup_not_mid_forward():
    with pytest.raises(NotImplementedError, match="not ported yet"):
        get_model("chai1")


def test_unknown_model_names_the_known_ones():
    with pytest.raises(KeyError, match="esmfold2"):
        get_model("no-such-model")


@pytest.mark.parametrize("name", ["af3", "protenix", "opendde"])
def test_new_ports_are_loadable(name):
    assert name in registered_models()
    assert callable(get_model(name))
