"""ESM-C, vendored from the Biohub Transformers fork (Apache-2.0).

Source: github.com/Biohub/transformers @ b435f1f92dd5b4a653be57157d4f4f5ddba4f145,
``src/transformers/models/esmc/`` (configuration, configuration_sae,
modeling, modeling_sae), unchanged except that the package-relative
``...`` imports now name ``transformers`` explicitly and the docstring
generator is a no-op. The fork was taken down;
its ESM-C is the implementation ESMFold2's language model was validated with
(hidden states matched the released ESMFold2 to correlation 0.9999), so it is
kept verbatim here and runs on the PyPI ``transformers==4.57.6`` it was built
on. See NOTICE.
"""

from foldforge.modules.esmc.configuration import ESMCConfig
from foldforge.modules.esmc.modeling import ESMCModel

__all__ = ["ESMCConfig", "ESMCModel"]
