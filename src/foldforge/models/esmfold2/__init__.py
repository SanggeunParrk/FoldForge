"""ESMFold2, assembled from miniworld-engine ops and team-gm blocks.

Ported from team-gm's ``exp/miniworld-integrated`` branch, where it was written
against the pre-migration layout. The rewire is recorded in
``docs/PORTING-esmfold2.md``; the part worth knowing before touching anything
here is that **engine ops own their own self-residual**. A block must call
``pair = self.tri_mul_out(pair, mask)`` and never ``pair + self.tri_mul_out(...)``
— the second form double-adds, raises nothing, and yields a plausible wrong
structure. Cross-tensor residuals are the reverse: the op returns a raw delta and
the model passes the target in as ``residual=``, as ``MSAEncoderBlock`` does for
``OuterProductMean``.

What is here:

* :class:`FoldingTrunk` / :class:`PairUpdateBlock` — the 48-layer trunk, and the
  same block at the depths used by the LM encoder, the parcae coda and the
  confidence head.
* :class:`MSAEncoder` / :class:`MSAEncoderBlock` — MSA conditioning of the pair
  representation. Hoisted out of the parcae recurrence by the caller: it is
  loop-invariant, so running it per pass is pure waste.
* :class:`PairTrunk` — the recurrent assembly, plus the optional distogram head.
* :class:`ConfidenceHead` — pLDDT, PAE, PDE, resolved, pTM and ipTM.
* :class:`AtomEncoder` / :class:`InputsEmbedder` and
  :class:`SWAAtomTransformer` — reference-conformer atom encoding pooled into the
  451-dim token inputs. The atom transformer is FoldForge's own copy; see its
  module docstring for why.
* :mod:`.convert` — remaps released ESMFold2 weights onto these modules.

ESMC is not reimplemented — it shares nothing with the folding stack, so
:mod:`.lm` handles the token plumbing and loads the released model through
``transformers``.

Two behaviours are enforced in code and must stay that way: the LM encoder
cannot be disabled (doing so takes 4YX2 from pLDDT 0.842 / RMSD 4.4 A to 0.550 /
11.3 A, to save ~6% of the fold), and the MSA encoder is hoisted out of the
recurrence.

Scope: **inference only.** This exists to run released weights. There is no
loss, no optimiser wiring, and no backward pass has been validated — the modules
inherit autograd, so one will *execute*, which is not evidence it is correct.

Backend: modules default to ``ImplementationType.MINIWORLD_ENGINE``. That path is
Triton/CUDA-only — pass ``ImplementationType.PYTORCH`` explicitly to run on CPU.
"""

from . import convert
from .atom_encoder import AtomEncoder, InputsEmbedder
from .atom_transformer import SWAAtomBlock, SWAAtomTransformer
from .confidence import ConfidenceHead, ConfidenceOutput, chain_pair_iptm
from .config import ESMFold2Config
from .diffusion import (
    AtomDecoder,
    DiffusionConditioning,
    DiffusionModule,
    DiffusionStructureHead,
    FourierEmbedding,
)
from .embeddings import (
    LanguageModelShim,
    RelativePositionEncoding,
    RowAttentionPooling,
    SingleToPair,
)
from .lm import compute_lm_hidden_states, load_esmc
from .model import ESMFold2Model, ESMFold2Output
from .msa_encoder import MSAEncoder, MSAEncoderBlock, msa_from_reference
from .pair_trunk import PairTrunk, PairTrunkOutput
from .solver import ESMFold2Solver
from .trunk import FoldingTrunk, PairUpdateBlock

#: The registry entry point — ``foldforge.get_model("esmfold2")`` returns this.
load = ESMFold2Model.from_checkpoint

__all__ = [
    "AtomDecoder",
    "AtomEncoder",
    "ConfidenceHead",
    "ConfidenceOutput",
    "DiffusionConditioning",
    "DiffusionModule",
    "DiffusionStructureHead",
    "ESMFold2Config",
    "ESMFold2Model",
    "ESMFold2Output",
    "ESMFold2Solver",
    "FoldingTrunk",
    "FourierEmbedding",
    "InputsEmbedder",
    "LanguageModelShim",
    "MSAEncoder",
    "MSAEncoderBlock",
    "PairTrunk",
    "PairTrunkOutput",
    "PairUpdateBlock",
    "RelativePositionEncoding",
    "RowAttentionPooling",
    "SWAAtomBlock",
    "SWAAtomTransformer",
    "SingleToPair",
    "chain_pair_iptm",
    "compute_lm_hidden_states",
    "convert",
    "load",
    "load_esmc",
    "msa_from_reference",
]
