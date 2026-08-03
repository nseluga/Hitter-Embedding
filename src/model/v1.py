"""
The v1 conditional-query network and its loss (docs/phase-d-spec.md §3 and §4).

Every failure this module can produce is silent. A head returning (B, 1) instead of
(B,) still trains, because broadcasting turns the loss into an all-pairs average that
decreases perfectly happily. A masked row that contributes anything at all teaches the
model from an observation that does not exist. A reserved row that receives gradient
gives unseen hitters a learned personality, which is the opposite of the cold-start
decision. None of these raise.

1. THE FACTORISATION (§1.2). One network, five factors, evaluated per pitch:

       p(swing | h, c) · p(contact | swing, h, c)
                       · p(ev | contact, h, c) · p(la | ev, ·) · p(spray | ev, la, ·)

   Take outcomes are not predicted; they follow from plate location, which is in c,
   and are composed in D.5.

2. TEACHER FORCING (§1.5). At training time the LA head conditions on the OBSERVED
   EV bin and the spray head on the observed EV and LA bins, not on the model's own
   guesses. At inference the caller passes sampled bins through the same argument.

3. MASK NESTING (2026-08-02). A factor is scored only where everything it conditions
   on was observed: contact on swings, ev on balls in play, la where ev and la are
   both present, spray where all three are. One-hot conditioning vectors are zeroed
   on masked rows because MASKED = -1 is not a valid index, and those rows are
   excluded from the loss regardless.

4. THE RESERVED ROW (2026-07-30). Row 0 is zero-initialised and frozen via
   padding_idx, so an unseen hitter and a generic hitter are the same point. AdamW's
   decay on a zero row is a no-op, so the row stays exactly zero without a guard.

5. THE LOSS RULE (2026-08-01, 2026-08-02). "log" is the default and is the plain
   likelihood of the factorisation, so the five terms add up as an identity rather
   than a choice. "rps" is the ranked probability score over all five factors,
   reducing to the Brier score on the binary ones; it exists for the promote-only
   screen and is never the default without a new decision-log entry.

6. THE INTERACTION TERM (2026-08-03). Low-rank, not full: the two towers are each
   projected to 32 shared interaction directions, multiplied position by position,
   and mapped back to the trunk's width. 13,312 parameters against the full form's
   1,048,832, which alone would outweigh the rest of the network five to one.
   Default OFF -- it is a D.8 arm, and a first run carrying an unmeasured term
   could not attribute its margin.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.data.model_dataset import MASKED, RESERVED_HITTER_INDEX

DEFAULT_EMBEDDING_DIM = 32
DEFAULT_N_BINS = 24
CONTEXT_HIDDEN = 128
TRUNK_HIDDEN = 256
TRUNK_DROPOUT = 0.1
BILINEAR_RANK = 32
EMBEDDING_INIT_STD = 0.01
BINARY_FACTORS = ("swing", "contact")
QUALITY_FACTORS = ("ev", "la", "spray")
FACTORS = BINARY_FACTORS + QUALITY_FACTORS
LOSS_RULES = ("log", "rps")


class HitterEmbeddingV1(nn.Module):
    """
    Hitter embedding + context MLP -> shared trunk -> five factor heads (spec §3).
    n_hitters: vocabulary size excluding the reserved row; n_context: context width;
    bilinear adds the low-rank interaction term, which is a D.8 arm and stays off.
    forward returns a dict of logits, one entry per factor.
    """

    def __init__(self, n_hitters, n_context, embedding_dim=DEFAULT_EMBEDDING_DIM,
                 n_bins=DEFAULT_N_BINS, context_hidden=CONTEXT_HIDDEN,
                 trunk_hidden=TRUNK_HIDDEN, dropout=TRUNK_DROPOUT, bilinear=False,
                 rank=BILINEAR_RANK):
        super().__init__()
        self.n_bins = n_bins

        self.embedding = nn.Embedding(n_hitters + 1, embedding_dim,
                                      padding_idx=RESERVED_HITTER_INDEX)
        nn.init.normal_(self.embedding.weight, std=EMBEDDING_INIT_STD)
        # nn.Embedding zeroes padding_idx at construction and the init above overwrites
        # it, so the reserved row is restored here or unseen hitters start at noise
        with torch.no_grad():
            self.embedding.weight[RESERVED_HITTER_INDEX].zero_()

        self.context_tower = nn.Sequential(
            nn.Linear(n_context, context_hidden), nn.ReLU(),
            nn.Linear(context_hidden, context_hidden), nn.ReLU())

        self.trunk = nn.Sequential(
            nn.Linear(embedding_dim + context_hidden, trunk_hidden), nn.ReLU(),
            nn.Linear(trunk_hidden, trunk_hidden), nn.ReLU(),
            nn.Dropout(dropout))

        # low-rank interaction (2026-08-03): ask both towers the same `rank` questions,
        # multiply the answers position by position, and let one map spend the result
        # across the trunk. Bias-free by construction -- a bias here would fire on every
        # pitch identically, which is a main effect the trunk already carries.
        if bilinear:
            self.project_hitter = nn.Linear(embedding_dim, rank, bias=False)
            self.project_context = nn.Linear(context_hidden, rank, bias=False)
            self.interaction = nn.Linear(rank, trunk_hidden, bias=False)
        else:
            self.project_hitter = self.project_context = self.interaction = None

        self.head_swing = nn.Linear(trunk_hidden, 1)
        self.head_contact = nn.Linear(trunk_hidden, 1)
        self.head_ev = nn.Linear(trunk_hidden, n_bins)
        self.head_la = nn.Linear(trunk_hidden + n_bins, n_bins)
        self.head_spray = nn.Linear(trunk_hidden + 2 * n_bins, n_bins)

    def conditioning(self, bins):
        """One-hot of an observed bin, all zeros where the bin is MASKED."""
        observed = (bins != MASKED).unsqueeze(1).to(torch.float32)
        return F.one_hot(bins.clamp(min=0), self.n_bins).to(torch.float32) * observed

    def forward(self, hitter, context, ev_bin, la_bin):
        assert hitter.dim() == 1, "hitter must be (B,), one index per pitch"
        assert context.dim() == 2 and context.shape[0] == hitter.shape[0], \
            "context must be (B, n_context) aligned with hitter"
        batch = hitter.shape[0]

        embedded = self.embedding(hitter)
        encoded = self.context_tower(context)
        trunk = self.trunk(torch.cat([embedded, encoded], dim=1))
        if self.interaction is not None:
            paired = self.project_hitter(embedded) * self.project_context(encoded)
            trunk = trunk + self.interaction(paired)

        ev_onehot = self.conditioning(ev_bin)
        la_onehot = self.conditioning(la_bin)
        outputs = {
            "swing": self.head_swing(trunk).squeeze(-1),
            "contact": self.head_contact(trunk).squeeze(-1),
            "ev": self.head_ev(trunk),
            "la": self.head_la(torch.cat([trunk, ev_onehot], dim=1)),
            "spray": self.head_spray(torch.cat([trunk, ev_onehot, la_onehot], dim=1)),
        }

        # a binary head returning (B, 1) would broadcast against a (B,) target and turn
        # the loss into an all-pairs average without failing
        for name in BINARY_FACTORS:
            assert outputs[name].shape == (batch,), f"{name} head must be (B,)"
        for name in QUALITY_FACTORS:
            assert outputs[name].shape == (batch, self.n_bins), \
                f"{name} head must be (B, n_bins)"
        return outputs


def factor_masks(labels):
    """
    Rows each factor is scored on, nested on what it conditions on (2026-08-02).
    labels: dict of int64 targets carrying MASKED where a factor is undefined.
    Returns one boolean mask per factor.
    """
    swing = torch.ones_like(labels["swing"], dtype=torch.bool)
    contact = labels["contact"] != MASKED
    ev = labels["ev"] != MASKED
    la = ev & (labels["la"] != MASKED)
    spray = la & (labels["spray"] != MASKED)
    return {"swing": swing, "contact": contact, "ev": ev, "la": la, "spray": spray}


def _binary_loss(logit, target, mask, rule):
    if not bool(mask.any()):
        return (logit * 0.0).sum()
    selected, observed = logit[mask], target[mask].to(torch.float32)
    if rule == "log":
        return F.binary_cross_entropy_with_logits(selected, observed, reduction="sum")
    return ((torch.sigmoid(selected) - observed) ** 2).sum()


def _quality_loss(logits, target, mask, rule):
    if not bool(mask.any()):
        return (logits * 0.0).sum()
    selected, observed = logits[mask], target[mask]
    if rule == "log":
        return F.cross_entropy(selected, observed, reduction="sum")
    # ranked probability score: squared gap between the predicted cumulative
    # distribution and the step function that jumps at the observed bin
    predicted = torch.softmax(selected, dim=1).cumsum(dim=1)
    positions = torch.arange(selected.shape[1], device=selected.device)
    step = (positions.unsqueeze(0) >= observed.unsqueeze(1)).to(predicted.dtype)
    return ((predicted - step) ** 2).sum()


def factorized_loss(outputs, labels, rule="log"):
    """
    Sum of the five factor losses over the rows each factor is defined on (spec §4).
    outputs: head logits; labels: int64 targets with MASKED where undefined;
    rule: "log" for the likelihood, "rps" for the ranked probability score.
    Returns (total, per-factor dict). Raw sums, never per-head means.
    """
    assert rule in LOSS_RULES, f"unknown loss rule {rule!r}"
    masks = factor_masks(labels)
    parts = {}
    for name in BINARY_FACTORS:
        parts[name] = _binary_loss(outputs[name], labels[name], masks[name], rule)
    for name in QUALITY_FACTORS:
        parts[name] = _quality_loss(outputs[name], labels[name], masks[name], rule)
    total = sum(parts.values())
    return total, parts


def weight_decay_groups(model, weight_decay):
    """
    Parameter groups for AdamW: weights and the embedding decay, biases do not (spec §5).
    Decay on the reserved embedding row is a no-op because that row is zero.
    """
    decayed, undecayed = [], []
    for name, parameter in model.named_parameters():
        (undecayed if name.endswith("bias") else decayed).append(parameter)
    return [{"params": decayed, "weight_decay": weight_decay},
            {"params": undecayed, "weight_decay": 0.0}]
