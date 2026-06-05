"""
NA2M — Neural Additive (×2) Model: mains + pairwise interactions.

A GAMI-Net-style additive model with a type-aware main bank (FeatureNN for
numericals, CategNet for integer-coded categoricals) plus dynamically added
pairwise InteractionNN subnets:

    y = b + Σ_j f_j(x_j) + Σ_(j,k) f_jk(x_j, x_k)

ONE backbone serves all three experiment arms:
    Arm A = NA2M with interactions disabled (no pairs added). The
            interactions-off forward path MUST be identical to a mains-only NAM
            forward (same dropout/sum/bias behavior). Do NOT use the old one-hot
            NAM as arm A.
    Arm B / C = mains + interactions (C additionally runs the concurvity filter).

Dynamic model: interaction subnets are built EXACTLY when selected (no
pre-allocation, no masking). After any structural change (add/remove
interaction), the optimizer MUST be rebuilt by the caller.

Terms are keyed by term_id, never positionally:
    main term  → ("main", j)
    inter term → ("inter", j, k)

Do NOT modify nam.py / feature_nn.py / activation — those are the frozen
reproduction baseline. This is a new, parallel model.
"""

import torch
import torch.nn as nn

from nam.models.feature_nn import FeatureNN
from .categnet import CategNet
from .interaction_nn import InteractionNN


class NA2M(nn.Module):
    """
    Neural Additive ×2 Model: type-aware main bank + dynamic pairwise interactions.
    """

    def __init__(
        self,
        num_features: int,
        feature_meta: list,
        num_units: int,
        hidden_sizes: list,
        dropout: float,
        feature_dropout: float,
        activation: str,
        inter_units: int,
        inter_hidden: list,
        inter_dropout: float,
    ):
        """Initialize the NA2M model.

        Args:
            num_features (int): Number of input features (main terms).
            feature_meta (list): Per-feature metadata (FeatureMeta), carries type/levels.
                                 Kept on the model — the harness needs type/levels.
            num_units (int): Width of the main FeatureNN activation layer.
            hidden_sizes (list): Hidden layer widths for main FeatureNNs.
            dropout (float): Dropout inside main subnets.
            feature_dropout (float): Probability of dropping an entire term before summation.
            activation (str): Main FeatureNN activation, 'exu' or 'relu'.
            inter_units (int): Width of the first InteractionNN layer.
            inter_hidden (list): Hidden layer widths for InteractionNNs.
            inter_dropout (float): Dropout inside interaction subnets.

        TODO:
            - Build a TYPE-AWARE main bank as an nn.ModuleList indexed by feature:
                FeatureNN for feature_meta[j].type == "num",
                CategNet(n_levels) for feature_meta[j].type == "cat".
            - self.interaction_nns = nn.ModuleDict()  (empty at init).
            - self._bias = nn.Parameter(torch.zeros(1))  (the ONE model-wide intercept).
            - self.dropout_layer = nn.Dropout(p=feature_dropout).
            - Store num_features, feature_meta, and interaction hyperparameters
              (needed when add_interactions builds new subnets).
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Structural mutation (rebuild the optimizer after ANY of these)
    # ------------------------------------------------------------------

    def add_interactions(self, pairs: list[tuple[int, int]]) -> None:
        """Build one InteractionNN per pair into the ModuleDict. Idempotent.

        Args:
            pairs: List of (j, k) feature-index pairs to add.

        TODO:
            - For each (j, k): key = f"{j},{k}"; skip if already present.
            - Build InteractionNN(inter_units, inter_hidden, inter_dropout) into the dict.
            - Caller MUST rebuild the optimizer afterwards.
        """
        raise NotImplementedError

    def remove_interaction(self, j: int, k: int) -> None:
        """Delete the interaction subnet for pair (j, k) from the ModuleDict.

        Args:
            j: First feature index.
            k: Second feature index.

        TODO:
            - del self.interaction_nns[f"{j},{k}"].
            - Caller MUST rebuild the optimizer afterwards.
        """
        raise NotImplementedError

    def set_main_trainable(self, flag: bool) -> None:
        """Toggle requires_grad on all main-bank parameters.

        Args:
            flag: True to train the main bank, False to freeze it.

        TODO:
            - Iterate main-bank params, set p.requires_grad = flag.
        """
        raise NotImplementedError

    def active_pairs(self) -> list[tuple[int, int]]:
        """Return active interaction pairs parsed from ModuleDict keys.

        Returns:
            List of (j, k) tuples; [] when no interactions are active.

        TODO:
            - Parse each "j,k" key back into (int(j), int(k)).
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Per-term evaluation
    # ------------------------------------------------------------------

    def main_outputs(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Evaluate every main term, type-aware.

        Args:
            x: Input batch, shape (batch_size, num_features).

        Returns:
            List of num_features tensors, each (batch_size, 1).

        TODO:
            - For feature j: slice column j; route to CategNet (cat) or FeatureNN (num).
        """
        raise NotImplementedError

    def inter_outputs(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Evaluate every active interaction term, in active_pairs() order.

        Args:
            x: Input batch, shape (batch_size, num_features).

        Returns:
            List of m tensors (m = number of active pairs), each (batch_size, 1). [] if none.

        TODO:
            - For each (j, k) in active_pairs(): stack columns j and k → (batch_size, 2).
            - Resolve the categorical-input handling flagged in InteractionNN.
            - Run the matching InteractionNN.
        """
        raise NotImplementedError

    def iter_terms(self):
        """Yield (term_id, fn) for every term: mains first, then active interactions.

        term_id is ("main", j) or ("inter", j, k). `fn` evaluates that ONE term on
        arbitrary inputs (1-col for main, 2-col for inter).

        Yields:
            (term_id, fn) tuples.

        TODO:
            - Yield (("main", j), fn_j) for each feature.
            - Yield (("inter", j, k), fn_jk) for each active pair.
            - Use the late-binding guard (j=j, k=k) in every closure over loop vars.
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Prediction / forward
    # ------------------------------------------------------------------

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Return only the scalar prediction (forward(x)[0]).

        Args:
            x: Input batch, shape (batch_size, num_features).

        Returns:
            Predictions, shape (batch_size,).

        TODO:
            - return self.forward(x)[0].
        """
        raise NotImplementedError

    def forward(self, x: torch.Tensor):
        """Forward pass: sum all term outputs + bias, with feature dropout.

        terms = main_outputs + inter_outputs; concat to (batch_size, K+m);
        feature dropout; sum over terms; + bias.

        The INTERACTIONS-OFF path (no pairs added) MUST be identical to a
        mains-only NAM forward — same dropout/sum/bias behavior. This is arm A.

        Args:
            x: Input batch, shape (batch_size, num_features).

        Returns:
            (out, dropped_terms): out shape (batch_size,); dropped_terms the
            per-term tensor after feature dropout (for the output penalty).

        TODO:
            - terms = self.main_outputs(x) + self.inter_outputs(x).
            - concat → dropout_layer → sum over dim=-1 → + self._bias.
            - assert/comment: with no active pairs this equals the mains-only forward.
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # GAMI-Net marginal-clarity penalty (NOT YET wired — Stage 3 gap)
    # ------------------------------------------------------------------

    def clarity_loss(self, x: torch.Tensor) -> torch.Tensor:
        """GAMI-Net marginal-clarity penalty over active interaction pairs.

        Penalises overlap between an interaction and its parent mains:
            Σ_(j,k) ( |mean_i f_j(x_ij) · f_jk(x_ij, x_ik)|
                    + |mean_i f_k(x_ik) · f_jk(x_ij, x_ik)| )

        NOT YET WIRED into any training loop — Stage 3 wires this into
        stage3_finetune. Correctness gap to fill there.

        Args:
            x: Input batch, shape (batch_size, num_features).

        Returns:
            Scalar penalty tensor.

        TODO:
            - For each active (j, k): compute the marginal-clarity product terms.
            - Confirm the exact GAMI-Net form against GamiNet-master before wiring.
        """
        raise NotImplementedError
