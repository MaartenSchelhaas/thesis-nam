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
                FeatureNN(num_units, hidden_sizes, dropout, activation) for type "num",
                CategNet(n_levels) for type "cat".
            - self.interaction_nns = nn.ModuleDict()  (empty at init).
            - self._bias = nn.Parameter(torch.zeros(1)).
            - self.dropout_layer = nn.Dropout(p=feature_dropout).
            - Store num_features, feature_meta, and interaction hyperparams
              (inter_units / inter_hidden / inter_dropout) for add_interactions.
            - Centering offsets:
                self.register_buffer("main_centers", torch.zeros(num_features))
                self.inter_centers = {}        # plain dict, key "j,k" -> tensor offset
                self._inter_folded = {}        # key "j,k" -> bool, folded into _bias?
              main_outputs / inter_outputs subtract the stored offset per term.
              inter_centers/_inter_folded are populated by add_interactions (zero,
              False) and updated by center_interactions. NOT buffers (dynamic keys).
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
            - For each (j,k): key=f"{j},{k}"; skip if present.
            - interaction_nns[key] = InteractionNN(inter_units, inter_hidden, inter_dropout).
            - inter_centers[key] = torch.zeros(1, device=self._bias.device)
            - _inter_folded[key] = False
            - Caller MUST rebuild the optimizer.
        """
        raise NotImplementedError

    def remove_interaction(self, j: int, k: int) -> None:
        """Delete the interaction subnet for pair (j, k) from the ModuleDict.

        Args:
            j: First feature index.
            k: Second feature index.

        TODO:
            - key = f"{j},{k}".
            - IF _inter_folded.get(key): with no_grad: self._bias -= inter_centers[key]
              (the offset was added to _bias when folded; deleting the subnet removes
               the matching '-offset' term, so the bias must give it back).
            - inter_centers.pop(key, None); _inter_folded.pop(key, None)
            - del interaction_nns[key]
            - Caller MUST rebuild the optimizer.
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

    def center_main_effects(self, X_pool: torch.Tensor) -> None:
        """Fold each main subnet's CURRENT mean into _bias. Idempotent.
        Call after stage1 AND re-call at the end of every fine-tune (stage 3, each
        stage-4 pass). Pure reparameterisation — predictions unchanged.
        TODO:
            - eval() + no_grad throughout (restore train mode after if needed).
            - For each j:
                raw = main_nn_j(X_pool[:, j])
                centered = raw - self.main_centers[j]      # current effective output
                delta = centered.mean()
                self.main_centers[j] += delta              # ACCUMULATE, don't overwrite
                self._bias += delta
              (accumulate-delta makes repeated calls idempotent; overwriting from raw
               would double-count on the 2nd call.)
        """
        raise NotImplementedError

    def center_interactions(self, X_pool: torch.Tensor) -> None:
        """Fold each active interaction's CURRENT mean. Idempotent.
        fold_bias=False  -> update inter_centers only, DO NOT touch _bias (used during
                            the η-prune sweep so excluded pairs leave no orphan bias).
        fold_bias=True   -> also add delta to _bias and mark _inter_folded[key]=True
                            (call once after the survivor set is fixed, and at the end
                            of each fine-tune).
        TODO:
            - eval() + no_grad.
            - For each (j,k) in active_pairs():
                key=f"{j},{k}"; cols = stack(X_pool[:,j], X_pool[:,k])
                raw = interaction_nns[key](cols)
                centered = raw - inter_centers[key]
                delta = centered.mean()
                inter_centers[key] += delta               # ACCUMULATE
                if fold_bias: self._bias += delta; _inter_folded[key] = True
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
            - For each (j,k) in active_pairs(): cols = stack(x[:,j], x[:,k]);
              out = interaction_nns[key](cols) - inter_centers.get(key, 0.0)
              (DEFENSIVE .get default 0 — inter_outputs runs on every forward from the
               moment a pair is added, before any centering.)
            - Resolve categorical-input handling (see InteractionNN TODO) consistently
              with how the subnet was built.
        """
        raise NotImplementedError

    def iter_terms(self):
        """Yield (term_id, fn) for every term: mains first, then active interactions.

        term_id is ("main", j) or ("inter", j, k). `fn` evaluates that ONE term on
        arbitrary inputs (1-col for main, 2-col for inter).

        Yields:
            (term_id, fn) tuples.

        TODO:
            - main fn: raw_main(j, col) - main_centers[j]   (late-bind j=j)
            - inter fn: raw_inter(key, cols) - inter_centers.get(key, 0.0)  (late-bind key=key)
            - returned fns yield CENTERED outputs (used by extract/concurvity).
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
        """Σ_(j,k) (|mean f_j·f_jk| + |mean f_k·f_jk|) on CENTERED outputs.
        TODO:
            - fj = main_nn_j(x[:,j]) - main_centers[j];  fk likewise for k.
            - fjk = interaction_nns[key](cols) - inter_centers.get(key, 0.0).
            - penalty += (fj*fjk).mean().abs() + (fk*fjk).mean().abs(), summed over pairs.
            - Centering is REQUIRED here: the penalty is only an orthogonality measure
              when both factors are zero-mean. Matches reference call() form.
        """
        raise NotImplementedError
