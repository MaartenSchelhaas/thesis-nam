"""
NA2M — Neural Additive (×2) Model: mains + pairwise interactions.

A GAMI-Net-style additive model with a type-aware main bank (FeatureNN for
numericals, CategNet for integer-coded categoricals) plus dynamically added
pairwise interaction subnets (FeatureNN with in_features=2, activation='relu'):

    y = b + Σ_j f_j(x_j) + Σ_(j,k) f_jk(x_j, x_k)

ONE backbone serves all three experiment arms:
    Arm A = NA2M with interactions disabled (no pairs added). The
            interactions-off forward path MUST be identical to a mains-only NAM
            forward (same dropout/sum/bias behavior). Do NOT use the old one-hot
            NAM as arm A.
    Arm B / C = mains + interactions. They share an IDENTICAL pipeline; the only
            difference is that arm C fires the concurvity gate during the Stage-2
            prune sweep (concurvity_filter=True). No per-removal re-fine-tune
            exists for either arm — both fine-tune exactly once.

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

from .feature_nn import FeatureNN
from .categnet import CategNet
from na2m.data.data_utils import FeatureMeta


class NA2M(nn.Module):
    """
    Neural Additive ×2 Model: type-aware main bank + dynamic pairwise interactions.
    """

    def __init__(
        self,
        num_features: int,
        feature_meta: list[FeatureMeta],
        num_units: int,
        hidden_sizes: list,
        dropout: float,
        feature_dropout: float,
        activation: str,
        inter_units: int = 32,
        inter_hidden: list = [],
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
            inter_units (int): Width of the interaction FeatureNN activation layer. Defaults to 32.
            inter_hidden (list): Hidden layer widths for interaction subnets. Defaults to [] (shallow).

        Attributes set:
            main_nns:       nn.ModuleList of FeatureNN (num) or CategNet (cat), one per feature.
            inter_nns:      nn.ModuleDict, empty at init; populated by add_interactions().
            _bias:          Global learnable scalar intercept (nn.Parameter).
            dropout_layer:  Feature dropout applied before summation.
            main_centers:   Registered buffer (num_features,); accumulated centering offsets
                            per main term. Subtracted in main_outputs.
            inter_centers:  Plain dict "j,k" → tensor; centering offsets per interaction.
                            Plain dict (not buffer) because keys are dynamic.
            _inter_folded:  Plain dict "j,k" → bool; True when the interaction's offset has
                            been folded into _bias. Used by remove_interaction to restore _bias.
        """
        super().__init__()
        #Sanity check
        assert num_features == len(feature_meta)
        self.num_features = num_features
        self.feature_meta = feature_meta
        
        self.inter_units = inter_units
        self.inter_hidden = inter_hidden
        self.dropout = dropout

        #Create main effect subnets
        def _make_subnet(meta):
            if meta.type == "num":
                return FeatureNN(num_units=num_units, hidden_sizes=hidden_sizes, dropout=dropout, activation=activation)
            assert meta.n_levels is not None, f"cat feature '{meta.name}' missing n_levels"
            return CategNet(n_levels=meta.n_levels)
        self.main_nns = nn.ModuleList([_make_subnet(meta) for meta in feature_meta])

        #Initialize interaction effect
        self.inter_nns = nn.ModuleDict() # str "j,k" -> FeatureNN(in_features=2, activation='relu')

        self.dropout_layer = nn.Dropout(p=feature_dropout)
        self._bias = nn.Parameter(data = torch.zeros(1))

        self.register_buffer("main_centers", torch.zeros(num_features))
        self.main_centers: torch.Tensor
        self.inter_centers: dict[str, torch.Tensor] = {}
        self._inter_folded: dict[str, bool] = {}


    # ------------------------------------------------------------------
    # Structural mutation (rebuild the optimizer after ANY of these)
    # ------------------------------------------------------------------

    def _encode_col(self, x: torch.Tensor, j: int) -> torch.Tensor:
        """One-hot encode column j if categorical, else return as (batch, 1)."""
        meta = self.feature_meta[j]
        if meta.type == "num":
            col = x[:, j:j+1]  # (batch, 1)
            return col
        assert meta.n_levels is not None
        indices = x[:, j].long()
        col = torch.nn.functional.one_hot(indices, num_classes=meta.n_levels).float()  # (batch, n_levels)
        return col

    def _inter_in_features(self, j: int, k: int) -> int:
        """Compute interaction subnet input width: 1 per num feature, n_levels per cat feature."""
        meta_j = self.feature_meta[j]
        meta_k = self.feature_meta[k]

        if meta_j.type == "num":
            contrib_j = 1
        else:
            assert meta_j.n_levels is not None
            contrib_j = meta_j.n_levels

        if meta_k.type == "num":
            contrib_k = 1
        else:
            assert meta_k.n_levels is not None
            contrib_k = meta_k.n_levels

        return contrib_j + contrib_k

    def add_interactions(self, pairs: list[tuple[int, int]]) -> None:
        """Build one interaction subnet per pair into the ModuleDict. Idempotent.

        Args:
            pairs: List of (j, k) feature-index pairs to add.

        Note:
            Caller MUST rebuild the optimizer after calling this.
        """
        for j, k in pairs:
            key = f"{j},{k}"
            if key in self.inter_nns:
                continue
            in_features = self._inter_in_features(j, k)
            self.inter_nns[key] = FeatureNN(
                num_units=self.inter_units,
                hidden_sizes=self.inter_hidden,
                dropout=self.dropout,
                activation="relu",
                in_features=in_features,
            )
            self.inter_centers[key] = torch.zeros(1, device=self._bias.device)
            self._inter_folded[key] = False
        

    def remove_interaction(self, j: int, k: int) -> None:
        """Remove the interaction subnet for pair (j, k) and clean up all associated state.

        If the interaction's centering offset was previously folded into _bias, it is
        subtracted back out to keep predictions unchanged after deletion.

        Args:
            j: First feature index.
            k: Second feature index.

        Note:
            Caller MUST rebuild the optimizer after calling this.
        """
        key = f"{j},{k}"

        if self._inter_folded.get(key):
            with torch.no_grad():
                self._bias -= self.inter_centers[key]

        del self.inter_nns[key]
        self.inter_centers.pop(key, None)
        self._inter_folded.pop(key, None)



    def set_main_trainable(self, flag: bool) -> None:
        """Freeze or unfreeze all main-bank parameters.

        Called before interaction block training (flag=False) to keep main
        effects fixed, and before joint fine-tuning (flag=True) to unfreeze them.

        Args:
            flag: True to unfreeze the main bank, False to freeze it.
        """
        for subnet in self.main_nns:
            subnet.requires_grad_(flag)

    def center_main_effects(self, X_pool: torch.Tensor) -> None:
        """Fold each main subnet's CURRENT mean into _bias. Idempotent.
        Call after Stage 1 AND at the end of the SINGLE Stage-3 fine-tune. Pure
        reparameterisation — predictions unchanged.
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

    def center_interactions(self, X_pool: torch.Tensor, fold_bias: bool) -> None:
        """Fold each active interaction's CURRENT mean. Idempotent.
        fold_bias=False  -> update inter_centers only, DO NOT touch _bias. Used
                            DURING the Stage-2 prune sweep: offsets are held
                            per-term so candidates that end up skipped/excluded
                            never contaminate _bias or the validation loss.
        fold_bias=True   -> also add delta to _bias and mark _inter_folded[key]=True.
                            Call ONCE after the survivor set is fixed, and again at
                            the end of the SINGLE Stage-3 fine-tune.
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
        """
        pairs = []
        for key in self.inter_nns:
            j, k = key.split(",")
            pairs.append((int(j), int(k)))
        return pairs

    # ------------------------------------------------------------------
    # Per-term evaluation
    # ------------------------------------------------------------------

    def main_outputs(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Pass each feature column through its main subnet and subtract the accumulated centering offset.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_features).

        Returns:
            list[torch.Tensor]: List of num_features tensors, each of shape (batch_size, 1),
                                representing the centered contribution f_i(x_i) - main_centers[i].
        """

        individual_outputs = []
        for i in range(self.num_features):
            feature_input = x[:,i].unsqueeze(-1)
            feature_output = self.main_nns[i](feature_input) - self.main_centers[i]
            individual_outputs.append(feature_output)
        return individual_outputs


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
        individual_outputs = []
        for j,k in self.active_pairs():
            key = f"{j,k}"

            #feature_input = 

            #feature_input = self._encode_col(x[:,i].unsqueeze(-1),i)



        raise NotImplementedError

    def iter_terms(self):
        """Yield (term_id, fn) for every term: mains first, then active interactions.

        term_id is ("main", j) or ("inter", j, k). `fn` evaluates that ONE term on
        arbitrary inputs (1-col for main, 2-col for inter), returning CENTERED
        (deployment) outputs. This is the path for anything that wants the model's
        own pool-centering. The EVAL metrics do NOT use this — they re-center over
        their own sample, so they call raw_term_output instead.

        Yields:
            (term_id, fn) tuples.

        TODO:
            - main fn: raw_main(j, col) - main_centers[j]   (late-bind j=j)
            - inter fn: raw_inter(key, cols) - inter_centers.get(key, 0.0)  (late-bind key=key)
            - returned fns yield CENTERED outputs.
        """
        raise NotImplementedError

    def raw_term_output(self, term_id, x: torch.Tensor) -> torch.Tensor:
        """Evaluate ONE subnet on an arbitrary input matrix; return its RAW output.

        The accessor the evaluation harness is built around: pass any input matrix
        (e.g. a held-out test fold the subnet never trained on) and get that term's
        UNCENTERED output. Centering is deliberately deferred to the consumer,
        because the correct reference sample differs per metric and is NOT the
        model's deployment (pool) centering:
            * stability  → reducer re-centers over the TEST fold.
            * concurvity → the OLS intercept centers over the POOL.
        Do NOT subtract main_centers / inter_centers here. (For the pool-centered
        deployment value, use iter_terms / main_outputs instead.)

        Args:
            term_id: ("main", j) or ("inter", j, k).
            x: Input matrix, shape (batch_size, num_features). Full-width rows;
               this method selects the column(s) the term needs.

        Returns:
            Raw output vector, shape (batch_size, 1).

        TODO:
            - ("main", j):    raw = main_nns[j](x[:, j:j+1])            # NO center
            - ("inter", j, k): cols = encode + stack columns j, k as the subnet was
                               built; raw = inter_nns[f"{j},{k}"](cols)  # NO center
            - Reuse _encode_col so categorical handling matches construction.
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
