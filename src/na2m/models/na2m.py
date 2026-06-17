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
from torch.utils.data import DataLoader

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
    # Helpers
    # ------------------------------------------------------------------
    def _encode_col(self, x: torch.Tensor, j: int) -> torch.Tensor:
        """Encode column j into a float tensor suitable for subnet input.

        Used at runtime (forward pass / raw_term_output) to feed columns into
        FeatureNN. Must match the width that _inter_in_features promised at
        construction time.

        Args:
            x: Full input batch, shape (batch, num_features).
            j: Feature index to encode.

        Returns:
            (batch, 1) for numerical; (batch, n_levels) for categorical (one-hot).
        """
        meta = self.feature_meta[j]
        if meta.type == "num":
            col = x[:, j:j+1]  # (batch, 1)
            return col
        assert meta.n_levels is not None
        indices = x[:, j].long()
        col = torch.nn.functional.one_hot(indices, num_classes=meta.n_levels).float()  # (batch, n_levels)
        return col

    def _inter_in_features(self, j: int, k: int) -> int:
        """Compute interaction subnet input width at construction time.

        Called by add_interactions to set FeatureNN's in_features correctly.
        Must stay consistent with what _encode_col returns at runtime.

        Args:
            j: First feature index.
            k: Second feature index.

        Returns:
            Total input width: 1 per numerical feature, n_levels per categorical.
        """
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
    
    def active_interaction_pairs(self) -> list[tuple[int, int]]:
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
    # Structural mutation (rebuild the optimizer after ANY of these)
    # ------------------------------------------------------------------

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
        self._bias.requires_grad_(flag)

    # ------------------------------------------------------------------
    # Centering
    # ------------------------------------------------------------------
    def center_main_effects(self, pool_loader: DataLoader) -> None:
        """Fold each main subnet's current mean into _bias.

        Iterates over pool_loader in eval/no_grad mode, accumulates the sum of
        each main subnet's effective output (raw - already-accumulated offset),
        then divides by N to get the mean delta. That delta is added to
        main_centers (so main_outputs subtracts it on future forward passes) and
        to _bias (so predictions remain unchanged). 

        Call after Stage 1 AND after the single Stage-3 fine-tune.

        Args:
            pool_loader: DataLoader over the full training pool (X, y, idx).
                         Yields batches of (X_batch, _, _).
        """
        was_training = self.training
        self.eval()  # disable dropout — we want deterministic outputs
        device = self._bias.device
        with torch.no_grad():
            # accum[j] will hold the sum of subnet j's outputs across all N samples
            accum = torch.zeros(self.num_features, device=device)
            n = 0  # total sample count across all batches

            for X_batch, _, _ in pool_loader:
                X_batch = X_batch.to(device)
                for j in range(self.num_features):
                    # raw output of subnet j for this batch, shape (batch, 1)
                    raw = self.main_nns[j](X_batch[:, j:j+1])

                    # subtract any offset already accumulated from previous centering calls;
                    # this is what main_outputs subtracts during forward, so this reflects
                    # what the model currently contributes — with previous centering accounted for
                    output_with_previous_centering = raw - self.main_centers[j]

                    accum[j] += output_with_previous_centering.sum()
                n += X_batch.shape[0]

            for j in range(self.num_features):
                # mean output of subnet j over the full pool
                delta = accum[j] / n

                # tell main_outputs to subtract this going forward
                self.main_centers[j] += delta

                # compensate in the global bias so predictions don't change
                self._bias += delta

        if was_training:
            self.train()

    def center_interactions(self, pool_loader: DataLoader, fold_bias: bool) -> None:
        """Fold each active interaction subnet's current mean into its centering offset.

        Same accumulate-then-apply pattern as center_main_effects, but per interaction pair.
        fold_bias controls whether the delta is also folded into _bias:
            False — update inter_centers only; used DURING the Stage-2 sweep so that
                    candidates that get skipped/cut never contaminate _bias.
            True  — also add delta to _bias and mark _inter_folded[key]=True; used ONCE
                    after the survivor set is fixed and after Stage-3 fine-tune.

        Args:
            pool_loader: DataLoader over the full training pool. Yields (X_batch, _, _).
            fold_bias: Whether to fold the delta into _bias (see above).
        """
        was_training = self.training
        self.eval()  # disable dropout
        device = self._bias.device
        with torch.no_grad():
            pairs = self.active_interaction_pairs()

            # one accumulator per active pair, keyed by "j,k"
            accum = {f"{j},{k}": torch.zeros(1, device=device) for j, k in pairs}
            n = 0

            for X_batch, _, _ in pool_loader:
                X_batch = X_batch.to(device)
                for j, k in pairs:
                    key = f"{j},{k}"

                    # concatenate the two encoded columns to form the subnet input
                    col_j = self._encode_col(X_batch, j)
                    col_k = self._encode_col(X_batch, k)
                    feature_input = torch.cat([col_j, col_k], dim=1)

                    # raw subnet output for this batch
                    raw = self.inter_nns[key](feature_input)

                    # subtract previous centering offset — same idempotent trick as mains
                    output_with_previous_centering = raw - self.inter_centers.get(key, 0.0)

                    accum[key] += output_with_previous_centering.sum()
                n += X_batch.shape[0]

            for j, k in pairs:
                key = f"{j},{k}"
                delta = accum[key] / n

                # accumulate into the per-term offset (subtracted in inter_outputs)
                self.inter_centers[key] = self.inter_centers.get(key, 0.0) + delta

                if fold_bias:
                    # compensate in global bias so predictions don't change
                    self._bias += delta
                    self._inter_folded[key] = True

        if was_training:
            self.train()

    # ------------------------------------------------------------------
    # Per-term evaluation
    # ------------------------------------------------------------------
    def iter_subnets(self):
        """Yield (subnet_id, fn) for every subnet: mains first, then active interactions.

        subnet_id is ("main", j) or ("inter", j, k). `fn` evaluates that ONE subnet on
        arbitrary inputs (1-col for main, 2-col for inter), returning CENTERED
        (deployment) outputs. This is the path for anything that wants the model's
        own pool-centering. The EVAL metrics do NOT use this — they re-center over
        their own sample, so they call raw_subnet_output instead.

        Yields:
            (subnet_id, fn) tuples.

        TODO:
            - main fn: raw_main(j, col) - main_centers[j]   (late-bind j=j)
            - inter fn: raw_inter(key, cols) - inter_centers.get(key, 0.0)  (late-bind key=key)
            - returned fns yield CENTERED outputs.
        """
        raise NotImplementedError

    def raw_subnet_output(self, subnet_id, x: torch.Tensor) -> torch.Tensor:
        """Evaluate ONE subnet on an arbitrary input matrix; return its RAW output.

        The accessor the evaluation harness is built around: pass any input matrix
        (e.g. a held-out test fold the subnet never trained on) and get that subnet's
        UNCENTERED output. Centering is deliberately deferred to the consumer,
        because the correct reference sample differs per metric and is NOT the
        model's deployment (pool) centering:
            * stability  → reducer re-centers over the TEST fold.
            * concurvity → the OLS intercept centers over the POOL.
        Do NOT subtract main_centers / inter_centers here. (For the pool-centered
        deployment value, use iter_subnets / main_outputs instead.)

        Args:
            subnet_id: ("main", j) or ("inter", j, k).
            x: Input matrix, shape (batch_size, num_features). Full-width rows;
               this method selects the column(s) the subnet needs.

        Returns:
            Raw output vector, shape (batch_size, 1).
        """
        kind = subnet_id[0]

        if kind == "main":
            j = subnet_id[1]
            feature_input = x[:, j:j + 1]          # (batch, 1), same as main_outputs
            return self.main_nns[j](feature_input)

        # kind == "inter"
        j, k = subnet_id[1], subnet_id[2]
        key = f"{j},{k}"
        col_j = self._encode_col(x, j)             # (batch, 1) or (batch, n_levels_j)
        col_k = self._encode_col(x, k)             # (batch, 1) or (batch, n_levels_k)
        feature_input = torch.cat([col_j, col_k], dim=1)
        return self.inter_nns[key](feature_input)

    # ------------------------------------------------------------------
    # Forward
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

        """
        individual_outputs = []
        for j,k in self.active_interaction_pairs():
            key = f"{j},{k}"
            col_j = self._encode_col(x, j)   # (batch, 1) or (batch, n_levels_j)
            col_k = self._encode_col(x, k)   # (batch, 1) or (batch, n_levels_k)

            feature_input = torch.cat([col_j, col_k], dim=1)   # (batch, 1+1) or wider for cats
            feature_output = self.inter_nns[key](feature_input) - self.inter_centers.get(key, 0.0) 
            individual_outputs.append(feature_output)
        return individual_outputs

    def forward(self, x: torch.Tensor):
        """Forward pass: sum all term outputs, main and interaction terms + bias, with feature dropout.

        Args:
            x: Input batch, shape (batch_size, num_features).

        Returns:
            (out, dropout_out): out shape (batch_size,); dropout_out the
            per-term tensor after feature dropout (for the output penalty).

        """
        individual_outputs = self.main_outputs(x) + self.inter_outputs(x)  # list of (batch, 1); inter part is [] for arm A

        conc_out = torch.cat(individual_outputs, dim=1)
        dropout_out = self.dropout_layer(conc_out) 

        out = dropout_out.sum(dim=1) + self._bias
        return out, dropout_out


    # ------------------------------------------------------------------
    # GAMI-Net marginal-clarity penalty
    # ------------------------------------------------------------------

    def clarity_loss(self, x: torch.Tensor) -> torch.Tensor:
        """Marginal-clarity penalty: Σ_(j,k) (|mean f_j·f_jk| + |mean f_k·f_jk|).

        Penalises covariance between each interaction subnet and the two main effect
        subnets it is built from. Centering is required — mean(f·g) is only a
        covariance when both terms are zero-mean. Returns 0 when no interactions
        are active (arm A / Stage 1).

        Args:
            x: Input batch, shape (batch_size, num_features).

        Returns:
            Scalar penalty tensor.
        """
        penalty = torch.zeros(1, device=self._bias.device)

        for j, k in self.active_interaction_pairs():
            key = f"{j},{k}"

            # centered main outputs for both features in this pair
            f_j = self.main_nns[j](x[:, j:j+1]) - self.main_centers[j]
            f_k = self.main_nns[k](x[:, k:k+1]) - self.main_centers[k]

            # centered interaction output
            col_j = self._encode_col(x, j)
            col_k = self._encode_col(x, k)
            f_jk = self.inter_nns[key](torch.cat([col_j, col_k], dim=1)) - self.inter_centers.get(key, 0.0)

            # empirical covariance between the interaction and each main effect
            penalty = penalty + (f_j * f_jk).mean().abs() + (f_k * f_jk).mean().abs()

        return penalty
