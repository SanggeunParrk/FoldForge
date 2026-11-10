"""trunk_seed x diffusion_seed batching: NT retained trunks, ONE sampler run over NT * A samples.

Lifted from the end-to-end harness ``miniworld-engine-scratch/tsb/ff_e2e55b.py`` (measured there on Protenix v1 L512: 5 x 5 end to end 7.35 -> 6.82 s on the
engine branch, 6.44 -> 5.90 s on engine 2.4.0) and generalised: no fixed L, the trunk count comes from the call, nothing is asserted about shapes.

What changes against the per-trunk_seed sampler: the denoiser sees ``[A, NT, ...]`` instead of ``[A, ...]``, the token DiT / the atom transformers run ONCE over the NT
folds (each fold keeps its own pair, conditioning and atom pair: the engine's grouped-bias attention reads sample s's pair from fold s % nbg) and the rest of the step
(coordinate scaling, the encoder / decoder glue, the reconstruction) is per fold. The Euler schedule, the augmentation and the realignment are the production
sampler's own: ``AlphaFold3._sample_diffusion`` is called with ``num_samples = A * NT`` and only the head's forward / conditioning table are swapped for the grouped ones.

``GroupedUnsupported`` is raised (before anything is sampled) when the diffusion transformers are not stacks of engine blocks (ESMFold2's atom stack) or when this model has no such head; the caller
then samples the folds one after another."""

from __future__ import annotations

from typing import Any

import torch


class GroupedUnsupported(RuntimeError):
    """The grouped denoiser cannot serve this model / shape; sample the folds separately."""


class GroupedSampler:
    def __init__(self, net: Any, *, atoms_grouped: bool = True) -> None:
        import foldforge.modules.dense.atom_cross_attention as aca
        import foldforge.modules.dense.diffusion_head as dh

        self.net, self.aca, self.dh, self.atoms_grouped = net, aca, dh, atoms_grouped
        head = getattr(net, "diffusion_head", None)
        if (
            head is None
            or not hasattr(head, "_conditioned")
            or not hasattr(head, "atom_cross_att_encoder")
        ):
            raise GroupedUnsupported("this model has no dense diffusion head")
        self.head = head
        self.enc_mod, self.dec_mod = (
            head.atom_cross_att_encoder,
            head.atom_cross_att_decoder,
        )
        # FoldForge's graph holds the engine blocks directly: each transformer is a stack with a ``blocks`` ModuleList of engine DiT blocks (ESMFold2's atom stack is
        # FoldForge's own dense transformer: no ``blocks`` -> not groupable).
        stacks = (
            head.transformer,
            self.enc_mod.atom_transformer_encoder,
            self.dec_mod.atom_transformer_decoder,
        )
        if not all(
            isinstance(getattr(s, "blocks", None), torch.nn.ModuleList)
            and len(s.blocks)
            for s in stacks
        ):
            raise GroupedUnsupported(
                "a diffusion transformer of this model is not a stack of engine blocks"
            )
        self.built_tok, self.built_enc, self.built_dec = (
            list(s.blocks) for s in stacks
        )
        self.state: dict[str, Any] = {}
        self.nt = 0

    # ---- per-fold conditioning, made once per fold ---------------------------------------------------------------------------------------------------
    def _conditioned(self, tr: Any) -> dict[str, Any]:
        sb, pc, lite, epoch, km, atom = self.head._conditioned(
            tr.diffusion.batch, tr.diffusion.embeddings, True
        )
        return {
            "batch": tr.diffusion.batch,
            "lite": lite,
            "base": sb,
            "pair": pc,
            "epoch": epoch,
            "km": km,
            "atom": atom,
        }

    def _table_of(self, c: dict[str, Any], levels: torch.Tensor) -> torch.Tensor:
        head, dh = self.head, self.dh
        base = c["base"]
        per_pass = max(1, dh.MAX_TABLE_ROWS // max(1, base.numel() // base.shape[-1]))
        rows = []
        for lv in levels.split(per_pass):
            term = head._noise_term(lv)
            term = term.reshape(term.shape[0], *(1,) * (base.ndim - 1), term.shape[-1])
            rows.append(head._finish_single_conditioning(base[None] + term))
        return rows[0] if len(rows) == 1 else torch.cat(rows)

    def _prep(self, cs: list[dict[str, Any]]) -> dict[str, Any]:
        nt = len(cs)
        length = cs[0]["pair"].shape[0]
        st: dict[str, Any] = {"cond": cs, "L": length}
        st["pair_all"] = torch.stack(
            [c["pair"] for c in cs], 0
        ).contiguous()  # [NT, L, L, 128]
        st["km_all"] = cs[0]["km"].expand(nt, length)
        ea = [c["atom"] for c in cs]
        st["atom_pair"] = torch.cat(
            [e.engine_pair for e in ea], 0
        ).contiguous()  # [NT, nwin, 32, 128, 16]
        atoms = ea[0].engine_key_mask.shape[-1]
        st["atom_km"] = ea[0].engine_key_mask.expand(
            nt, atoms
        )  # one input's key mask, expanded over NT
        st["atoms"] = atoms
        st["cond_enc"] = torch.stack(
            [e.queries_single_cond.reshape(atoms, -1) for e in ea], 0
        ).contiguous()  # [NT, atoms, 128]
        post = self.dec_mod.post_atom_cond_layer_norm
        st["cond_dec"] = (
            st["cond_enc"]
            if post is None
            else torch.stack(
                [post(e.queries_single_cond).reshape(atoms, -1) for e in ea], 0
            ).contiguous()
        )
        return st

    # ---- the encoder / decoder step split around their transformer (copies of AtomCrossAttEncoder.step / AtomCrossAttDecoder.forward) -------------------
    def _enc_pre(self, b: int, token_atoms_act: torch.Tensor) -> torch.Tensor:
        c = self.state["cond"][b]
        st = c["atom"]
        queries_act = self.aca.atom_layout.convert(
            c["batch"].atom_cross_att.token_atoms_to_queries,
            token_atoms_act,
            layout_axes=(-3, -2),
        )
        queries_act = self.enc_mod.atom_positions_to_features(queries_act)
        if self.enc_mod.spec.atom_chiral_features:
            queries_act = queries_act + self.enc_mod._chiral_features(
                c["batch"], token_atoms_act
            )
        queries_act *= st.queries_mask[..., None]
        queries_act += st.query_base
        return queries_act

    def _enc_post(self, b: int, queries_act: torch.Tensor) -> Any:
        aca = self.aca
        c = self.state["cond"][b]
        st = c["atom"]
        batch_b = c["batch"]
        queries_act *= st.queries_mask[..., None]
        skip = queries_act.clone()
        queries_act = self.enc_mod.project_atom_features_for_aggr(queries_act)
        token_act = aca._pool_engine(
            queries_act,
            batch_b.atom_cross_att.queries_to_token_atoms,
            batch_b.predicted_structure_info.atom_mask,
        )
        if token_act is None:
            raise GroupedUnsupported("the engine pooling did not serve")
        return aca.AtomCrossAttEncoderOutput(
            token_act=token_act,
            skip_connection=skip,
            queries_mask=st.queries_mask,
            queries_single_cond=st.queries_single_cond,
            keys_mask=st.keys_mask,
            keys_single_cond=st.keys_single_cond,
            pair_cond=st.pair_cond,
            pair_mask=st.pair_mask,
            rope_q=st.rope_q,
            rope_k=st.rope_k,
            engine_pair=st.engine_pair,
            engine_key_mask=st.engine_key_mask,
            engine_uid=st.engine_uid,
            engine_even=st.engine_even,
        )

    def _dec_pre(self, b: int, token_act: torch.Tensor, enc: Any) -> torch.Tensor:
        aca, dec_mod = self.aca, self.dec_mod
        batch_b = self.state["cond"][b]["batch"]
        token_act = dec_mod.project_token_features_for_broadcast(token_act)
        num_token, per_tok = batch_b.atom_cross_att.queries_to_token_atoms.shape
        tok_atom = torch.broadcast_to(
            token_act[..., :, None, :],
            (*token_act.shape[:-2], num_token, per_tok, dec_mod.per_atom_channels),
        )
        q = aca.atom_layout.convert(
            batch_b.atom_cross_att.token_atoms_to_queries,
            tok_atom,
            layout_axes=(-3, -2),
        )
        q += enc.skip_connection
        q *= enc.queries_mask[..., None]
        return q

    def _dec_post(self, b: int, queries_act: torch.Tensor, enc: Any) -> torch.Tensor:
        aca, dh = self.aca, self.dh
        batch_b = self.state["cond"][b]["batch"]
        queries_act *= enc.queries_mask[..., None]
        upd = self.dec_mod.atom_features_to_position_update(
            self.dec_mod.atom_features_layer_norm(queries_act)
        )
        return aca.atom_layout.convert(
            batch_b.atom_cross_att.queries_to_token_atoms, upd, layout_axes=(-3, -2)
        )

    def _atom_stack(
        self, built: list, xs: list[torch.Tensor], cond_all: torch.Tensor
    ) -> list[torch.Tensor]:
        """The atom transformer blocks ONCE over [A, NT, atoms, C]."""
        a_, subsets, queries, ch = xs[0].shape
        atoms = subsets * queries
        x = torch.stack([t.reshape(a_, atoms, ch) for t in xs], 1)  # [A, NT, atoms, C]
        cnd = cond_all[None].expand(a_, -1, -1, -1)
        for blk in built:
            x = blk(x, cnd, self.state["atom_pair"], self.state["atom_km"])
        return [x[:, b].reshape(a_, subsets, queries, ch) for b in range(len(xs))]

    def _run_token(
        self, acts: list[torch.Tensor], single_cond: torch.Tensor, a: int
    ) -> torch.Tensor:
        length, nt = self.state["L"], self.nt
        x = torch.stack(acts, 1)  # [A, NT, L, 768]
        cnd = single_cond.reshape(nt, length, -1)[None].expand(a, -1, -1, -1)
        for blk in self.built_tok:
            x = blk(x, cnd, self.state["pair_all"], self.state["km_all"])
        return x

    def _forward(
        self,
        positions_noisy: torch.Tensor,
        noise_level: torch.Tensor,
        batch: Any = None,
        embeddings: Any = None,
        use_conditioning: bool = True,
        single_cond: torch.Tensor | None = None,
    ) -> torch.Tensor:
        dh, head, nt = self.dh, self.head, self.nt
        cs = self.state["cond"]
        a = positions_noisy.shape[0] // nt
        coords = positions_noisy.view(a, nt, *positions_noisy.shape[1:])
        acts, encs, qs = [], [], []
        for b, c in enumerate(cs):
            amask = c["batch"].predicted_structure_info.atom_mask
            act = dh.scale_coordinates(
                coords[:, b], noise_level, dh.SIGMA_DATA, mask=amask
            )
            if self.atoms_grouped:
                qs.append(self._enc_pre(b, act))
            else:
                enc = self.enc_mod.step(c["batch"], act, c["atom"])
                encs.append(enc)
                acts.append(enc.token_act)
        if self.atoms_grouped:
            qs = self._atom_stack(self.built_enc, qs, self.state["cond_enc"])
            for b in range(nt):
                enc = self._enc_post(b, qs[b])
                encs.append(enc)
                acts.append(enc.token_act)
        if (
            single_cond is None
        ):  # no sampler graph (graphs off): no precomputed table, so make this step's row here
            single_cond = self._table(batch, embeddings, noise_level.reshape(-1)[:1])[0]
        for b in range(nt):
            tsc = single_cond[b]
            if head.single_cond_embedding_norm is None:
                acts[b] = acts[b] + head.single_cond_embedding_projection(tsc)
            else:
                acts[b] += head.single_cond_embedding_projection(
                    head.single_cond_embedding_norm(tsc)
                )
        x = self._run_token(acts, single_cond, a)
        if self.atoms_grouped:
            dq = [
                self._dec_pre(b, head.output_norm(x[:, b]), encs[b]) for b in range(nt)
            ]
            dq = self._atom_stack(self.built_dec, dq, self.state["cond_dec"])
            upds = [self._dec_post(b, dq[b], encs[b]) for b in range(nt)]
        else:
            upds = [
                self.dec_mod(
                    batch=c["batch"], token_act=head.output_norm(x[:, b]), enc=encs[b]
                )
                for b, c in enumerate(cs)
            ]
        outs = []
        for b, c in enumerate(cs):
            amask = c["batch"].predicted_structure_info.atom_mask
            outs.append(
                dh.reconstruct_coordinates(
                    coords[:, b], upds[b], noise_level, dh.SIGMA_DATA, mask=amask
                )
            )
        return torch.stack(outs, 1).reshape(positions_noisy.shape)

    def _table(self, batch: Any, embeddings: Any, levels: torch.Tensor) -> torch.Tensor:
        return torch.stack(
            [self._table_of(c, levels) for c in self.state["cond"]], 1
        )  # [steps, NT, L, 384]

    # ---- the call ----------------------------------------------------------------------------------------------------------------------------------
    def sample(self, trunks: list[Any], samples: int) -> torch.Tensor:
        """Positions ``[samples * NT, ...]`` (fold-major inside: row ``a * NT + b`` is sample ``a`` of fold ``b``)."""
        net, head = self.net, self.head
        self.nt = len(trunks)
        cs = [self._conditioned(t) for t in trunks]
        self.state.clear()
        self.state.update(self._prep(cs))
        head.forward, head.conditioning_table = self._forward, self._table
        saved = net.num_samples
        net.num_samples = samples * self.nt
        try:
            out = net._sample_diffusion(
                trunks[0].diffusion.batch, trunks[0].diffusion.embeddings
            )
        finally:
            net.num_samples = saved
            del head.forward, head.conditioning_table
            head._fold = None
            self.state.clear()
        return out["atom_positions"]
