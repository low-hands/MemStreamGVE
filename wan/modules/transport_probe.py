"""
ΔKV Transport Probe (EXPERIMENT_PLAN Block 1, FINAL_PROPOSAL §4).

GO/NO-GO experiment for the central, falsifiable claim of the project:

    The edit can be represented as a *source-anchored foreground KV residual*
        ΔK = K_tgt_fg - K_src_fg ,  ΔV = V_tgt_fg - V_src_fg
    and transported to a later shot by reconstruction
        K_hat_tgt_cur = K_src_cur + ΔK_retrieved
        V_hat_tgt_cur = V_src_cur + ΔV_retrieved
    so that the *current* source branch carries current geometry/pose/lighting
    while the stored residual carries only the edit transformation.

This module is intentionally OFFLINE and MINIMAL. It does NOT build the full
online EditResidualMemory system (that is Block 3). It only:
  1. DUMP : record per-(layer, timestep-bucket) foreground KV from shot A.
  2. INJECT (C0/C1/C2): overwrite the target-branch foreground KV on shot B
     right before attention, under three conditions, so the only variable that
     changes across conditions is the *representation choice*.

Conditions (FINAL_PROPOSAL §4):
  C0  no persistent memory          -> probe is a no-op (vanilla StreamGVE).
  C1  direct target KV capsule      -> K_hat = retrieved K_tgt_A.
  C2  source-anchored residual      -> K_hat = K_src_cur_B + (K_tgt-K_src)_A.

IMPORTANT — RoPE space:
  Injection is performed in PRE-RoPE K/V space. The residual hypothesis assumes
  ΔK is position-agnostic; roped K carries shot-specific absolute-position phase,
  so transporting a roped residual across shots would inject a spurious phase
  mismatch and could fake a NO-GO. By dumping/injecting before causal_rope_apply,
  the residual stays position-free and is re-roped at the current shot's
  positions, which is the faithful implementation of §3.1.

Retrieval (per current foreground token j on shot B):
  anchor a_cur_j = flatten(K_src_cur_B[j])            # [H*D]
  sim_i = cos(a_cur_j, a_i) over stored anchors a_i   # a_i = flatten(K_src_A_i)
  w = softmax(TopK(sim)/temp)
  Delta_j = sum_i w_i * Delta_i                        # [H, D]

The probe is keyed by (layer_idx, timestep_bucket) so we can later report which
layers/timesteps carry transportable identity (bonus probe in the plan).
"""

import os
import torch


VALID_MODES = {"off", "dump", "C0", "C1", "C2"}


class TransportProbe:
    """Offline ΔKV transport probe. One instance per inference run.

    Parameters
    ----------
    mode : str
        One of {"off","dump","C0","C1","C2"}.
          - "dump"      : shot A pass; records fg KV, writes to ``save_path`` at end.
          - "C0"        : shot B pass; no-op (vanilla baseline).
          - "C1"/"C2"   : shot B pass; loads ``capsule_path`` and injects.
          - "off"       : disabled.
    layers : iterable[int] or None
        Restrict dump/inject to these self-attn layer indices (e.g. range(12,27)).
        None means all layers.
    save_path : str or None
        Where to write capsules at end of a "dump" run (.pt).
    capsule_path : str or None
        Where to read capsules from for C1/C2 runs (.pt produced by a dump run).
    topk : int
        Number of nearest stored anchors blended per query token.
    temp : float
        Softmax temperature over anchor cosine similarity.
    max_per_key : int
        Cap on stored prototypes per (layer, bucket). Excess is randomly
        subsampled at save time. Keeps disk/compute bounded; the probe does not
        need online k-means (that is Block 3).
    """

    def __init__(
        self,
        mode="off",
        layers=None,
        save_path=None,
        capsule_path=None,
        topk=4,
        temp=0.07,
        max_per_key=512,
    ):
        assert mode in VALID_MODES, f"probe mode must be one of {VALID_MODES}, got {mode}"
        self.mode = mode
        self.layers = set(int(l) for l in layers) if layers is not None else None
        self.save_path = save_path
        self.capsule_path = capsule_path
        self.topk = int(topk)
        self.temp = float(temp)
        self.max_per_key = int(max_per_key)

        # dump store: {(layer, bucket): {"anchor":[N,HD], "dK":[N,H,D], "dV":[N,H,D],
        #                                 "Ktgt":[N,H,D], "Vtgt":[N,H,D]}}
        self._store = {}
        # loaded capsules (same structure), tensors moved to device lazily.
        self._capsules = None
        self._loaded_device = None

        if self.mode in ("C1", "C2"):
            assert self.capsule_path is not None, "C1/C2 require capsule_path"
            assert os.path.exists(self.capsule_path), (
                f"capsule_path not found: {self.capsule_path}"
            )

    # ------------------------------------------------------------------ utils
    @property
    def active(self):
        return self.mode != "off"

    @property
    def is_dump(self):
        return self.mode == "dump"

    @property
    def is_inject(self):
        return self.mode in ("C1", "C2")

    def _layer_enabled(self, layer_idx):
        return self.layers is None or int(layer_idx) in self.layers

    def _load_capsules(self, device):
        if self._capsules is not None and self._loaded_device == device:
            return
        raw = torch.load(self.capsule_path, map_location="cpu")
        caps = {}
        for k, v in raw.items():
            caps[k] = {
                name: t.to(device=device, dtype=torch.float32)
                for name, t in v.items()
            }
        self._capsules = caps
        self._loaded_device = device

    # ------------------------------------------------------------------ dump
    @torch.no_grad()
    def record(self, layer_idx, bucket, k_src_fg, v_src_fg, k_trg_fg, v_trg_fg):
        """Record foreground KV prototypes from a shot-A dual forward.

        All tensors are PRE-RoPE and shaped [n_fg, H, D] (single batch element).
        """
        if not self.is_dump or not self._layer_enabled(layer_idx):
            return
        if k_src_fg.shape[0] == 0:
            return
        key = (int(layer_idx), int(bucket))
        anchor = k_src_fg.reshape(k_src_fg.shape[0], -1).float()  # [N, H*D]
        dK = (k_trg_fg - k_src_fg).float()
        dV = (v_trg_fg - v_src_fg).float()
        entry = self._store.setdefault(
            key, {"anchor": [], "dK": [], "dV": [], "Ktgt": [], "Vtgt": []}
        )
        entry["anchor"].append(anchor.cpu())
        entry["dK"].append(dK.cpu())
        entry["dV"].append(dV.cpu())
        entry["Ktgt"].append(k_trg_fg.float().cpu())
        entry["Vtgt"].append(v_trg_fg.float().cpu())

    @torch.no_grad()
    def save(self):
        """Concatenate, optionally subsample, and write capsules to disk."""
        if not self.is_dump:
            return
        assert self.save_path is not None, "dump mode requires save_path"
        out = {}
        for key, entry in self._store.items():
            anchor = torch.cat(entry["anchor"], dim=0)
            dK = torch.cat(entry["dK"], dim=0)
            dV = torch.cat(entry["dV"], dim=0)
            Ktgt = torch.cat(entry["Ktgt"], dim=0)
            Vtgt = torch.cat(entry["Vtgt"], dim=0)
            n = anchor.shape[0]
            if n > self.max_per_key:
                idx = torch.randperm(n)[: self.max_per_key]
                anchor, dK, dV = anchor[idx], dK[idx], dV[idx]
                Ktgt, Vtgt = Ktgt[idx], Vtgt[idx]
            out[key] = {
                "anchor": anchor,
                "dK": dK,
                "dV": dV,
                "Ktgt": Ktgt,
                "Vtgt": Vtgt,
            }
        os.makedirs(os.path.dirname(os.path.abspath(self.save_path)), exist_ok=True)
        torch.save(out, self.save_path)
        n_keys = len(out)
        n_proto = sum(v["anchor"].shape[0] for v in out.values())
        print(f"[transport_probe] saved {n_proto} prototypes over {n_keys} "
              f"(layer,bucket) keys -> {self.save_path}")

    # ---------------------------------------------------------------- inject
    @torch.no_grad()
    def reconstruct(self, layer_idx, bucket, k_src_cur_fg, v_src_cur_fg):
        """Return injected (k_hat_fg, v_hat_fg) for shot-B foreground tokens.

        Inputs are PRE-RoPE [n_fg, H, D] for the current shot-B source branch.
        Returns None if the probe should not modify this layer/bucket (caller
        leaves the original target KV untouched -> that is the C0 path, and also
        the fallback when no capsules exist for this key).
        """
        if self.mode == "C0" or not self.is_inject:
            return None
        if not self._layer_enabled(layer_idx):
            return None
        if k_src_cur_fg.shape[0] == 0:
            return None

        self._load_capsules(k_src_cur_fg.device)
        key = (int(layer_idx), int(bucket))
        cap = self._capsules.get(key, None)
        if cap is None or cap["anchor"].shape[0] == 0:
            return None

        H, D = k_src_cur_fg.shape[1], k_src_cur_fg.shape[2]
        q = k_src_cur_fg.reshape(k_src_cur_fg.shape[0], -1).float()  # [Nq, HD]
        a = cap["anchor"]  # [Ns, HD]

        # cosine similarity
        qn = torch.nn.functional.normalize(q, dim=-1)
        an = torch.nn.functional.normalize(a, dim=-1)
        sim = qn @ an.t()  # [Nq, Ns]

        k = min(self.topk, sim.shape[1])
        top_sim, top_idx = sim.topk(k, dim=-1)  # [Nq, k]
        w = torch.softmax(top_sim / self.temp, dim=-1)  # [Nq, k]

        if self.mode == "C2":
            dK = cap["dK"].reshape(cap["dK"].shape[0], -1)  # [Ns, H*D]
            dV = cap["dV"].reshape(cap["dV"].shape[0], -1)
            gathered_dK = dK[top_idx]  # [Nq, k, H*D]
            gathered_dV = dV[top_idx]
            blend_dK = (w.unsqueeze(-1) * gathered_dK).sum(dim=1).reshape(-1, H, D)
            blend_dV = (w.unsqueeze(-1) * gathered_dV).sum(dim=1).reshape(-1, H, D)
            k_hat = k_src_cur_fg.float() + blend_dK
            v_hat = v_src_cur_fg.float() + blend_dV
        else:  # C1 — direct target KV capsule
            Kt = cap["Ktgt"].reshape(cap["Ktgt"].shape[0], -1)  # [Ns, H*D]
            Vt = cap["Vtgt"].reshape(cap["Vtgt"].shape[0], -1)
            gathered_Kt = Kt[top_idx]
            gathered_Vt = Vt[top_idx]
            k_hat = (w.unsqueeze(-1) * gathered_Kt).sum(dim=1).reshape(-1, H, D)
            v_hat = (w.unsqueeze(-1) * gathered_Vt).sum(dim=1).reshape(-1, H, D)

        return k_hat.to(k_src_cur_fg.dtype), v_hat.to(v_src_cur_fg.dtype)


def build_probe_from_shared_dict(shared_dict):
    """Helper: lazily construct/cache a TransportProbe on the shared_dict.

    The pipeline puts a small config dict at shared_dict['probe_cfg']; this
    builds the probe object once and stores it at shared_dict['probe'].
    """
    if shared_dict is None:
        return None
    if shared_dict.get("probe", None) is not None:
        return shared_dict["probe"]
    cfg = shared_dict.get("probe_cfg", None)
    if not cfg or cfg.get("mode", "off") == "off":
        return None
    probe = TransportProbe(
        mode=cfg.get("mode", "off"),
        layers=cfg.get("layers", None),
        save_path=cfg.get("save_path", None),
        capsule_path=cfg.get("capsule_path", None),
        topk=cfg.get("topk", 4),
        temp=cfg.get("temp", 0.07),
        max_per_key=cfg.get("max_per_key", 512),
    )
    shared_dict["probe"] = probe
    return probe
