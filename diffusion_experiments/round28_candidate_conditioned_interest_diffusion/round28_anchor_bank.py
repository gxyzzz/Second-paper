from __future__ import annotations
import gc, hashlib
from collections import defaultdict
import numpy as np
import torch

from diffusion_experiments.round28_candidate_conditioned_interest_diffusion import round28_core as c
from modules.attribute import attribute_z, build_item_matrices, build_profiles
from modules.coliftrec import fit_backgrounds, score_coliftrec
from modules.ranking import semantic_z_for_candidates, topk_from_embeddings
from pipelines.coliftrec import _params
from pipelines.dataset_config import load_dataset_config

class FrozenAnchorBank:
    """TRAIN-only frozen Full-CoLift boundary anchors, one deterministic 4-anchor bank per TRAIN event."""
    def __init__(self, seed, model, events, persist=True, force_rebuild=False):
        self.seed = int(seed)
        self.events = events
        self.model = model
        self.path = c.ASSETS / f'ROUND28_FROZEN_ANCHOR_BANK_SEED{seed}.npz'
        self.audit_path = c.EVID / f'ROUND28_FROZEN_ANCHOR_AUDIT_SEED{seed}.json'
        self.cfg = load_dataset_config('baby')
        if self.path.exists() and not force_rebuild:
            z = np.load(self.path)
            self.event_users = z['event_users'].astype(np.int64)
            self.event_pos = z['event_pos'].astype(np.int64)
            self.anchors = z['anchors'].astype(np.int64)
            self.anchor_ranks = z['anchor_ranks'].astype(np.int64)
            self.users = z['users'].astype(np.int64)
            self.ranked_top100 = z['ranked_top100'].astype(np.int64)
            if not np.array_equal(self.event_users, events.u) or not np.array_equal(self.event_pos, events.p):
                raise RuntimeError('frozen anchor event plan mismatch')
        else:
            self._build()
            if persist:
                np.savez_compressed(
                    self.path,
                    event_users=self.event_users.astype(np.int32),
                    event_pos=self.event_pos.astype(np.int32),
                    anchors=self.anchors.astype(np.int32),
                    anchor_ranks=self.anchor_ranks.astype(np.int16),
                    users=self.users.astype(np.int32),
                    ranked_top100=self.ranked_top100.astype(np.int32),
                )
        self.user_row = {int(u): i for i, u in enumerate(self.users.tolist())}
        self.event_groups = defaultdict(list)
        for eid, (u, p) in enumerate(zip(self.event_users.tolist(), self.event_pos.tolist())):
            self.event_groups[(int(u), int(p))].append(int(eid))
        self._epoch_counts = None

    @staticmethod
    def _choice(seed, user, event_id, lo, hi, salt):
        payload = f'{seed}:{user}:{event_id}:{lo}:{hi}:{salt}'.encode()
        x = int.from_bytes(hashlib.sha256(payload).digest()[:8], 'little')
        rng = np.random.default_rng(x)
        return rng.choice(np.arange(lo, hi, dtype=np.int64), size=2, replace=False)

    def _build(self):
        cfg = self.cfg
        p = cfg['resolved_paths']
        ccfg = cfg['coliftrec']
        params = _params(ccfg)
        enabled = {m: bool(ccfg[m]['enabled']) for m in ('text', 'attribute', 'visual')}
        histories = self.events.histories
        pseudo = self.events.pseudo
        pusers = self.events.pusers
        self.users = np.asarray([u for u, h in enumerate(histories) if h], dtype=np.int64)

        self.model.eval()
        with torch.no_grad():
            fu, fi = self.model.forward(test=True)
            ci, cs = topk_from_embeddings(fu, fi, self.users, histories, 100, 1024)
            pi, ps = topk_from_embeddings(fu, fi, pusers, pseudo, 100, 1024)

        ztp, _ = semantic_z_for_candidates(p['text_feature'], pseudo, pusers, pi, 256)
        ztf, _ = semantic_z_for_candidates(p['text_feature'], histories, self.users, ci, 256)
        zvp, _ = semantic_z_for_candidates(p['visual_feature'], pseudo, pusers, pi, 128)
        zvf, _ = semantic_z_for_candidates(p['visual_feature'], histories, self.users, ci, 128)
        ac = ccfg['attribute']
        mats, _ = build_item_matrices(
            p['metadata'], self.model.n_items,
            min_df=int(ac.get('tfidf_min_df', 2)),
            max_df=float(ac.get('tfidf_max_df', .8)),
            description_len=int(ac.get('description_len', 128)),
            weights=ac.get('weights'),
        )
        pp = build_profiles(mats, pseudo, self.model.n_items)
        fp = build_profiles(mats, histories, self.model.n_items)
        zap, _ = attribute_z(mats, pp, pusers, pi, 256, ac.get('weights'))
        zaf, _ = attribute_z(mats, fp, self.users, ci, 256, ac.get('weights'))
        bg = fit_backgrounds(pi, ztp, zap, zvp, self.model.n_items)
        full, _ = score_coliftrec(cs, ci, ztf, zaf, zvf, bg, params, enabled)
        order = np.argsort(-full, axis=1, kind='stable')
        self.ranked_top100 = np.take_along_axis(ci, order, axis=1).astype(np.int64)

        self.event_users = self.events.u.copy()
        self.event_pos = self.events.p.copy()
        row = {int(u): i for i, u in enumerate(self.users.tolist())}
        self.anchors = np.empty((len(self.event_users), c.N_ANCHORS), dtype=np.int64)
        self.anchor_ranks = np.empty_like(self.anchors)
        for eid, (u, pos) in enumerate(zip(self.event_users.tolist(), self.event_pos.tolist())):
            r = row[int(u)]
            ranked = self.ranked_top100[r]
            chosen_rank0 = list(self._choice(self.seed, u, eid, 10, 20, 0))
            chosen_rank0 += list(self._choice(self.seed, u, eid, 20, 30, 1))
            bad = set(histories[int(u)])
            bad.add(int(pos))
            used = set()
            for j, start in enumerate(chosen_rank0):
                k = int(start)
                while k < len(ranked) and (int(ranked[k]) in bad or int(ranked[k]) in used):
                    k += 1
                if k >= len(ranked):
                    k = 0
                    while k < len(ranked) and (int(ranked[k]) in bad or int(ranked[k]) in used):
                        k += 1
                if k >= len(ranked):
                    raise RuntimeError(f'no legal frozen anchor for event {eid}')
                self.anchors[eid, j] = int(ranked[k])
                self.anchor_ranks[eid, j] = k + 1
                used.add(int(ranked[k]))

    def bank_hash(self):
        h = hashlib.sha256()
        for arr in (self.event_users, self.event_pos, self.anchors, self.anchor_ranks):
            h.update(np.ascontiguousarray(arr).tobytes())
        return h.hexdigest()

    def reset_epoch(self):
        self._epoch_counts = defaultdict(int)

    def resolve_event_ids(self, users_t, pos_t):
        if self._epoch_counts is None:
            self.reset_epoch()
        users = users_t.detach().cpu().numpy().astype(np.int64)
        pos = pos_t.detach().cpu().numpy().astype(np.int64)
        out = np.empty(len(users), dtype=np.int64)
        for i, (u, p) in enumerate(zip(users.tolist(), pos.tolist())):
            key = (int(u), int(p))
            ids = self.event_groups.get(key)
            if not ids:
                raise RuntimeError(f'event key absent from frozen bank: {key}')
            k = self._epoch_counts[key]
            if k >= len(ids):
                raise RuntimeError(f'event duplicate count exceeded within epoch: {key}')
            out[i] = ids[k]
            self._epoch_counts[key] = k + 1
        return out

    def anchors_for_event_ids(self, event_ids, epoch, device):
        event_ids = np.asarray(event_ids, dtype=np.int64)
        slot = int(epoch) % c.N_ANCHORS
        return torch.as_tensor(self.anchors[event_ids, slot], device=device, dtype=torch.long)

    def all_anchors_for_event_ids(self, event_ids, device):
        event_ids = np.asarray(event_ids, dtype=np.int64)
        return [
            torch.as_tensor(self.anchors[event_ids, j], device=device, dtype=torch.long)
            for j in range(c.N_ANCHORS)
        ]

    def provenance_audit(self):
        hist = self.events.histories
        legal = True
        for eid, (u, p) in enumerate(zip(self.event_users.tolist(), self.event_pos.tolist())):
            bad = set(hist[int(u)])
            bad.add(int(p))
            if any(int(x) in bad for x in self.anchors[eid]):
                legal = False
                break
        ranks = self.anchor_ranks.reshape(-1)
        return {
            'protocol': c.PROTOCOL,
            'seed': self.seed,
            'TRAIN_ONLY': True,
            'source': 'frozen initial MSCA candidate Top100 reranked by frozen Full CoLiftRec',
            'event_count': int(len(self.event_users)),
            'anchors_per_event': int(c.N_ANCHORS),
            'bank_hash': self.bank_hash(),
            'anchor_rank': c.stats(ranks),
            'rank11_20_fraction': float(((ranks >= 11) & (ranks <= 20)).mean()),
            'rank21_30_fraction': float(((ranks >= 21) & (ranks <= 30)).mean()),
            'all_anchors_exclude_train_history_and_current_positive': bool(legal),
            'deterministic_schedule': 'epoch % 4',
            'VALIDATION_USED_FOR_CONSTRUCTION': False,
            'TEST_ACCESSED': False,
            'SPORTS_ACCESSED': False,
            'ELECTRONICS_ACCESSED': False,
        }

def build_anchor_audit(seed=999):
    model, config, td, vd, pack = c.instantiate_shared(seed)
    events = c.TrainEvents(model)
    bank = FrozenAnchorBank(seed, model, events, persist=True)
    a1 = bank.provenance_audit()
    h1 = bank.bank_hash()
    bank2 = FrozenAnchorBank(seed, model, events, persist=False, force_rebuild=True)
    h2 = bank2.bank_hash()
    out = dict(a1)
    out['repeat_hash'] = h2
    out['repeat_exact'] = bool(h1 == h2 and np.array_equal(bank.anchors, bank2.anchors))
    out['PASS'] = bool(out['repeat_exact'] and out['all_anchors_exclude_train_history_and_current_positive'])
    c.json_write(c.EVID / f'ROUND28_FROZEN_ANCHOR_AUDIT_SEED{seed}.json', out)
    del model, bank, bank2
    torch.cuda.empty_cache()
    gc.collect()
    return out
