from __future__ import annotations
import math
import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize
import torch

from diffusion_experiments.round28_candidate_conditioned_interest_diffusion import round28_core as c
from modules.attribute import build_item_matrices, build_profiles, attribute_z, _sparse_candidate_scores
from modules.coliftrec import fit_backgrounds
from modules.ranking import histories_csr, l2_normalize_rows, semantic_z_for_candidates, topk_from_embeddings
from pipelines.coliftrec import _params
from pipelines.dataset_config import load_dataset_config

def _masked_z(x, mask, eps=1e-8):
    x = np.asarray(x, np.float32)
    m = np.asarray(mask, bool)
    n = m.sum(1, keepdims=True).astype(np.float32)
    den = np.maximum(n, 1.0)
    mean = (x * m).sum(1, keepdims=True) / den
    var = (((x - mean) * m) ** 2).sum(1, keepdims=True) / den
    std = np.sqrt(np.maximum(var, eps ** 2))
    return (((x - mean) / std) * m).astype(np.float32)

def _masked_softmax(a, mask):
    a = np.asarray(a, np.float32)
    m = np.asarray(mask, bool)
    out = np.zeros_like(a)
    for r in range(len(a)):
        ix = np.flatnonzero(m[r])
        if len(ix):
            v = a[r, ix]
            e = np.exp(v - v.max())
            out[r, ix] = e / np.maximum(e.sum(), 1e-12)
    return out

def _dense_scores(feat, profiles, items, batch=256):
    out = np.zeros(items.shape, np.float32)
    for st in range(0, len(items), batch):
        en = min(st + batch, len(items))
        out[st:en] = np.einsum('bkd,bd->bk', feat[items[st:en]], profiles[st:en], optimize=True)
    return out

def _norm_rows(x):
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)

class CoherentInterestContext:
    """TRAIN-only candidate-conditioned coherent-interest target context."""
    def __init__(self, seed, model):
        self.seed = int(seed)
        self.cfg = load_dataset_config('baby')
        p = self.cfg['resolved_paths']
        self.paths = p
        self.ccfg = self.cfg['coliftrec']
        self.params = _params(self.ccfg)
        self.n_items = model.n_items
        self.n_users = model.n_users

        df = pd.read_csv(p['interaction'], sep='\t', usecols=['userID', 'itemID', 'timestamp', 'x_label'])
        df['_row'] = np.arange(len(df), dtype=np.int64)
        tr = df[df.x_label == 0].copy().sort_values(['userID', 'timestamp', '_row'], kind='stable')
        self.histories = [[] for _ in range(self.n_users)]
        for u, g in tr.groupby('userID', sort=False):
            self.histories[int(u)] = g.itemID.astype(np.int64).tolist()
        self.counts = np.asarray([len(h) for h in self.histories], np.int64)
        self.pseudo = [list(h[:-1]) if len(h) >= 2 else list(h) for h in self.histories]
        self.pusers = np.asarray([u for u, h in enumerate(self.histories) if len(h) >= 2], np.int64)

        self.text = l2_normalize_rows(np.asarray(np.load(p['text_feature'], mmap_mode='r'), dtype=np.float32))
        self.visual = l2_normalize_rows(np.asarray(np.load(p['visual_feature'], mmap_mode='r'), dtype=np.float32))
        ac = self.ccfg['attribute']
        self.attr_weights = ac.get('weights') or {'title': .45, 'brand': .20, 'description': .35}
        self.mats, self.attr_audit = build_item_matrices(
            p['metadata'], self.n_items,
            min_df=int(ac.get('tfidf_min_df', 2)),
            max_df=float(ac.get('tfidf_max_df', .8)),
            description_len=int(ac.get('description_len', 128)),
            weights=self.attr_weights,
        )
        pseudo_profiles = build_profiles(self.mats, self.pseudo, self.n_items)
        model.eval()
        with torch.no_grad():
            fu, fi = model.forward(test=True)
            pi, _ = topk_from_embeddings(fu, fi, self.pusers, self.pseudo, 100, 1024)
        zt, _ = semantic_z_for_candidates(p['text_feature'], self.pseudo, self.pusers, pi, 256)
        zv, _ = semantic_z_for_candidates(p['visual_feature'], self.pseudo, self.pusers, pi, 128)
        za, _ = attribute_z(self.mats, pseudo_profiles, self.pusers, pi, 256, self.attr_weights)
        self.bg = fit_backgrounds(pi, zt, za, zv, self.n_items)

        H = histories_csr(self.histories, self.n_items, mean=False)
        self.text_sum = np.asarray(H @ self.text, dtype=np.float32)
        self.visual_sum = np.asarray(H @ self.visual, dtype=np.float32)
        self.attr_sum = {f: (H @ self.mats[f]).tocsr() for f in ('title', 'brand', 'description')}
        self.audit = {
            'seed': self.seed,
            'TRAIN_ONLY': True,
            'train_interactions': int(self.counts.sum()),
            'users_with_history': int((self.counts > 0).sum()),
            'K_HISTORY': c.K_HISTORY,
            'MIN_HISTORY': c.MIN_HISTORY,
            'MIN_COHERENT': c.MIN_COHERENT,
            'colift': self.params.to_dict(),
            'attribute_weights': self.attr_weights,
            'background_definition': 'TRAIN pseudo-history Top100 only',
            'current_positive_excluded_from_target': True,
            'VALIDATION_HISTORY_ACCESSED': False,
            'TEST_ACCESSED': False,
            'SPORTS_ACCESSED': False,
            'ELECTRONICS_ACCESSED': False,
        }

    def _hminus(self, users, pos):
        hs = []
        occ = np.zeros(len(users), np.int64)
        for r, (u, p) in enumerate(zip(users, pos)):
            h = self.histories[int(u)]
            q = [int(x) for x in h if int(x) != int(p)]
            hs.append(q)
            occ[r] = len(h) - len(q)
        return hs, occ

    def _locality_support(self, hm, anchors):
        B = len(hm)
        lengths = np.asarray([len(x) for x in hm], np.int64)
        offsets = np.zeros(B + 1, np.int64)
        np.cumsum(lengths, out=offsets[1:])
        sup = np.zeros((B, c.K_HISTORY), np.int64)
        mask = np.zeros((B, c.K_HISTORY), bool)
        loc = np.zeros((B, c.K_HISTORY), np.float32)
        if offsets[-1] == 0:
            return sup, mask, loc, lengths
        flat = np.concatenate([np.asarray(x, np.int64) for x in hm if x])
        rows = np.repeat(np.arange(B, dtype=np.int64), lengths)
        ar = anchors[rows]
        ct = np.einsum('ij,ij->i', self.text[flat], self.text[ar], optimize=True).astype(np.float32)
        cv = np.empty(len(flat), np.float32)
        for st in range(0, len(flat), 2048):
            en = min(st + 2048, len(flat))
            cv[st:en] = np.einsum('ij,ij->i', self.visual[flat[st:en]], self.visual[ar[st:en]], optimize=True)
        ca = np.zeros(len(flat), np.float32)
        for f, w in self.attr_weights.items():
            ca += float(w) * np.asarray(
                self.mats[f][flat].multiply(self.mats[f][ar]).sum(axis=1)
            ).reshape(-1).astype(np.float32)
        locality = (ct + cv + ca) / 3.0
        for r in range(B):
            st, en = offsets[r], offsets[r + 1]
            if st == en:
                continue
            ids = flat[st:en]
            val = locality[st:en]
            order = np.lexsort((ids, -val))[:c.K_HISTORY]
            n = len(order)
            sup[r, :n] = ids[order]
            mask[r, :n] = True
            loc[r, :n] = val[order]
        return sup, mask, loc, lengths

    def _profiles_and_lift(self, users, pos, occ, support, mask):
        B = len(users)
        cnt = (self.counts[users] - occ).astype(np.int64)
        tp = self.text_sum[users] - occ[:, None].astype(np.float32) * self.text[pos]
        vp = self.visual_sum[users] - occ[:, None].astype(np.float32) * self.visual[pos]
        tp = _norm_rows(tp)
        vp = _norm_rows(vp)
        raw_t = _dense_scores(self.text, tp, support)
        raw_v = _dense_scores(self.visual, vp, support)

        field_raw = {}
        urow = np.arange(B, dtype=np.int64)
        for f in ('title', 'brand', 'description'):
            sub = self.mats[f][pos].multiply(occ[:, None])
            prof = (self.attr_sum[f][users] - sub).tocsr()
            if f in ('title', 'description'):
                prof = normalize(prof, norm='l2', axis=1, copy=False)
            else:
                scale = np.divide(1.0, np.maximum(cnt, 1), dtype=np.float32)
                prof = prof.multiply(scale[:, None]).tocsr()
            field_raw[f] = _sparse_candidate_scores(prof, self.mats[f], urow, support, batch=256)

        fz = {f: _masked_z(v, mask) for f, v in field_raw.items()}
        raw_a = sum(float(self.attr_weights[f]) * fz[f] for f in fz)
        z_a = _masked_z(raw_a, mask)
        z_t = _masked_z(raw_t, mask)
        z_v = _masked_z(raw_v, mask)
        pt = self.params
        lt = _masked_z(z_t - float(pt.lambda_text) * self.bg['text']['shrunk_mean'][support], mask)
        la = _masked_z(z_a - float(pt.lambda_attribute) * self.bg['attribute']['shrunk_mean'][support], mask)
        lv = _masked_z(z_v - float(pt.lambda_visual) * self.bg['visual']['shrunk_mean'][support], mask)
        q = float(pt.alpha_text) * lt + float(pt.alpha_attribute) * la + float(pt.alpha_visual) * lv
        return q, {
            'z_text': z_t, 'z_attribute': z_a, 'z_visual': z_v,
            'lift_text': lt, 'lift_attribute': la, 'lift_visual': lv,
        }

    def targets(self, b, users_t, pos_t, target_ids):
        users = users_t.detach().cpu().numpy().astype(np.int64)
        pos = pos_t.detach().cpu().numpy().astype(np.int64)
        hm, occ = self._hminus(users, pos)
        packs = []
        for band, ids_t in enumerate(target_ids):
            anchors = ids_t.detach().cpu().numpy().astype(np.int64)
            support, local_mask, locality, hist_avail = self._locality_support(hm, anchors)
            q, mods = self._profiles_and_lift(users, pos, occ, support, local_mask)

            pref_mask = np.zeros_like(local_mask)
            gate_count = np.zeros(len(users), np.int64)
            for r in range(len(users)):
                ix = np.flatnonzero(local_mask[r])
                if len(ix) < c.MIN_HISTORY:
                    continue
                n_keep = max(3, int(math.ceil(len(ix) / 2.0)))
                ids = support[r, ix]
                order = np.lexsort((ids, -q[r, ix]))[:n_keep]
                keep = ix[order]
                pref_mask[r, keep] = True
                gate_count[r] = len(keep)

            dev = b['fi'].device
            sid = torch.as_tensor(support, device=dev, dtype=torch.long)
            pm = torch.as_tensor(pref_mask, device=dev)
            anchor = b['fi'][ids_t].detach()
            sh = b['fi'][sid].detach()
            v = sh - anchor[:, None, :]
            denom = (anchor * anchor).sum(1, keepdim=True).unsqueeze(2) + c.EPS
            tan = v - ((v * anchor[:, None, :]).sum(2, keepdim=True) / denom) * anchor[:, None, :]
            dn = tan.norm(dim=2, keepdim=True)
            dirs = tan / (dn + c.EPS)
            dirs = dirs * pm.unsqueeze(2)

            coherent_mask = np.zeros_like(pref_mask)
            coherence = np.zeros_like(locality, dtype=np.float32)
            medoid_slot = np.full(len(users), -1, np.int64)
            dirs_np = dirs.detach().cpu().numpy()
            for r in range(len(users)):
                ix = np.flatnonzero(pref_mask[r])
                if len(ix) < c.MIN_HISTORY:
                    continue
                dd = dirs_np[r, ix]
                sim = dd @ dd.T
                score = np.maximum(sim, 0).mean(axis=1)
                med_local = int(np.argmax(score))
                med = int(ix[med_local])
                medoid_slot[r] = med
                coh = np.maximum(sim[:, med_local], 0).astype(np.float32)
                keep = ix[coh > 0]
                coherent_mask[r, keep] = True
                coherence[r, ix] = coh

            zloc = _masked_z(locality, coherent_mask)
            semw = _masked_softmax(zloc, coherent_mask)
            weights = semw * coherence
            denw = weights.sum(1, keepdims=True)
            weights = np.divide(weights, np.maximum(denw, 1e-12), out=np.zeros_like(weights), where=denw > 0)
            wt = torch.as_tensor(weights, device=dev)
            cm = torch.as_tensor(coherent_mask, device=dev)
            vf = (dirs * wt.unsqueeze(2)).sum(1)
            rho = vf.norm(dim=1)
            dfield = vf / (rho.unsqueeze(1) + c.EPS)
            ncoh = cm.sum(1)
            eligible = (ncoh >= c.MIN_COHERENT) & (torch.as_tensor(hist_avail, device=dev) >= c.MIN_HISTORY) & (rho > c.EPS)
            support_factor = torch.clamp(ncoh.float() / 4.0, max=1.0)
            conf = (rho * support_factor).clamp(0, 1)
            conf = torch.where(eligible, conf, torch.zeros_like(conf))
            dfield = torch.where(eligible[:, None], dfield, torch.zeros_like(dfield))

            diag = {
                'band': int(band),
                'history_size': hist_avail,
                'candidate_local_support_count': local_mask.sum(1),
                'personalized_gate_support_count': gate_count,
                'coherent_support_count': coherent_mask.sum(1),
                'semantic_locality': locality[local_mask],
                'personalized_lift': q[local_mask],
                'coherence': coherence[coherent_mask],
                'rho': rho.detach().cpu().numpy(),
                'confidence': conf.detach().cpu().numpy(),
                'weights_sum': weights.sum(1),
                'invalid': (~eligible).detach().cpu().numpy(),
                'medoid_slot': medoid_slot,
                'current_positive_excluded': np.asarray([
                    int(p) not in set(support[r, local_mask[r]].tolist())
                    for r, p in enumerate(pos)
                ], bool),
            }
            packs.append({
                'direction': dfield.detach(),
                'confidence': conf.detach(),
                'support_ids': sid,
                'support_mask': cm,
                'support_weights': wt,
                'support_dirs': dirs.detach(),
                'local_mask': torch.as_tensor(local_mask, device=dev),
                'pref_mask': pm,
                'coherent_mask': cm,
                'diag': diag,
                'mods': mods,
            })
        return packs

def summarize_field(packs_batches):
    rows = [p['diag'] for batch in packs_batches for p in batch]
    if not rows:
        return {}
    cat = lambda k: np.concatenate([np.asarray(r[k]).reshape(-1) for r in rows])
    return {
        'history_size': c.stats(cat('history_size')),
        'candidate_local_support_count': c.stats(cat('candidate_local_support_count')),
        'personalized_gate_support_count': c.stats(cat('personalized_gate_support_count')),
        'coherent_support_count': c.stats(cat('coherent_support_count')),
        'invalid_fraction': float(cat('invalid').mean()),
        'rho': c.stats(cat('rho')),
        'confidence': c.stats(cat('confidence')),
        'semantic_locality': c.stats(cat('semantic_locality')),
        'personalized_lift': c.stats(cat('personalized_lift')),
        'coherence': c.stats(cat('coherence')) if len(cat('coherence')) else None,
        'weights_sum': c.stats(cat('weights_sum')),
        'current_positive_excluded_all': bool(cat('current_positive_excluded').all()),
    }
