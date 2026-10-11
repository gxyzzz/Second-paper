from __future__ import annotations
import gc
import hashlib
import math
import numpy as np
import torch
from sklearn.preprocessing import normalize

from diffusion_experiments.round30_candidate_side_diffusion_purification import round30_core as c
from diffusion_experiments.round30_candidate_side_diffusion_purification.round30_colift_reference import FrozenCoLiftReference
from modules.attribute import build_item_matrices, build_profiles, attribute_z
from modules.ranking import histories_csr, l2_normalize_rows, semantic_z_for_candidates, topk_from_embeddings
from modules.coliftrec import score_coliftrec
from pipelines.coliftrec import _params
from pipelines.dataset_config import load_dataset_config


def _norm_rows(x):
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


def reliable_z(ref, sup, floor=c.STD_FLOOR):
    ref = np.asarray(ref, np.float32)
    sup = np.asarray(sup, np.float32)
    mu = ref.mean(1, keepdims=True)
    sd = ref.std(1, keepdims=True)
    good = (sd[:, 0] >= floor) & np.isfinite(sd[:, 0])
    zr = np.zeros_like(ref, np.float32)
    zs = np.zeros_like(sup, np.float32)
    if good.any():
        zr[good] = ((ref[good] - mu[good]) / sd[good]).astype(np.float32)
        zs[good] = ((sup[good] - mu[good]) / sd[good]).astype(np.float32)
    return zr, zs, good, sd[:, 0].astype(np.float32)


def _sparse_scores(profile, item_matrix, items):
    B, K = items.shape
    rr = np.repeat(np.arange(B), K)
    cand = item_matrix[items.reshape(-1)]
    return np.asarray(profile[rr].multiply(cand).sum(axis=1)).reshape(B, K).astype(np.float32)


def _masked_softmax_z(values, mask):
    out = np.zeros_like(values, np.float32)
    for r in range(len(values)):
        ix = np.flatnonzero(mask[r])
        if not len(ix):
            continue
        v = values[r, ix]
        sd = float(v.std())
        z = (v - v.mean()) / (sd if sd >= c.STD_FLOOR else 1.0)
        e = np.exp(z - z.max())
        out[r, ix] = e / e.sum()
    return out


def _det_rank(user, seed, band, lo, hi):
    h = hashlib.sha256(f'{int(user)}|{int(seed)}|{int(band)}'.encode()).digest()
    return lo + (int.from_bytes(h[:8], 'little') % (hi - lo + 1))


class CandidateAnchorBank:
    """Four deterministic unlabeled candidate anchors/user from frozen B0 Full-CoLift TRAIN Top100."""
    def __init__(self, seed, model, events, reference=None, force=False, path_override=None):
        self.seed = int(seed)
        self.model = model
        self.events = events
        self.ref = reference or FrozenCoLiftReference(seed, model, events)
        self.path = path_override or c.ASSETS / f'ROUND30_CANDIDATE_ANCHOR_BANK_SEED{seed}.npz'
        if force or not self.path.exists():
            self._build()
        z = np.load(self.path)
        self.users = z['users'].astype(np.int64)
        self.anchors = z['anchors'].astype(np.int64)
        self.ranks = z['ranks'].astype(np.int64)
        self.full_top100 = z['full_top100'].astype(np.int64)
        self.event_users = np.repeat(self.users, 4)
        self.event_items = self.anchors.reshape(-1)
        self.event_ranks = self.ranks.reshape(-1)

    def _build(self):
        cfg = load_dataset_config('baby')
        paths = cfg['resolved_paths']
        ccfg = cfg['coliftrec']
        params = _params(ccfg)
        enabled = {m: bool(ccfg[m]['enabled']) for m in ('text', 'attribute', 'visual')}
        users = self.ref.users
        histories = self.events.histories
        with torch.no_grad():
            b = c.forward_bundle(self.model)
            ci, base_scores = topk_from_embeddings(b['fu'], b['fi'], users, histories, 100, 1024)
        zt, _ = semantic_z_for_candidates(paths['text_feature'], histories, users, ci, 256)
        zv, _ = semantic_z_for_candidates(paths['visual_feature'], histories, users, ci, 128)
        ac = ccfg['attribute']
        mats, _ = build_item_matrices(
            paths['metadata'], self.model.n_items,
            min_df=int(ac.get('tfidf_min_df', 2)), max_df=float(ac.get('tfidf_max_df', .8)),
            description_len=int(ac.get('description_len', 128)), weights=ac.get('weights'))
        profiles = build_profiles(mats, histories, self.model.n_items)
        za, _ = attribute_z(mats, profiles, users, ci, 256, ac.get('weights'))
        full_scores, _ = score_coliftrec(base_scores, ci, zt, za, zv, self.ref.bg, params, enabled)
        order = np.argsort(-full_scores, axis=1, kind='stable')
        ranked = np.take_along_axis(ci, order, axis=1)
        anchors = np.empty((len(users), 4), np.int32)
        ranks = np.empty((len(users), 4), np.int16)
        for r, u in enumerate(users):
            for j, (lo, hi) in enumerate(c.ANCHOR_BANDS):
                rk = _det_rank(u, self.seed, j, lo, hi)
                anchors[r, j] = ranked[r, rk - 1]
                ranks[r, j] = rk
        np.savez_compressed(self.path, users=users.astype(np.int32), anchors=anchors,
                            ranks=ranks, full_top100=ranked.astype(np.int32))
        del b, mats, profiles
        torch.cuda.empty_cache(); gc.collect()

    def bank_hash(self):
        h = hashlib.sha256()
        for x in (self.users, self.anchors, self.ranks, self.full_top100):
            h.update(np.ascontiguousarray(x).tobytes())
        return h.hexdigest()


class FieldComputer:
    def __init__(self, seed, model, events, reference=None):
        self.seed = int(seed)
        self.model = model
        self.events = events
        self.ref = reference or FrozenCoLiftReference(seed, model, events)
        self.cfg = load_dataset_config('baby')
        self.paths = self.cfg['resolved_paths']
        self.ccfg = self.cfg['coliftrec']
        self.params = _params(self.ccfg)
        hist = events.histories
        n_items = model.n_items
        self.text = l2_normalize_rows(np.asarray(np.load(self.paths['text_feature'], mmap_mode='r'), dtype=np.float32))
        self.visual = l2_normalize_rows(np.asarray(np.load(self.paths['visual_feature'], mmap_mode='r'), dtype=np.float32))
        ac = self.ccfg['attribute']
        self.weights = ac.get('weights') or {'title': .45, 'brand': .20, 'description': .35}
        self.mats, _ = build_item_matrices(
            self.paths['metadata'], n_items, min_df=int(ac.get('tfidf_min_df', 2)),
            max_df=float(ac.get('tfidf_max_df', .8)), description_len=int(ac.get('description_len', 128)),
            weights=self.weights)
        self.counts = np.asarray([len(h) for h in hist], np.int64)
        H = histories_csr(hist, n_items, mean=False)
        self.tsum = np.asarray(H @ self.text, dtype=np.float32)
        self.vsum = np.asarray(H @ self.visual, dtype=np.float32)
        self.asum = {f: (H @ self.mats[f]).tocsr() for f in ('title', 'brand', 'description')}

    def compute_pairs(self, users, candidates, batch=256):
        users = np.asarray(users, np.int64)
        candidates = np.asarray(candidates, np.int64)
        N = len(users)
        support = np.zeros((N, c.K_HISTORY), np.int32)
        locality = np.zeros((N, c.K_HISTORY), np.float32)
        local_mask = np.zeros((N, c.K_HISTORY), bool)
        qout = np.zeros((N, c.K_HISTORY), np.float32)
        pref = np.zeros((N, c.K_HISTORY), bool)
        hsize = np.zeros(N, np.int16)
        semrel = np.zeros((N, 3), bool)
        liftrel = np.zeros((N, 3), bool)
        semstd = np.zeros((N, 3), np.float32)
        liftstd = np.zeros((N, 3), np.float32)
        histories = self.events.histories
        refids_all = self.ref.candidates_for_users(users)
        for st in range(0, N, batch):
            en = min(st + batch, N)
            us = users[st:en]
            anchor_ids = candidates[st:en]
            B = len(us)
            hs = [list(map(int, histories[int(u)])) for u in us]
            hsize[st:en] = [len(x) for x in hs]
            lengths = np.asarray([len(x) for x in hs], np.int64)
            offsets = np.zeros(B + 1, np.int64)
            np.cumsum(lengths, out=offsets[1:])
            if offsets[-1]:
                flat = np.concatenate([np.asarray(x, np.int64) for x in hs if x])
                rows = np.repeat(np.arange(B, dtype=np.int64), lengths)
                aa = anchor_ids[rows]
                ct = np.einsum('ij,ij->i', self.text[flat], self.text[aa], optimize=True).astype(np.float32)
                cv = np.einsum('ij,ij->i', self.visual[flat], self.visual[aa], optimize=True).astype(np.float32)
                ca = np.zeros(len(flat), np.float32)
                for f, w in self.weights.items():
                    ca += float(w) * np.asarray(
                        self.mats[f][flat].multiply(self.mats[f][aa]).sum(axis=1)
                    ).reshape(-1).astype(np.float32)
                loc = (ct + cv + ca) / 3.0
                for r in range(B):
                    a, bb = offsets[r], offsets[r + 1]
                    if a == bb:
                        continue
                    ids = flat[a:bb]
                    vv = loc[a:bb]
                    order = np.lexsort((ids, -vv))[:c.K_HISTORY]
                    nn = len(order)
                    support[st + r, :nn] = ids[order]
                    locality[st + r, :nn] = vv[order]
                    local_mask[st + r, :nn] = True
            sup = support[st:en].astype(np.int64)
            mask = local_mask[st:en]
            refs = refids_all[st:en].astype(np.int64)
            cnt = self.counts[us]
            tp = _norm_rows(self.tsum[us])
            vp = _norm_rows(self.vsum[us])
            rt = np.einsum('bkd,bd->bk', self.text[refs], tp, optimize=True).astype(np.float32)
            st_sup = np.einsum('bkd,bd->bk', self.text[sup], tp, optimize=True).astype(np.float32)
            rv = np.einsum('bkd,bd->bk', self.visual[refs], vp, optimize=True).astype(np.float32)
            sv_sup = np.einsum('bkd,bd->bk', self.visual[sup], vp, optimize=True).astype(np.float32)
            ztr, zts, grt, sdt = reliable_z(rt, st_sup)
            zvr, zvs, grv, sdv = reliable_z(rv, sv_sup)
            fref, fsup = {}, {}
            for f in ('title', 'brand', 'description'):
                prof = self.asum[f][us].tocsr()
                if f in ('title', 'description'):
                    prof = normalize(prof, norm='l2', axis=1, copy=False)
                else:
                    prof = prof.multiply((1.0 / np.maximum(cnt, 1)).astype(np.float32)[:, None]).tocsr()
                rr = _sparse_scores(prof, self.mats[f], refs)
                ss = _sparse_scores(prof, self.mats[f], sup)
                zr, zs, _, _ = reliable_z(rr, ss)
                fref[f], fsup[f] = zr, zs
            attr_ref = sum(float(self.weights[f]) * fref[f] for f in fref)
            attr_sup = sum(float(self.weights[f]) * fsup[f] for f in fsup)
            zar, zas, gra, sda = reliable_z(attr_ref, attr_sup)
            pt = self.params
            lm = {'text': float(pt.lambda_text), 'attribute': float(pt.lambda_attribute), 'visual': float(pt.lambda_visual)}
            al = {'text': float(pt.alpha_text), 'attribute': float(pt.alpha_attribute), 'visual': float(pt.alpha_visual)}
            q = np.zeros((B, c.K_HISTORY), np.float32)
            modalities = (
                ('text', ztr, zts, grt, sdt),
                ('attribute', zar, zas, gra, sda),
                ('visual', zvr, zvs, grv, sdv),
            )
            for mi, (name, zr, zs, gsem, sdsem) in enumerate(modalities):
                rawr = zr - lm[name] * self.ref.bg[name]['shrunk_mean'][refs]
                raws = zs - lm[name] * self.ref.bg[name]['shrunk_mean'][sup]
                _, Ls, glift, sdlift = reliable_z(rawr, raws)
                q += al[name] * Ls
                semrel[st:en, mi] = gsem
                liftrel[st:en, mi] = glift
                semstd[st:en, mi] = sdsem
                liftstd[st:en, mi] = sdlift
            q[~mask] = 0
            qout[st:en] = q
            for r in range(B):
                ix = np.flatnonzero(mask[r])
                if len(ix) < c.MIN_HISTORY:
                    continue
                nkeep = max(3, int(math.ceil(len(ix) / 2)))
                order = np.lexsort((sup[r, ix], -q[r, ix]))[:nkeep]
                pref[st + r, ix[order]] = True
        return {
            'users': users, 'candidates': candidates,
            'support_ids': support, 'locality': locality, 'local_mask': local_mask,
            'q': qout, 'pref_mask': pref, 'history_size': hsize,
            'semantic_reliable': semrel, 'lift_reliable': liftrel,
            'semantic_std': semstd, 'lift_std': liftstd,
        }


class CandidateFieldBank:
    def __init__(self, seed, model, events, reference=None, anchors=None, force=False, path_override=None):
        self.seed = int(seed)
        self.model = model
        self.events = events
        self.ref = reference or FrozenCoLiftReference(seed, model, events)
        self.anchors = anchors or CandidateAnchorBank(seed, model, events, self.ref)
        self.path = path_override or c.ASSETS / f'ROUND30_CANDIDATE_FIELD_BANK_SEED{seed}.npz'
        if force or not self.path.exists():
            comp = FieldComputer(seed, model, events, self.ref)
            z = comp.compute_pairs(self.anchors.event_users, self.anchors.event_items)
            np.savez_compressed(self.path, **z)
            del comp
        z = np.load(self.path)
        self.users = z['users'].astype(np.int64)
        self.candidates = z['candidates'].astype(np.int64)
        self.support = z['support_ids'].astype(np.int64)
        self.locality = z['locality'].astype(np.float32)
        self.local_mask = z['local_mask'].astype(bool)
        self.q = z['q'].astype(np.float32)
        self.pref_mask = z['pref_mask'].astype(bool)
        self.history_size = z['history_size'].astype(np.int64)
        self.semantic_reliable = z['semantic_reliable'].astype(bool)
        self.lift_reliable = z['lift_reliable'].astype(bool)
        self.semantic_std = z['semantic_std'].astype(np.float32)
        self.lift_std = z['lift_std'].astype(np.float32)

    def bank_hash(self):
        h = hashlib.sha256()
        for x in (self.users, self.candidates, self.support, self.locality, self.local_mask,
                  self.q, self.pref_mask, self.history_size, self.semantic_reliable, self.lift_reliable):
            h.update(np.ascontiguousarray(x).tobytes())
        return h.hexdigest()

    def batch_static(self, idx, device):
        e = np.asarray(idx, np.int64)
        return {
            'support_ids': torch.as_tensor(self.support[e], device=device, dtype=torch.long),
            'locality': self.locality[e],
            'local_mask': torch.as_tensor(self.local_mask[e], device=device),
            'q': self.q[e],
            'pref_mask': torch.as_tensor(self.pref_mask[e], device=device),
            'history_size': self.history_size[e],
        }


def dynamic_field(b, candidate_t, static):
    sid = static['support_ids']
    pm = static['pref_mask']
    anchor = b['fi'][candidate_t].detach()
    sh = b['fi'][sid].detach()
    v = sh - anchor[:, None, :]
    den = (anchor * anchor).sum(1, keepdim=True).unsqueeze(2) + c.EPS
    tan = v - ((v * anchor[:, None, :]).sum(2, keepdim=True) / den) * anchor[:, None, :]
    dirs = tan / (tan.norm(dim=2, keepdim=True) + c.EPS)
    dirs = dirs * pm.unsqueeze(2)
    pref = pm.detach().cpu().numpy()
    dirs_np = dirs.detach().cpu().numpy()
    coherent = np.zeros_like(pref)
    coherence = np.zeros_like(static['locality'], np.float32)
    for r in range(len(dirs_np)):
        ix = np.flatnonzero(pref[r])
        if len(ix) < c.MIN_HISTORY:
            continue
        dd = dirs_np[r, ix]
        sim = dd @ dd.T
        score = np.maximum(sim, 0).mean(1)
        med = int(np.argmax(score))
        coh = np.maximum(sim[:, med], 0).astype(np.float32)
        keep = ix[coh > 0]
        coherent[r, keep] = True
        coherence[r, ix] = coh
    sem = _masked_softmax_z(static['locality'], coherent)
    weights = sem * coherence
    dw = weights.sum(1, keepdims=True)
    weights = np.divide(weights, np.maximum(dw, 1e-12), out=np.zeros_like(weights), where=dw > 0)
    wt = torch.as_tensor(weights, device=anchor.device)
    cm = torch.as_tensor(coherent, device=anchor.device)
    vf = (dirs * wt.unsqueeze(2)).sum(1)
    rho = vf.norm(dim=1)
    dfield = vf / (rho[:, None] + c.EPS)
    ncoh = cm.sum(1)
    eligible = (
        (torch.as_tensor(static['history_size'], device=anchor.device) >= c.MIN_HISTORY)
        & (ncoh >= c.MIN_COHERENT) & (rho > c.EPS)
    )
    conf = (rho * torch.clamp(ncoh.float() / 4.0, max=1.0)).clamp(0, 1)
    conf = torch.where(eligible, conf, torch.zeros_like(conf))
    dfield = torch.where(eligible[:, None], dfield, torch.zeros_like(dfield))
    return {
        'direction': dfield.detach(), 'confidence': conf.detach(),
        'support_ids': sid, 'support_mask': cm, 'support_weights': wt,
        'support_dirs': dirs.detach(),
        'diag': {
            'history_size': static['history_size'],
            'local_support_count': static['local_mask'].detach().cpu().numpy().sum(1),
            'pref_support_count': pref.sum(1),
            'coherent_support_count': coherent.sum(1),
            'rho': rho.detach().cpu().numpy(),
            'confidence': conf.detach().cpu().numpy(),
            'weights_sum': weights.sum(1),
            'q': static['q'][static['local_mask'].detach().cpu().numpy()],
            'locality': static['locality'][static['local_mask'].detach().cpu().numpy()],
            'invalid': (~eligible).detach().cpu().numpy(),
        }
    }
