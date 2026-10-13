from __future__ import annotations

import gc
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from models.msca import MSCA
from modules.attribute import attribute_z, build_item_matrices, build_profiles
from modules.coliftrec import fit_backgrounds, score_coliftrec
from modules.preference_diffusion import (
    BAND_NAMES, aux_bpr, calibrate, capture_rng, curriculum_count,
    direction_inner_step, forward_bundle, latent_stats, make_diffusion,
    mine_current_ids, msca_loss_from_bundle, predict_directions, rank_offsets,
    restore_rng, state_hash, summarize, valid_mask, warmup_step,
)
from modules.ranking import (
    histories_csr, l2_normalize_rows, metric_arrays, metrics_at, rank_by_score,
    row_zscore, sha256_file, topk_from_embeddings,
)
from pipelines.coliftrec import _params
from pipelines.dataset_config import load_dataset_config
from pipelines.evaluate_coliftrec import build_test_eval
from pipelines.msca_assets import build_train_histories_and_validation
from utils.configurator import Config
from utils.dataloader import EvalDataLoader, TrainDataLoader
from utils.dataset import RecDataset
from utils.utils import init_seed

ROOT = Path(__file__).resolve().parents[2]
SEED = 1000
ALL = ("R10", "N10", "R20", "N20", "R50", "N50")
PRIMARY = ("R10", "N10", "R20", "N20")


def _write_json(path, payload):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=_json_default) + "\n", encoding="utf-8")


def _json_default(value):
    if isinstance(value, np.generic): return value.item()
    if isinstance(value, Path): return str(value)
    raise TypeError(type(value).__name__)


def delta_pack(new, base):
    relative = {key: float((new[key] - base[key]) / base[key]) for key in ALL}
    return {
        "U": float(np.mean([relative[key] for key in PRIMARY])),
        "relative_delta": relative,
        "absolute_delta": {key: float(new[key] - base[key]) for key in ALL},
        "primary_positive_count": int(sum(new[key] > base[key] for key in PRIMARY)),
        "overall_positive_count": int(sum(new[key] > base[key] for key in ALL)),
    }


def _make_config(dataset, gpu_id, run_dir):
    config = Config("MSCA", dataset, {
        "gpu_id": int(gpu_id), "use_gpu": True, "seed": [SEED],
        "data_path": str(ROOT / "data") + "/",
        "checkpoint_dir": str(Path(run_dir) / "native_checkpoints"),
        "recommend_topk": str(Path(run_dir) / "recommend_topk") + "/",
    })
    for key in config["hyper_parameters"]:
        value = config[key]
        if isinstance(value, (list, tuple)) and len(value) == 1:
            config[key] = value[0]
    config["seed"] = SEED
    config["gpu_id"] = int(gpu_id); config["use_gpu"] = True
    config["device"] = torch.device(f"cuda:{int(gpu_id)}")
    return config


def _make_data(config):
    dataset = RecDataset(config); train, valid, _ = dataset.split()
    train.inter_num = len(train.df); valid.inter_num = len(valid.df)
    train_data = TrainDataLoader(config, train, batch_size=config["train_batch_size"], shuffle=True)
    valid_data = EvalDataLoader(config, valid, additional_dataset=train, batch_size=config["eval_batch_size"])
    return dataset, train, valid, train_data, valid_data


def _shared_path(run_dir):
    return Path(run_dir) / "assets" / "shared_seed1000_init.pt"


def create_shared_init(dataset, gpu_id, run_dir):
    path = _shared_path(run_dir); path.parent.mkdir(parents=True, exist_ok=True)
    config = _make_config(dataset, gpu_id, run_dir)
    dataset_obj, train, _, train_data, _ = _make_data(config)
    init_seed(SEED); train_data.pretrain_setup(); model = MSCA(config, train_data).to(config["device"])
    state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
    payload = {
        "dataset": dataset, "seed": SEED, "state_dict": state,
        "state_hash": state_hash(state), "rng": capture_rng(),
        "all_items": np.asarray(train_data.all_items, dtype=np.int64),
        "n_users": int(model.n_users), "n_items": int(model.n_items),
        "train_interactions": int(len(train.df)),
    }
    torch.save(payload, path)
    del model, dataset_obj; torch.cuda.empty_cache(); gc.collect()
    return payload


def instantiate_shared(dataset, gpu_id, run_dir):
    path = _shared_path(run_dir)
    if not path.exists(): create_shared_init(dataset, gpu_id, run_dir)
    pack = torch.load(path, map_location="cpu", weights_only=False)
    config = _make_config(dataset, gpu_id, run_dir)
    dataset_obj, train, valid, train_data, valid_data = _make_data(config)
    model = MSCA(config, train_data).to(config["device"])
    model.load_state_dict(pack["state_dict"], strict=True)
    train_data.all_items = list(map(int, pack["all_items"].tolist()))
    restore_rng(pack["rng"])
    if state_hash(model.state_dict()) != pack["state_hash"]:
        raise RuntimeError("shared seed1000 initialization mismatch")
    return model, config, train_data, valid_data, pack


class TrainEvents:
    def __init__(self, dataset, model):
        cfg = load_dataset_config(dataset)
        histories, _, _, _, _ = build_train_histories_and_validation(cfg["resolved_paths"]["interaction"], model.n_users)
        self.histories = histories


class DomainEvaluator:
    def __init__(self, dataset, model, split="validation", max_users=None):
        self.dataset = dataset; self.device = model.device; self.split = split
        cfg = load_dataset_config(dataset); paths = cfg["resolved_paths"]
        self.cfg = cfg; self.paths = paths; self.ccfg = cfg["coliftrec"]
        self.params = _params(self.ccfg)
        self.enabled = {name: bool(self.ccfg[name]["enabled"]) for name in ("text", "attribute", "visual")}
        histories, pseudo, pseudo_users, valid_users, valid_sets = build_train_histories_and_validation(
            paths["interaction"], model.n_users
        )
        self.histories = histories; self.pseudo = pseudo
        self.pseudo_users = pseudo_users
        if split == "validation":
            self.users = valid_users; self.sets = valid_sets
        elif split == "test":
            test_histories, test_users, test_sets = build_test_eval(paths["interaction"], model.n_users)
            if test_histories != histories:
                raise RuntimeError("TRAIN history mismatch in Test evaluator")
            self.users = test_users; self.sets = test_sets
        else:
            raise ValueError(split)
        if max_users is not None:
            self.users = self.users[:int(max_users)]
            self.sets = {int(user): self.sets[int(user)] for user in self.users}
            pseudo_limit = min(len(self.pseudo_users), max(int(max_users) * 2, int(max_users)))
            self.pseudo_users = self.pseudo_users[:pseudo_limit]
        self.n_items = model.n_items
        self.text = l2_normalize_rows(np.asarray(np.load(paths["text_feature"], mmap_mode="r"), dtype=np.float32))
        self.visual = l2_normalize_rows(np.asarray(np.load(paths["visual_feature"], mmap_mode="r"), dtype=np.float32))
        full_hist = histories_csr(self.histories, self.n_items, users=self.users, mean=True)
        pseudo_hist = histories_csr(self.pseudo, self.n_items, users=self.pseudo_users, mean=True)
        self.text_full = l2_normalize_rows(np.asarray(full_hist @ self.text, dtype=np.float32))
        self.text_pseudo = l2_normalize_rows(np.asarray(pseudo_hist @ self.text, dtype=np.float32))
        self.visual_full = l2_normalize_rows(np.asarray(full_hist @ self.visual, dtype=np.float32))
        self.visual_pseudo = l2_normalize_rows(np.asarray(pseudo_hist @ self.visual, dtype=np.float32))
        attr = self.ccfg["attribute"]
        self.mats, _ = build_item_matrices(
            paths["metadata"], self.n_items,
            min_df=int(attr.get("tfidf_min_df", 2)), max_df=float(attr.get("tfidf_max_df", .8)),
            description_len=int(attr.get("description_len", 128)), weights=attr.get("weights"),
        )
        self.full_profiles = build_profiles(self.mats, self.histories, self.n_items)
        self.pseudo_profiles = build_profiles(self.mats, self.pseudo, self.n_items)
        self.attribute_weights = attr.get("weights")

    @staticmethod
    def _semantic(feature, profile, items, batch):
        outputs = []
        for start in range(0, len(items), batch):
            end = min(start + batch, len(items))
            outputs.append(np.einsum("bld,bd->bl", feature[items[start:end]], profile[start:end], optimize=True).astype(np.float32))
        return row_zscore(np.concatenate(outputs, axis=0))

    @torch.no_grad()
    def evaluate(self, model, return_rank=False):
        model.eval(); user_emb, item_emb = model.forward(test=True)
        items, scores = topk_from_embeddings(user_emb, item_emb, self.users, self.histories, 100, 1024)
        pseudo_items, _ = topk_from_embeddings(user_emb, item_emb, self.pseudo_users, self.pseudo, 100, 1024)
        backbone = metrics_at(items, self.users, self.sets)
        ztp = self._semantic(self.text, self.text_pseudo, pseudo_items, 128)
        zte = self._semantic(self.text, self.text_full, items, 128)
        zvp = self._semantic(self.visual, self.visual_pseudo, pseudo_items, 64)
        zve = self._semantic(self.visual, self.visual_full, items, 64)
        zap, _ = attribute_z(self.mats, self.pseudo_profiles, self.pseudo_users, pseudo_items, 256, self.attribute_weights)
        zae, _ = attribute_z(self.mats, self.full_profiles, self.users, items, 256, self.attribute_weights)
        backgrounds = fit_backgrounds(pseudo_items, ztp, zap, zvp, self.n_items)
        full_scores, _ = score_coliftrec(scores, items, zte, zae, zve, backgrounds, self.params, self.enabled)
        ranked = rank_by_score(items, full_scores)
        result = {
            "backbone": backbone,
            "colift": metrics_at(ranked, self.users, self.sets),
            "user_count": int(len(self.users)),
            "candidate_shape": [int(items.shape[0]), int(items.shape[1])],
            "diffusion_inference_calls": 0,
        }
        if return_rank:
            result["ranked_items"] = ranked
        return result


def _optimizer(model, config):
    weight_decay = float(config["weight_decay"]) if config["weight_decay"] is not None else 0.0
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config["learning_rate"]), weight_decay=weight_decay)
    factor = config["learning_rate_scheduler"]
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: factor[0] ** (epoch / factor[1]))
    return optimizer, scheduler, weight_decay


def _save_best(model, diffusion, run_dir, branch, epoch, score):
    path = Path(run_dir) / "checkpoints" / branch / "best.pt"; path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "dataset": None, "seed": SEED, "branch": branch, "epoch": int(epoch),
        "full_colift_R20": float(score),
        "model_state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "diffusion_state_dict": None if diffusion is None else {key: value.detach().cpu() for key, value in diffusion.state_dict().items()},
    }
    torch.save(payload, path)
    return path


def train_baseline(dataset, evaluator, gpu_id, run_dir, smoke=False, selection="colift"):
    model, config, train_data, _, pack = instantiate_shared(dataset, gpu_id, run_dir)
    optimizer, scheduler, weight_decay = _optimizer(model, config)
    epochs = [0] if smoke else list(range(int(load_dataset_config(dataset)["diffusion"]["formal_epochs"])))
    best_score = -float("inf"); best_epoch = None; best_eval = None; best_path = None
    plan_hashes = {}; lr_trajectory = []; trajectory = []
    for epoch in epochs:
        digest = hashlib.sha256(); losses = []; lr_trajectory.append(float(optimizer.param_groups[0]["lr"])); model.train()
        for batch_idx, interaction in enumerate(train_data):
            if smoke and batch_idx >= 2: break
            for tensor in interaction[:3]: digest.update(tensor.detach().cpu().numpy().astype(np.int64).tobytes())
            optimizer.zero_grad(set_to_none=True); bundle = forward_bundle(model)
            loss = msca_loss_from_bundle(model, interaction, bundle)
            if not torch.isfinite(loss): raise RuntimeError("nonfinite baseline loss")
            loss.backward()
            if config["clip_grad_norm"]: torch.nn.utils.clip_grad_norm_(model.parameters(), **config["clip_grad_norm"])
            optimizer.step(); losses.append(float(loss.detach()))
        scheduler.step(); plan_hashes[str(epoch)] = digest.hexdigest()
        evaluation = evaluator.evaluate(model)
        score = float(evaluation[selection]["R20"])
        trajectory.append({"epoch": epoch, "loss": float(np.mean(losses)), "evaluation": evaluation})
        if not smoke and score > best_score:
            best_score = score; best_epoch = epoch; best_eval = evaluation
            best_path = _save_best(model, None, run_dir, "baseline", epoch, score)
    result = {
        "dataset": dataset, "seed": SEED, "branch": "baseline", "selection": f"Validation {selection} R20",
        "initial_state_hash": pack["state_hash"], "normal_plan_hashes": plan_hashes,
        "lr_trajectory": lr_trajectory, "epochs_trained": len(epochs), "weight_decay": weight_decay,
        "best_epoch": best_epoch, "best_evaluation": best_eval,
        "best_checkpoint": None if best_path is None else str(best_path),
        "checkpoint_sha256": None if best_path is None else sha256_file(best_path),
        "trajectory": trajectory, "TEST_ACCESSED": False,
    }
    if not smoke: _write_json(Path(run_dir) / "evidence" / "BASELINE_TRAIN.json", result)
    del model, optimizer; torch.cuda.empty_cache(); gc.collect()
    return result


def _calibrate_bands(bundle, interaction, ids, directions, half_margin, theta):
    user = bundle["fu"][interaction[0]].detach(); positive = bundle["fi"][interaction[1]].detach()
    deltas, diagnostics = [], []
    for index, item_ids in enumerate(ids):
        delta, diagnostic = calibrate(user, positive, bundle["fi"][item_ids].detach(), directions[index],
                                      half_margin=half_margin, theta_max_deg=theta)
        diagnostic["band"] = BAND_NAMES[index]; deltas.append(delta); diagnostics.append(diagnostic)
    return deltas, diagnostics


def train_full(dataset, evaluator, gpu_id, run_dir, smoke=False):
    model, config, train_data, _, pack = instantiate_shared(dataset, gpu_id, run_dir)
    events = TrainEvents(dataset, model); optimizer, scheduler, weight_decay = _optimizer(model, config)
    diffusion_cfg = load_dataset_config(dataset)["diffusion"]
    network, diffusion_optimizer, diffusion_schedule = make_diffusion(SEED, model.device, lr=float(diffusion_cfg["optimizer"]["lr"]))
    mask = valid_mask(train_data, model.n_items, model.device)
    epochs = [0, 10] if smoke else list(range(int(diffusion_cfg["formal_epochs"])))
    best_score = -float("inf"); best_epoch = None; best_eval = None; best_path = None
    plan_hashes = {}; pair_hashes = {}; offset_hashes = {}; lr_trajectory = []; trajectory = []
    lambda_hn = float(diffusion_cfg["lambda_hn"]); half_margin = float(diffusion_cfg["half_margin"])
    theta = float(diffusion_cfg["theta_max_deg"]); active_steps = int(diffusion_cfg["active_steps"])
    local_t = tuple(int(value) for value in diffusion_cfg["local_t"])
    for epoch in epochs:
        active = curriculum_count(epoch); latent = latent_stats(model)
        normal_digest = hashlib.sha256(); pair_digest = hashlib.sha256(); offset_digest = hashlib.sha256()
        normal_losses = []; aux_losses = []; diffusion_losses = []; direction_batches = []; calibration_batches = []
        lr_trajectory.append(float(optimizer.param_groups[0]["lr"])); model.train()
        for batch_idx, interaction in enumerate(train_data):
            if smoke and batch_idx >= 2: break
            users = interaction[0].detach().cpu().numpy(); pos = interaction[1].detach().cpu().numpy(); neg = interaction[2].detach().cpu().numpy()
            normal_digest.update(users.tobytes()); normal_digest.update(pos.tobytes()); normal_digest.update(neg.tobytes())
            pair_digest.update(users.tobytes()); pair_digest.update(pos.tobytes())
            offsets = rank_offsets(SEED, epoch, batch_idx, len(interaction[0]), active)
            for values in offsets: offset_digest.update(np.asarray(values, dtype=np.int64).tobytes())
            optimizer.zero_grad(set_to_none=True); bundle = forward_bundle(model)
            base_loss = msca_loss_from_bundle(model, interaction, bundle); ids = []
            auxiliary = torch.zeros((), device=model.device)
            if active > 0:
                ids, used_offsets, _ = mine_current_ids(bundle, interaction, events.histories, mask, SEED, epoch, batch_idx, active)
                for expected, used in zip(offsets, used_offsets):
                    if not np.array_equal(expected, used): raise RuntimeError("boundary offset mismatch")
            if active == 0:
                record = warmup_step(network, diffusion_optimizer, diffusion_schedule, bundle, interaction, latent,
                                     202860000 + SEED * 100000 + epoch * 1000 + batch_idx, local_t=local_t)
                diffusion_losses.append(record)
            else:
                for inner in range(active_steps):
                    record = direction_inner_step(
                        network, diffusion_optimizer, diffusion_schedule, bundle, interaction, latent, ids,
                        202861000 + SEED * 100000 + epoch * 10000 + batch_idx * 10 + inner, local_t=local_t,
                    )
                    diffusion_losses.append(record)
                directions, direction_diag, _ = predict_directions(
                    network, diffusion_schedule, bundle, interaction, latent, ids,
                    202862000 + SEED * 100000 + epoch * 1000 + batch_idx, local_t=local_t,
                )
                direction_batches.append(direction_diag)
                deltas, calibration_diag = _calibrate_bands(bundle, interaction, ids, directions, half_margin, theta)
                calibration_batches.append(calibration_diag)
                refined = [bundle["fi"][ids[index]] + deltas[index] for index in range(active)]
                auxiliary = aux_bpr(bundle["fu"][interaction[0]], bundle["fi"][interaction[1]], refined)
            total = base_loss + (lambda_hn * auxiliary if active > 0 else 0.0)
            if not torch.isfinite(total): raise RuntimeError("nonfinite full loss")
            total.backward()
            if config["clip_grad_norm"]: torch.nn.utils.clip_grad_norm_(model.parameters(), **config["clip_grad_norm"])
            optimizer.step(); normal_losses.append(float(base_loss.detach())); aux_losses.append(float(auxiliary.detach()))
        scheduler.step(); plan_hashes[str(epoch)] = normal_digest.hexdigest(); pair_hashes[str(epoch)] = pair_digest.hexdigest(); offset_hashes[str(epoch)] = offset_digest.hexdigest()
        evaluation = evaluator.evaluate(model); score = float(evaluation["colift"]["R20"])
        record = {"epoch": epoch, "active_bands": active, "normal_loss": float(np.mean(normal_losses)),
                  "aux_loss": float(np.mean(aux_losses)), "evaluation": evaluation}
        if active > 0:
            mechanism = summarize(direction_batches, calibration_batches, diffusion_losses, theta_max_deg=theta)
            if not mechanism["angle_bound_pass"] or not mechanism["target_never_exceeds_positive"]:
                raise RuntimeError("bounded refinement audit failed")
            record["mechanism"] = mechanism
        trajectory.append(record)
        if not smoke and score > best_score:
            best_score = score; best_epoch = epoch; best_eval = evaluation
            best_path = _save_best(model, network, run_dir, "full", epoch, score)
    result = {
        "dataset": dataset, "seed": SEED, "branch": "full",
        "method": "Preference-Directed Diffusion Boundary Refinement",
        "config": {"lambda_hn": lambda_hn, "half_margin": half_margin, "theta_max_deg": theta,
                   "local_t": list(local_t), "active_steps": active_steps},
        "initial_state_hash": pack["state_hash"], "normal_plan_hashes": plan_hashes,
        "pair_hashes": pair_hashes, "rank_offset_hashes": offset_hashes, "lr_trajectory": lr_trajectory,
        "epochs_trained": len(epochs), "weight_decay": weight_decay,
        "best_epoch": best_epoch, "best_evaluation": best_eval,
        "best_checkpoint": None if best_path is None else str(best_path),
        "checkpoint_sha256": None if best_path is None else sha256_file(best_path),
        "trajectory": trajectory, "checkpoint_selection": "best Full CoLiftRec Validation R20",
        "RANKING_SCORE_GRAD_TO_DIFFUSION": 0, "TEST_ACCESSED": False,
    }
    if not smoke: _write_json(Path(run_dir) / "evidence" / "FULL_TRAIN.json", result)
    del model, optimizer, network, diffusion_optimizer; torch.cuda.empty_cache(); gc.collect()
    return result


def _load_selected(dataset, gpu_id, run_dir, branch):
    evidence = json.loads((Path(run_dir) / "evidence" / ("BASELINE_TRAIN.json" if branch == "baseline" else "FULL_TRAIN.json")).read_text())
    checkpoint_path = Path(evidence["best_checkpoint"])
    if sha256_file(checkpoint_path) != evidence["checkpoint_sha256"]: raise RuntimeError("checkpoint SHA mismatch")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model, _, _, _, _ = instantiate_shared(dataset, gpu_id, run_dir)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True); model.eval()
    return model, evidence


def _per_user_primary(ranked, users, sets):
    _, _, hit, positive_lengths = metric_arrays(ranked, users, sets, max_k=20)
    ranks = np.arange(1, 21, dtype=np.float64)[None, :]
    recall = np.cumsum(hit, axis=1) / positive_lengths[:, None]
    dcg = np.cumsum(hit / np.log2(ranks + 1.0), axis=1)
    idcg = np.cumsum(np.ones((len(users), 20), dtype=np.float64) / np.log2(ranks + 1.0), axis=1)
    for row, length in enumerate(positive_lengths):
        cutoff = min(int(length), 20)
        if cutoff < 20: idcg[row, cutoff:] = idcg[row, cutoff - 1]
    ndcg = dcg / idcg
    return np.stack([recall[:, 9], ndcg[:, 9], recall[:, 19], ndcg[:, 19]], axis=1)


def _bootstrap(base, new, seed, resamples=1000):
    def utility(a, b):
        am = a.mean(0); bm = b.mean(0)
        return float(np.mean((bm - am) / np.maximum(am, 1e-12)))
    rng = np.random.default_rng(int(seed)); values = np.empty(resamples, np.float64)
    for index in range(resamples):
        sample = rng.integers(0, len(base), len(base)); values[index] = utility(base[sample], new[sample])
    return {"point_U": utility(base, new), "ci95": [float(np.quantile(values, .025)), float(np.quantile(values, .975))],
            "positive_replicate_fraction": float((values > 0).mean()), "resamples": int(resamples)}


def evaluate_test(dataset, stage, gpu_id, run_dir, baseline_train, full_train=None):
    marker = Path(run_dir) / "test" / "TEST_RUN_COMPLETED"
    if marker.exists():
        raise RuntimeError(f"formal Test already executed for run: {run_dir}")
    baseline_model, baseline_evidence = _load_selected(dataset, gpu_id, run_dir, "baseline")
    evaluator = DomainEvaluator(dataset, baseline_model, split="test")
    baseline_eval = evaluator.evaluate(baseline_model, return_rank=True)
    baseline_rank = baseline_eval.pop("ranked_items")
    output = {
        "dataset": dataset, "seed": SEED, "stage": stage,
        "MSCA": baseline_eval["backbone"],
        "diffusion_inference_calls": 0, "TEST_USED_FOR_TUNING": False,
        "baseline_best_epoch": baseline_evidence["best_epoch"],
    }
    if stage in ("coliftrec", "full"):
        output["MSCA_FULL_COLIFTREC"] = baseline_eval["colift"]
        output["CoLiftRec_vs_MSCA"] = delta_pack(baseline_eval["colift"], baseline_eval["backbone"])
    if stage == "full":
        full_model, full_evidence = _load_selected(dataset, gpu_id, run_dir, "full")
        full_eval = evaluator.evaluate(full_model, return_rank=True)
        full_rank = full_eval.pop("ranked_items")
        output["MSCA_FULL_COLIFTREC_DIFFUSION"] = full_eval["colift"]
        output["Diffusion_vs_CoLiftRec"] = delta_pack(full_eval["colift"], baseline_eval["colift"])
        output["full_best_epoch"] = full_evidence["best_epoch"]
        base_users = _per_user_primary(baseline_rank, evaluator.users, evaluator.sets)
        full_users = _per_user_primary(full_rank, evaluator.users, evaluator.sets)
        output["bootstrap_Diffusion_vs_CoLiftRec"] = _bootstrap(base_users, full_users, 20263300 + {"baby": 0, "sports": 1, "elec": 2}[dataset])
        del full_model
    marker.parent.mkdir(parents=True, exist_ok=True); marker.write_text("TEST_RUN_COUNT=1\n", encoding="utf-8")
    _write_json(Path(run_dir) / "evidence" / "TEST_RESULTS.json", output)
    del baseline_model, evaluator; torch.cuda.empty_cache(); gc.collect()
    return output


def run_publication_pipeline(dataset, stage, run_dir, gpu_id=0, smoke=False):
    run_dir = Path(run_dir); run_dir.mkdir(parents=True, exist_ok=True)
    if not _shared_path(run_dir).exists(): create_shared_init(dataset, gpu_id, run_dir)
    probe, _, _, _, _ = instantiate_shared(dataset, gpu_id, run_dir)
    evaluator = DomainEvaluator(dataset, probe, split="validation", max_users=128 if smoke else None)
    del probe; torch.cuda.empty_cache(); gc.collect()
    selection = "backbone" if stage == "msca" else "colift"
    baseline = train_baseline(dataset, evaluator, gpu_id, run_dir, smoke=smoke, selection=selection)
    if smoke and stage == "msca":
        result = {"dataset": dataset, "stage": stage, "smoke": True, "baseline": baseline["trajectory"][-1]}
        _write_json(run_dir / "SMOKE_RESULT.json", result); return result
    full = None
    if stage == "full":
        full = train_full(dataset, evaluator, gpu_id, run_dir, smoke=smoke)
    if smoke:
        result = {"dataset": dataset, "stage": stage, "smoke": True,
                  "baseline": baseline["trajectory"][-1],
                  "full": None if full is None else full["trajectory"][-1],
                  "TEST_ACCESSED": False}
        _write_json(run_dir / "SMOKE_RESULT.json", result); return result
    validation = {
        "dataset": dataset, "seed": SEED, "stage": stage,
        "baseline": baseline["best_evaluation"], "baseline_best_epoch": baseline["best_epoch"],
        "checkpoint_selection": baseline["selection"], "TEST_ACCESSED": False,
    }
    if stage in ("coliftrec", "full"):
        validation["CoLiftRec_vs_MSCA"] = delta_pack(baseline["best_evaluation"]["colift"], baseline["best_evaluation"]["backbone"])
    if full is not None:
        fairness = {
            "initial_state_exact": baseline["initial_state_hash"] == full["initial_state_hash"],
            "normal_plan_exact": baseline["normal_plan_hashes"] == full["normal_plan_hashes"],
            "lr_trajectory_exact": baseline["lr_trajectory"] == full["lr_trajectory"],
        }
        fairness["PASS"] = bool(all(fairness.values()))
        if not fairness["PASS"]: raise RuntimeError(f"baseline/full fairness failure: {fairness}")
        validation["full"] = full["best_evaluation"]; validation["full_best_epoch"] = full["best_epoch"]
        validation["Diffusion_vs_CoLiftRec"] = delta_pack(full["best_evaluation"]["colift"], baseline["best_evaluation"]["colift"])
        validation["fairness"] = fairness
        _write_json(run_dir / "evidence" / "FAIRNESS.json", fairness)
    _write_json(run_dir / "evidence" / "VALIDATION_RESULTS.json", validation)
    test = evaluate_test(dataset, stage, gpu_id, run_dir, baseline, full)
    return {"validation": validation, "test": test}
