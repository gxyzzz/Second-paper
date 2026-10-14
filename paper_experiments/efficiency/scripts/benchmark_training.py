from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT))

import modules.preference_diffusion as pd
import pipelines.preference_pipeline as pp
from pipelines.dataset_config import load_dataset_config
from paper_experiments.pdbr_ablation.scripts.run_pdbr_ablation import _prepare_shared_init

WARMUP = 20
TIMED = 100
SEED = 1000


class LoaderLoop:
    def __init__(self, loader):
        self.loader = loader
        self.it = iter(loader)

    def next(self):
        try:
            return next(self.it)
        except StopIteration:
            self.it = iter(self.loader)
            return next(self.it)


def sync(device):
    torch.cuda.synchronize(device)


def _reset_peak(device):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)


def _result(dt, total_samples, device, config, **extra):
    return {
        'ms_per_batch': dt / TIMED * 1000.0,
        'samples_per_second': total_samples / dt,
        'peak_cuda_allocated_GB': torch.cuda.max_memory_allocated(device) / 1024**3,
        'peak_cuda_reserved_GB': torch.cuda.max_memory_reserved(device) / 1024**3,
        'timed_batches': TIMED,
        'warmup_batches': WARMUP,
        'batch_size_config': int(config['train_batch_size']),
        **extra,
    }


def run_baseline(dataset, gpu_id, run_dir):
    _prepare_shared_init(dataset, gpu_id, run_dir)
    model, config, train_data, _, _ = pp.instantiate_shared(dataset, gpu_id, run_dir)
    optimizer, _, _ = pp._optimizer(model, config)
    device = model.device
    loop = LoaderLoop(train_data)

    def step():
        interaction = loop.next()
        model.train(); optimizer.zero_grad(set_to_none=True)
        bundle = pd.forward_bundle(model)
        loss = pd.msca_loss_from_bundle(model, interaction, bundle)
        if not torch.isfinite(loss):
            raise RuntimeError('nonfinite baseline benchmark loss')
        loss.backward()
        if config['clip_grad_norm']:
            torch.nn.utils.clip_grad_norm_(model.parameters(), **config['clip_grad_norm'])
        optimizer.step()
        return int(interaction[0].shape[0])

    for _ in range(WARMUP):
        step()
    sync(device); _reset_peak(device); sync(device)
    total_samples = 0; start = time.perf_counter()
    for _ in range(TIMED):
        total_samples += step()
    sync(device); elapsed = time.perf_counter() - start
    result = _result(elapsed, total_samples, device, config, benchmark_mode='normal MSCA training')
    del model, optimizer
    torch.cuda.empty_cache()
    return result


def run_pdbr(dataset, gpu_id, run_dir):
    _prepare_shared_init(dataset, gpu_id, run_dir)
    model, config, train_data, _, _ = pp.instantiate_shared(dataset, gpu_id, run_dir)
    optimizer, _, _ = pp._optimizer(model, config)
    dcfg = load_dataset_config(dataset)['diffusion']
    network, diffusion_optimizer, schedule = pd.make_diffusion(
        SEED, model.device, lr=float(dcfg['optimizer']['lr'])
    )
    events = pp.TrainEvents(dataset, model)
    mask = pd.valid_mask(train_data, model.n_items, model.device)
    device = model.device
    loop = LoaderLoop(train_data)
    local_t = tuple(int(v) for v in dcfg['local_t'])
    lambda_hn = float(dcfg['lambda_hn'])
    half_margin = float(dcfg['half_margin'])
    theta = float(dcfg['theta_max_deg'])
    active_steps = int(dcfg['active_steps'])
    active = 3
    epoch = 30
    step_index = 0
    latent = pd.latent_stats(model)

    def step():
        nonlocal step_index, latent
        interaction = loop.next()
        model.train(); optimizer.zero_grad(set_to_none=True)
        bundle = pd.forward_bundle(model)
        base_loss = pd.msca_loss_from_bundle(model, interaction, bundle)
        ids, _, _ = pd.mine_current_ids(
            bundle, interaction, events.histories, mask,
            SEED, epoch, step_index, active,
        )
        for inner in range(active_steps):
            pd.direction_inner_step(
                network, diffusion_optimizer, schedule, bundle, interaction, latent, ids,
                202861000 + SEED * 100000 + epoch * 10000 + step_index * 10 + inner,
                local_t=local_t,
            )
        directions, _, _ = pd.predict_directions(
            network, schedule, bundle, interaction, latent, ids,
            202862000 + SEED * 100000 + epoch * 1000 + step_index,
            local_t=local_t,
        )
        deltas, diagnostics = pp._calibrate_bands(
            bundle, interaction, ids, directions, half_margin, theta
        )
        for diag in diagnostics:
            if not bool(diag['target_never_exceeds_positive'].all()):
                raise RuntimeError('PDBR benchmark target safety failure')
        refined = [bundle['fi'][ids[j]] + deltas[j] for j in range(active)]
        auxiliary = pd.aux_bpr(
            bundle['fu'][interaction[0]], bundle['fi'][interaction[1]], refined
        )
        total = base_loss + lambda_hn * auxiliary
        if not torch.isfinite(total):
            raise RuntimeError('nonfinite PDBR benchmark loss')
        total.backward()
        if config['clip_grad_norm']:
            torch.nn.utils.clip_grad_norm_(model.parameters(), **config['clip_grad_norm'])
        optimizer.step()
        step_index += 1
        return int(interaction[0].shape[0])

    for _ in range(WARMUP):
        step()
    # Match formal semantics: latent normalization is an epoch-level quantity.
    latent = pd.latent_stats(model)
    sync(device); _reset_peak(device); sync(device)
    total_samples = 0; start = time.perf_counter()
    for _ in range(TIMED):
        total_samples += step()
    sync(device); elapsed = time.perf_counter() - start
    result = _result(
        elapsed, total_samples, device, config,
        benchmark_mode='full-active upper-bound training benchmark',
        active_boundary_bands=3,
        active_steps=active_steps,
    )
    del model, optimizer, network, diffusion_optimizer
    torch.cuda.empty_cache()
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', choices=['baby', 'sports', 'elec', 'all'], default='all')
    ap.add_argument('--gpu-id', type=int, default=0)
    args = ap.parse_args()
    datasets = ['baby', 'sports', 'elec'] if args.dataset == 'all' else [args.dataset]
    rows = []
    overhead = []
    for dataset in datasets:
        print('[Training benchmark]', dataset, flush=True)
        base = run_baseline(dataset, args.gpu_id, ROOT / f'paper_experiments/efficiency/outputs/{dataset}/baseline')
        full = run_pdbr(dataset, args.gpu_id, ROOT / f'paper_experiments/efficiency/outputs/{dataset}/pdbr')
        rows.append({'dataset': dataset, 'method': 'MSCA', **base})
        rows.append({'dataset': dataset, 'method': 'MSCA+CoLiftRec', **base, 'benchmark_mode': 'same training as MSCA; CoLiftRec is post-hoc'})
        rows.append({'dataset': dataset, 'method': 'MSCA+CoLiftRec+PDBR', **full})
        overhead.append({
            'dataset': dataset,
            'time_overhead_percent': (full['ms_per_batch'] / base['ms_per_batch'] - 1.0) * 100.0,
            'memory_overhead_percent': (full['peak_cuda_allocated_GB'] / base['peak_cuda_allocated_GB'] - 1.0) * 100.0,
            'benchmark_scope': 'full-active upper-bound, not whole-training average',
        })

    ev = ROOT / 'paper_experiments/efficiency/evidence'; ev.mkdir(parents=True, exist_ok=True)
    fields = [
        'dataset', 'method', 'ms_per_batch', 'samples_per_second',
        'peak_cuda_allocated_GB', 'peak_cuda_reserved_GB', 'timed_batches',
        'warmup_batches', 'batch_size_config', 'benchmark_mode',
        'active_boundary_bands', 'active_steps',
    ]
    with (ev / 'TRAINING_BENCHMARK.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore', lineterminator=chr(10))
        writer.writeheader(); writer.writerows(rows)
    (ev / 'TRAINING_OVERHEAD.json').write_text(json.dumps(overhead, indent=2) + '\n', encoding='utf-8')
    print('wrote', ev / 'TRAINING_BENCHMARK.csv')


if __name__ == '__main__':
    main()
