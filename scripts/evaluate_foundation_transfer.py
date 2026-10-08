"""Adapter fitting and held-out prediction evaluation; not robot rollout validation."""
from __future__ import annotations
import argparse
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from hdml.models.foundation import HDMLFoundationModel
from hdml.data.multi_embodiment_dataset import FastEmbodimentBuffer


def evaluate_transfer(checkpoint_path: str, target_embodiment: str, target_dataset_path: str,
                      prop_dim: int, action_dim: int, epochs: int = 3,
                      steps_per_epoch: int = 100, batch_size: int = 128,
                      lr: float = 0.001, device: str = 'cuda', seed: int = 42,
                      allow_pretraining_overlap: bool = False,
                      output: str | None = None) -> dict:
    if epochs < 1 or steps_per_epoch < 1 or batch_size < 1:
        raise ValueError('epochs, steps_per_epoch and batch_size must be positive')
    if device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable; choose device=cpu explicitly if supported')
    dev = torch.device(device)
    torch.manual_seed(seed)
    np.random.seed(seed)
    ckpt = torch.load(checkpoint_path, map_location=dev, weights_only=False)
    cfg = ckpt['config']
    model_cfg = dict(cfg['model'])
    model_cfg['device'] = dev
    context_length = cfg.get('training', {}).get('context_length', 30)
    gamma = cfg.get('training', {}).get('gamma', 0.99)
    train = FastEmbodimentBuffer(target_embodiment, target_dataset_path, context_length,
                                 gamma=gamma, split='train', split_seed=seed)
    val = FastEmbodimentBuffer(target_embodiment, target_dataset_path, context_length,
                               gamma=gamma, split='validation', split_seed=seed)
    if (train.prop_dim, train.action_dim) != (prop_dim, action_dim):
        raise ValueError('Declared robot dimensions do not match dataset')
    seen = target_embodiment in cfg.get('embodiments', {})
    manifest = ckpt.get('data_manifest')
    known_hashes = set()
    if manifest:
        for value in manifest.values():
            known_hashes.update(value['train_episode_hashes'])
            known_hashes.update(value['validation_episode_hashes'])
    overlapping = bool(known_hashes.intersection(train.episode_hashes + val.episode_hashes))
    clean_provenance = manifest is not None and not seen and not overlapping
    if not clean_provenance and not allow_pretraining_overlap:
        raise ValueError('Unseen transfer is not established: target is seen, overlaps, or checkpoint has no data manifest. '
                         'For a diagnostic only, explicitly allow pretraining overlap; it will not be labeled unseen transfer.')
    model = HDMLFoundationModel(**model_cfg)
    shared = {k:v for k,v in ckpt['model_state_dict'].items() if not k.startswith('adapters.')}
    model.load_state_dict(shared, strict=True)
    adapter = model.register_embodiment(target_embodiment, prop_dim, action_dim)
    model.freeze_backbone()
    # New embodiment ID must be unused; seen diagnostic uses its original ID when known.
    id_map = ckpt.get('embodiment_indices', {})
    if target_embodiment in id_map:
        embodiment_idx = id_map[target_embodiment]
    else:
        embodiment_idx = max(id_map.values(), default=len(cfg.get('embodiments', {}))-1) + 1
    if embodiment_idx >= model.embodiment_embedding.num_embeddings:
        raise ValueError('No unused embodiment embedding slot')
    train.embodiment_idx = val.embodiment_idx = embodiment_idx
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=lr, weight_decay=1e-4)

    def predict(batch):
        return model(batch['states'], batch['rtgs'], batch['actions_in'], batch['timesteps'],
                     target_embodiment, batch['embodiment_idx'])[0]

    def validation_metrics():
        model.eval()
        generator = torch.Generator().manual_seed(seed + 10000)
        losses, jerks = [], []
        with torch.inference_mode():
            for _ in range(20):
                batch = val.sample_batch(batch_size, dev, generator=generator)
                pred = predict(batch)
                losses.append(F.smooth_l1_loss(pred, batch['actions_target']).item())
                if pred.shape[1] >= 3:
                    jerks.append((pred[:,2:] - 2*pred[:,1:-1] + pred[:,:-2]).abs().mean().item())
        return {'action_loss': float(np.mean(losses)),
                'action_second_difference': float(np.mean(jerks)) if jerks else None,
                'sampled_batches': 20, 'sampling_seed': seed + 10000}

    before = validation_metrics()
    losses = []
    if dev.type == 'cuda': torch.cuda.synchronize(dev)
    started = time.perf_counter()
    for _ in range(epochs):
        model.eval()
        adapter.train()
        for _ in range(steps_per_epoch):
            batch = train.sample_batch(batch_size, dev)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(dev.type, enabled=dev.type == 'cuda', dtype=torch.bfloat16):
                loss = F.smooth_l1_loss(predict(batch), batch['actions_target'])
            if not torch.isfinite(loss): raise FloatingPointError('Nonfinite adapter loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            losses.append(loss.item())
    if dev.type == 'cuda': torch.cuda.synchronize(dev)
    adaptation_seconds = time.perf_counter() - started
    after = validation_metrics()
    result = {'protocol_version': 2, 'target': target_embodiment, 'seed': seed,
              'evaluation_kind': 'held_out_prediction_after_adapter_fit',
              'pretraining_exposure': 'not_seen_in_recorded_manifest' if clean_provenance else 'seen_or_unverified',
              'closed_loop_control_evaluated': False,
              'train_episodes': train.num_episodes, 'validation_episodes': val.num_episodes,
              'train_episode_hashes': train.episode_hashes, 'validation_episode_hashes': val.episode_hashes,
              'validation_before': before, 'validation_after': after,
              'mean_training_loss': float(np.mean(losses)),
              'adaptation_seconds': adaptation_seconds, 'gradient_steps': epochs * steps_per_epoch,
              'batch_size': batch_size, 'context_length': context_length,
              'trainable_params': sum(p.numel() for p in model.parameters() if p.requires_grad),
              'frozen_params': sum(p.numel() for p in model.parameters() if not p.requires_grad),
              'checkpoint_sha256': hashlib.sha256(Path(checkpoint_path).read_bytes()).hexdigest(),
              'dataset_sha256': hashlib.sha256(Path(target_dataset_path).read_bytes()).hexdigest()}
    if output:
        destination = Path(output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--target-embodiment', required=True)
    parser.add_argument('--target-dataset', required=True)
    parser.add_argument('--prop-dim', type=int, required=True)
    parser.add_argument('--action-dim', type=int, required=True)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--allow-pretraining-overlap', action='store_true')
    parser.add_argument('--output', default='results/corrected/adapter_evaluation.json')
    args = parser.parse_args()
    evaluate_transfer(args.checkpoint, args.target_embodiment, args.target_dataset, args.prop_dim,
                      args.action_dim, epochs=args.epochs, seed=args.seed, device=args.device,
                      allow_pretraining_overlap=args.allow_pretraining_overlap, output=args.output)
