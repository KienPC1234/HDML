#!/usr/bin/env bash
# Long, optimized training + 3-seed benchmark + latency. Runs inside tmux.
set -u
cd /data/HDML_Model

CONFIG=configs/halfcheetah_v5_default.yaml
DATASET=data/halfcheetah_v5_expert.npz
OUT=checkpoints/rebuild_long/halfcheetah_v5
RES=results/rebuild_long_final
EPOCHS=${EPOCHS:-60}
BS=${BS:-512}

mkdir -p "$RES" "$OUT/baselines" results/train_long_20261009

echo "[pipeline] start $(date) epochs=$EPOCHS bs=$BS"
.venv/bin/python scripts/train_offline.py \
  --config "$CONFIG" --dataset "$DATASET" \
  --epochs "$EPOCHS" --batch-size "$BS" --seed 42 \
  --output-dir "$OUT" --no-tensorboard \
  > results/train_long_20261009/train_halfcheetah_v5_e${EPOCHS}.log 2>&1
echo "[pipeline] train exit=$?"

cp -n checkpoints/halfcheetah_v5/baselines/*_best.pt "$OUT/baselines/" 2>/dev/null || true
CKPT="$OUT/best_model.pt"
for SEED in 42 100 2024; do
  echo "[pipeline] benchmark seed=$SEED"
  .venv/bin/python scripts/benchmark_baselines.py \
    --config "$CONFIG" --checkpoint "$CKPT" \
    --seed "$SEED" --episodes 5 --device cuda \
    --output-dir "$RES/seed$SEED" > "$RES/seed$SEED.log" 2>&1
  echo "[pipeline] seed $SEED exit=$?"
done
.venv/bin/python scripts/measure_closed_loop.py \
  --config "$CONFIG" --checkpoint "$CKPT" --episodes 3 --device cuda \
  --output "$RES/latency.json" > "$RES/latency.log" 2>&1
echo "[pipeline] DONE $(date)"
