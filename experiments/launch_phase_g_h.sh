#!/bin/bash
# Launch Tier 3 (image) and Tier 6 (graph) multi-seed campaigns in tmux.
# Each campaign runs in its own tmux session so the user can detach/reattach.
#
# Usage:
#   bash launch_phase_g_h.sh smoke     # quick smoke test (1 seed, 1 dataset each)
#   bash launch_phase_g_h.sh graph     # only the graph (Tier 6) campaign
#   bash launch_phase_g_h.sh image     # only the image (Tier 3) campaign
#   bash launch_phase_g_h.sh all       # both, graph first

set -e

mode="${1:-all}"
WORKSPACE="/workspace"
LOG_DIR="$WORKSPACE/experiments/results/multiseed_logs"
mkdir -p "$LOG_DIR"

cd "$WORKSPACE"

# ---- import smoke test --------------------------------------------------
echo "===== Import smoke test ====="
python3 -c "
import sys
sys.path.insert(0, '/workspace')
sys.path.insert(0, '/workspace/experiments')
from csann import CSANNConfig, CSANNModel, CSANNTrainer, CSANNOptimizerConfig
import torch
print('  CSANN imports OK')
print('  CUDA available:', torch.cuda.is_available())
print('  GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE')
try:
    import asann_cuda_ops
    print('  asann_cuda_ops imported from:', asann_cuda_ops.__file__)
except ImportError as e:
    print('  asann_cuda_ops NOT available:', e)
"

if [ "$mode" = "smoke" ]; then
    echo ""
    echo "===== Quick smoke run (CiteSeer 1 seed, 5 epochs) ====="
    cd /workspace
    TIER_SEED=42 RESULTS_SUFFIX=_smoke timeout 300 python3 experiments/tier_6/exp_6a_citeseer.py 2>&1 | tail -10
    exit 0
fi

# ---- launch tmux sessions -----------------------------------------------
launch_graph() {
    SESS="phase_h_graph"
    if tmux has-session -t "$SESS" 2>/dev/null; then
        echo "  tmux session $SESS already exists, killing"
        tmux kill-session -t "$SESS"
    fi
    tmux new-session -d -s "$SESS" "cd /workspace && python3 experiments/tier_6/run_tier6_graph_multiseed.py 2>&1 | tee $LOG_DIR/phase_h_graph.log"
    echo "  Launched: tmux session '$SESS' (logs at $LOG_DIR/phase_h_graph.log)"
}

launch_image() {
    SESS="phase_g_image"
    if tmux has-session -t "$SESS" 2>/dev/null; then
        echo "  tmux session $SESS already exists, killing"
        tmux kill-session -t "$SESS"
    fi
    tmux new-session -d -s "$SESS" "cd /workspace && python3 experiments/tier_3/run_tier3_multiseed.py 2>&1 | tee $LOG_DIR/phase_g_image.log"
    echo "  Launched: tmux session '$SESS' (logs at $LOG_DIR/phase_g_image.log)"
}

case "$mode" in
    graph)
        launch_graph
        ;;
    image)
        launch_image
        ;;
    all)
        launch_graph
        launch_image
        ;;
    *)
        echo "Unknown mode: $mode"
        exit 1
        ;;
esac

echo ""
echo "Active tmux sessions:"
tmux ls 2>/dev/null || echo "  (none)"
echo ""
echo "Attach with:  tmux attach -t phase_h_graph    or  tmux attach -t phase_g_image"
echo "Detach with:  Ctrl-b d"
echo "Tail logs:     tail -f $LOG_DIR/phase_h_graph.log"
echo "               tail -f $LOG_DIR/phase_g_image.log"
