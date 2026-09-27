#!/bin/bash
# Full automated, guarded benchmark orchestration:
# 1. Run Swift-Qwen3.8-Flash-Next-V3 on port 8090
# 2. Stop server 8090, reclaim memory
# 3. Start Swift-Qwen3.8-27B-Splash-HQ on port 8000
# 4. Run Swift-Qwen3.8-27B-Splash-HQ on port 8000
# 5. Stop server 8000, reclaim memory
# 6. Restore Swift-Qwen3.8-Flash-Next-V3 on port 8090
# 7. Generate full comparative scorecard

set -euo pipefail

REPO="/Users/nitin/Documents/shared-with-google-drive/model-serving/slipstream-gguf"
PYTHON="$REPO/.venv/bin/python"
LOGDIR="$HOME/models/logs"
mkdir -p "$LOGDIR"

ORCH_LOG="$LOGDIR/orchestration-$(date +%Y%m%d-%H%M%S).log"
exec > >(tee -a "$ORCH_LOG") 2>&1

echo "=================================================================="
echo "Starting Full Comparative Benchmark Orchestrator at $(date)"
echo "Repo: $REPO"
echo "Log: $ORCH_LOG"
echo "=================================================================="

# Function to check memory
free_gib() {
    vm_stat | awk '/page size/{gsub("[^0-9]","",$8);ps=$8} /Pages (free|inactive|speculative|purgeable)/{gsub("[^0-9.]","",$NF);f+=$NF} END{printf "%.1f", f*ps/1073741824}'
}

# -----------------------------------------------------------------------------
# PHASE 1: Run Swift-Qwen3.8-Flash-Next-V3
# -----------------------------------------------------------------------------
echo ""
echo "=== PHASE 1: Evaluating Swift-Qwen3.8-Flash-Next-V3 on :8090 ==="
echo "Free memory before run: $(free_gib) GiB"

# Check port 8090 is running
if ! curl -s http://127.0.0.1:8090/v1/models >/dev/null 2>&1; then
    echo "Server on :8090 is not running. Starting it..."
    nohup "$HOME/models/bin/swift-flashnext-server.sh" > "$LOGDIR/swift-flashnext-server.log" 2>&1 &
    echo "Waiting for :8090 to become healthy..."
    for i in $(seq 1 45); do
        if curl -s http://127.0.0.1:8090/v1/models >/dev/null 2>&1; then
            echo ":8090 is ready!"
            break
        fi
        sleep 2
    done
fi

if ! curl -s http://127.0.0.1:8090/v1/models >/dev/null 2>&1; then
    echo "ERROR: :8090 failed to start!"
    exit 1
fi

echo "Running benchmark against Swift-Qwen3.8-Flash-Next-V3..."
"$PYTHON" "$REPO/dev/benchmarks/benchmark_swift27b_vs_swiftv3.py" --target v3

echo "Phase 1 complete! Result saved."

# -----------------------------------------------------------------------------
# PHASE 2: Graceful Swap: Stop 8090, Verify Memory
# -----------------------------------------------------------------------------
echo ""
echo "=== PHASE 2: Stopping :8090 & Clearing Memory ==="
"$HOME/models/bin/slipstream-stop.sh"

echo "Waiting for port 8090 release..."
for i in $(seq 1 20); do
    if ! lsof -iTCP:8090 -sTCP:LISTEN >/dev/null 2>&1; then
        echo "Port 8090 successfully closed."
        break
    fi
    sleep 1
done

# Kill any leftover splash process if still hanging
pkill -9 -f "splash serve-native" 2>/dev/null || true
pkill -9 -f "server.server" 2>/dev/null || true
sleep 3

echo "Free memory after stopping :8090: $(free_gib) GiB"

# -----------------------------------------------------------------------------
# PHASE 3: Start Swift-Qwen3.8-27B-Splash-HQ on :8000
# -----------------------------------------------------------------------------
echo ""
echo "=== PHASE 3: Launching Swift-Qwen3.8-27B-Splash-HQ on :8000 ==="

nohup splash-q8 serve --model nitinpanj/Swift-Qwen3.8-27B-Splash-HQ > "$LOGDIR/swift27b-server.log" 2>&1 &
B27_PID=$!
echo "Launched splash-q8 with PID $B27_PID"

echo "Waiting for port 8000 to become healthy..."
READY=0
for i in $(seq 1 45); do
    if curl -s http://127.0.0.1:8000/v1/models >/dev/null 2>&1; then
        echo ":8000 is ready and answering!"
        READY=1
        break
    fi
    sleep 2
done

if [ "$READY" -ne 1 ]; then
    echo "ERROR: :8000 did not start within 90 seconds. Server log:"
    tail -n 30 "$LOGDIR/swift27b-server.log"
    exit 1
fi

echo "Free memory with 27B active: $(free_gib) GiB"

# -----------------------------------------------------------------------------
# PHASE 4: Run Swift-Qwen3.8-27B-Splash-HQ
# -----------------------------------------------------------------------------
echo ""
echo "=== PHASE 4: Evaluating Swift-Qwen3.8-27B-Splash-HQ on :8000 ==="
"$PYTHON" "$REPO/dev/benchmarks/benchmark_swift27b_vs_swiftv3.py" --target 27b

echo "Phase 4 complete! Result saved."

# -----------------------------------------------------------------------------
# PHASE 5: Graceful Swap: Stop 8000, Verify Memory
# -----------------------------------------------------------------------------
echo ""
echo "=== PHASE 5: Stopping :8000 & Clearing Memory ==="
"$HOME/models/bin/swift27b-stop.sh"

echo "Waiting for port 8000 release..."
for i in $(seq 1 20); do
    if ! lsof -iTCP:8000 -sTCP:LISTEN >/dev/null 2>&1; then
        echo "Port 8000 successfully closed."
        break
    fi
    sleep 1
done

pkill -9 -f "Splash-Q8" 2>/dev/null || true
pkill -9 -f "splash-q8" 2>/dev/null || true
sleep 3

echo "Free memory after stopping :8000: $(free_gib) GiB"

# -----------------------------------------------------------------------------
# PHASE 6: Restore Swift-Qwen3.8-Flash-Next-V3 on :8090
# -----------------------------------------------------------------------------
echo ""
echo "=== PHASE 6: Restoring Swift-Qwen3.8-Flash-Next-V3 on :8090 ==="
nohup "$HOME/models/bin/swift-flashnext-server.sh" > "$LOGDIR/swift-flashnext-server.log" 2>&1 &
echo "Restoring server in background. Checking health..."
for i in $(seq 1 45); do
    if curl -s http://127.0.0.1:8090/v1/models >/dev/null 2>&1; then
        echo ":8090 restored and ready for daily use!"
        break
    fi
    sleep 2
done

# -----------------------------------------------------------------------------
# PHASE 7: Generate Scorecard Report
# -----------------------------------------------------------------------------
echo ""
echo "=== PHASE 7: Compiling Comparative Scorecard Report ==="
"$PYTHON" "$REPO/dev/benchmarks/benchmark_swift27b_vs_swiftv3.py" --target report

echo ""
echo "=================================================================="
echo "Benchmark Orchestration Completed Successfully at $(date)!"
echo "=================================================================="
