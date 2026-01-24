#!/bin/bash
# Dual-Mode Configuration Test
# Tests that the agent correctly switches between Gradio and A2A modes

set -e

echo "========================================"
echo "NurseSim-Triage Dual-Mode Test"
echo "========================================"
echo ""

# Test 1: A2A Mode
echo "Test 1: Setting AGENT_MODE=a2a"
export AGENT_MODE=a2a
# Run in background and redirect to file
./run.sh > a2a_output.log 2>&1 &
pid=$!
echo "Process started with PID: $pid"
sleep 5
# Kill the process
kill $pid 2>/dev/null || true

# Check output file
if grep -q "AgentBeats A2A mode" a2a_output.log; then
    echo "✓ A2A mode launched correctly"
    cat a2a_output.log
    rm a2a_output.log
else
    echo "✗ A2A mode failed. Log output:"
    cat a2a_output.log
    rm a2a_output.log || true
    exit 1
fi

# Test 2: Gradio Mode
echo ""
echo "Test 2: Setting AGENT_MODE=gradio"
export AGENT_MODE=gradio
./run.sh > gradio_output.log 2>&1 &
pid=$!
echo "Process started with PID: $pid"
sleep 5
kill $pid 2>/dev/null || true

if grep -q "Gradio demo" gradio_output.log; then
    echo "✓ Gradio mode launched correctly"
    cat gradio_output.log
    rm gradio_output.log
else
    echo "✗ Gradio mode failed. Log output:"
    cat gradio_output.log
    rm gradio_output.log || true
    exit 1
fi

# Test 3: Invalid Mode
echo ""
echo "Test 3: Invalid AGENT_MODE"
export AGENT_MODE=invalid
if ./run.sh 2>&1 | grep -q "Error: Invalid AGENT_MODE"; then
    echo "✓ Invalid mode properly rejected"
else
    echo "✗ Invalid mode handling failed"
    exit 1
fi

echo ""
echo "========================================"
echo "✓ All dual-mode tests passed!"
echo "========================================"
