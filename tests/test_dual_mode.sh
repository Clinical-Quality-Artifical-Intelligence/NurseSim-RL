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
output=$(./run.sh 2>&1 &)
pid=$!
sleep 2
kill $pid 2>/dev/null || true

if echo "$output" | grep -q "AgentBeats A2A mode"; then
    echo "✓ A2A mode launched correctly"
else
    echo "✗ A2A mode failed"
    exit 1
fi

# Test 2: Gradio Mode
echo ""
echo "Test 2: Setting AGENT_MODE=gradio"
export AGENT_MODE=gradio
output=$(./run.sh 2>&1 &)
pid=$!
sleep 2
kill $pid 2>/dev/null || true

if echo "$output" | grep -q "Gradio demo"; then
    echo "✓ Gradio mode launched correctly"
else
    echo "✗ Gradio mode failed"
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
