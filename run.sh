#!/bin/bash
# Launcher script for NurseSim-Triage agent
set -e

# Fix for libgomp Runtime Error on Hugging Face Spaces (CPU Upgrade only)
# If NO GPU (CUDA devices empty), restrict threads to avoid crash
if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
    echo "Running on CPU - Setting OMP_NUM_THREADS=1"
    export OMP_NUM_THREADS=1
else
    echo "Running on GPU (detected) - Allowing automatic threading"
fi


AGENT_MODE=${AGENT_MODE:-a2a}

echo "NurseSim-Triage Launcher"
echo "========================"
echo "Agent Mode: $AGENT_MODE"
echo ""

if [ "$AGENT_MODE" = "gradio" ]; then
    echo "Starting Gradio demo for human interaction..."
    exec python app.py
elif [ "$AGENT_MODE" = "a2a" ]; then
    echo "Starting AgentBeats A2A mode..."
    exec python agent_main.py
else
    echo "Error: Invalid AGENT_MODE '$AGENT_MODE'"
    echo "Valid modes: 'gradio' or 'a2a'"
    exit 1
fi
