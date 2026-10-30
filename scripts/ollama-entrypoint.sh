#!/bin/bash
# Start Ollama in the background
/bin/ollama serve &
pid=$!

# Wait for Ollama to be ready
echo "Waiting for Ollama to start..."
while ! ollama list > /dev/null 2>&1; do
    sleep 1
done

# Resolve target model (defaults to qwen3:1.7b)
MODEL="${OLLAMA_MODEL:-qwen3:1.7b}"

# Pull the model
echo "Pulling model ${MODEL}..."
ollama pull "${MODEL}"
echo "Model pull complete."

# Pre-warm model and keep resident in RAM permanently
echo "Pre-warming ${MODEL} with keep_alive=-1..."
ollama run "${MODEL}" --keepalive -1 "hi" > /dev/null 2>&1 || true
echo "Model pre-warmed."

# Wait for the background process to finish
wait $pid

