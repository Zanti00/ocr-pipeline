#!/bin/bash
# Start Ollama in the background
/bin/ollama serve &
pid=$!

# Wait for Ollama to be ready
echo "Waiting for Ollama to start..."
while ! ollama list > /dev/null 2>&1; do
    sleep 1
done

# Pull the model
echo "Pulling model qwen2.5:1.5b..."
ollama pull qwen2.5:1.5b
echo "Model pull complete."

# Pre-warm model and keep resident in RAM permanently
echo "Pre-warming qwen2.5:1.5b with keep_alive=-1..."
ollama run qwen2.5:1.5b --keepalive -1 "hi" > /dev/null 2>&1 || true
echo "Model pre-warmed."

# Wait for the background process to finish
wait $pid
