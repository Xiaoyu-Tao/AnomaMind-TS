export CUDA_VISIBLE_DEVICES=2
vllm serve ./Model/RL/WSD_0.6B_RL/global_step_225/actor/huggingface \
    --port 8095 \
    --max-model-len 11000 \
    --gpu-memory-utilization 0.95 \
    --tensor-parallel-size 2 \
    --max-num-seqs 400 \
    --enable-auto-tool-choice \
    --tool-call-parser hermes