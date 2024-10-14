## Run
An example for vllm

```bash
CUDA_VISIBLE_DEVICES=0 VLLM_SERVER_URL=http://localhost:8000/v1 OPENAI_API_KEY=token-abc123 torchrun --nproc-per-node 1 --master-port 1234 examples/vLLM/gsm8k/inference.py --base_lm /dataset/crosspipe/OriginModel/Llama-3-8B --n_action 1 --n_confidence 1 --n_iters 1 --temperature 0.0
```


