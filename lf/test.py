import fire
import json
from vllm import LLM, SamplingParams

prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]
sampling_params = SamplingParams(temperature=0.8, top_p=0)

llm = LLM(model="facebook/opt-125m")

outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generat

def main(dataset: str, data_path: str, model_path: str, adapter_path: str | None = None):
    # Load data
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    prompts = [item["input"] for item in data]

    # Load model
    llm_kwargs = {
        "dtype": "float16",
        "tensor_parallel_size": torch.cuda.device_count(),
        "enable_prefix_caching": True,
        "swap_space": 8,
        "gpu_memory_utilization": 0.96,
        "max_logprobs": 0,
        "num_scheduler_steps": 8,
        "enable_chunked_prefill": True,
        "preemption_mode": "swap",
        **kwargs,
    }
    llm = LLM(
        model=model_name_or_path,
        trust_remote_code=True,
        **llm_kwargs,
    )
    self.tokenizer = self.llm.get_tokenizer()


if __name__ == "__main__":
    fire.Fire(main)