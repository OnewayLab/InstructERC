import os
import fire
import json
import torch
from typing import Literal
from vllm import LLM, SamplingParams
from sklearn import metrics
from sklearn.metrics import accuracy_score, f1_score


LABELS = {
    "iemocap": ["happy", "sad", "neutral", "angry", "excited", "frustrated"],
    "meld": ["neutral", "surprise", "fear", "sad", "joyful", "disgust", "angry"],
    "EmoryNLP": ["Joyful", "Mad", "Peaceful", "Neutral", "Sad", "Powerful", "Scared"],
    "dialydailog": ["happy", "neutral", "angry", "sad", "fear", "surprise", "disgust"],
}
LABEL2ID = {dataset: {label: i for i, label in enumerate(labels)} for dataset, labels in LABELS.items()}


def main(
    model_name_or_path: str,
    dataset: Literal["iemocap", "meld", "EmoryNLP"],
    data_path: str,
    output_dir: str,
    use_chat_template: bool = False,
    max_tokens: int = 1024,
    temperature: float = 1,
    top_p: float = 1,
    top_k: int = -1,
    **kwargs,
):
    """ERC Tester.

    Args:
        model_name_or_path: Name or path of the model.
        dataset: Dataset name chosen from ["iemocap", "meld", "EmoryNLP"].
        data_path: Path to the dataset file.
        output_path: Path to the output file.
        use_chat_template: Whether to use chat template.
        max_tokens: Maximum tokens to generate.
        temperature: Temperature for sampling.
        top_p: Top-p for sampling.
        top_k: Top-k for sampling.
        **kwargs: Other keyword arguments for the `vllm.LLM` class.
    """
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

    # Load data
    with open(data_path, "r", encoding="utf-8") as f:
        data: list[dict[str, str]] = json.load(f)

    # Inference
    sampling_params = SamplingParams(temperature=temperature, top_p=top_p, top_k=top_k)
    if use_chat_template:
        messages = [[{"role": "user", "content": item["input"]}] for item in data]
        outputs = llm.chat(messages, sampling_params=sampling_params)  # type: ignore
    else:
        prompts = [item["input"] for item in data]
        outputs = llm.generate(prompts, sampling_params=sampling_params)
    outputs = [output.outputs[0].text for output in outputs]

    # Save results
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "predictions.json"), "w", encoding="utf-8") as f:
        json.dump(
            [
                {"input": item["input"], "output": output, "target": item["target"]}
                for item, output in zip(data, outputs)
            ],
            f,
            ensure_ascii=False,
            indent=4,
        )

    # Compute metrics
    label2id = LABEL2ID[dataset]
    predictions, labels = [], []
    confused_cases = []
    for i, (item, output) in enumerate(zip(data, outputs)):
        label = item["target"].strip().split("\n")[-1].strip()
        labels.append(label2id[label])
        prediction = output.strip().split("\n")[-1].strip()
        match_result = _match_text(prediction, LABELS[dataset])
        if match_result:
            predictions.append(label2id[match_result[0]])
        else:
            predictions.append(label2id[_optimize_output(prediction, LABELS[dataset])])
            confused_cases.append(i)
    assert len(predictions) == len(labels)
    score, res_matrix = report_score(dataset, labels, predictions)

    # Save metrics
    with open(os.path.join(output_dir, "metrics.json"), "w", encoding="utf-8") as f:
        f.write(json.dumps(score))
        f.write(f"\n{res_matrix}")
        f.write(f"\nconfuse_case:{confused_cases}\n")


def report_score(dataset: str, golds: list[str], preds: list[str]):
    if dataset == "iemocap":
        target_names = ["hap", "sad", "neu", "ang", "exc", "fru"]
        digits = 6
    elif dataset == "meld":
        target_names = ["neutral", "surprise", "fear", "sad", "joyful", "disgust", "angry"]
        digits = 7
    elif dataset == "EmoryNLP":
        target_names = ["Joyful", "Mad", "Peaceful", "Neutral", "Sad", "Powerful", "Scared"]
        digits = 7

    res = {}
    res["Acc_SA"] = accuracy_score(golds, preds)
    res["F1_SA"] = f1_score(golds, preds, average="weighted")
    for k, v in res.items():
        if isinstance(v, float):
            res[k] = round(v * 100, 3)
    res_matrix = metrics.classification_report(golds, preds, target_names=target_names, digits=digits)
    return res, res_matrix


def _match_text(text, word_set_):
    if text is None:
        return []
    len_text = len(text)
    s_idx = 0
    match_res = []
    while s_idx < len_text:
        cache = []
        span_length = 1
        while span_length < 12 and s_idx + span_length <= len_text:
            span = text[s_idx : s_idx + span_length]
            if span in word_set_:
                cache.append(span)
            span_length += 1
        if len(cache) > 0:
            match_res.append(cache[-1])
            s_idx += len(cache[-1])
        else:
            s_idx += 1
    return match_res


def _optimize_output(output, label_set):
    """
    Calculate output and label_ Set the editing distance of each label in the set and return the label corresponding to the minimum editing distance
    """
    min_distance = float("inf")
    optimized_output = ""
    for label in label_set:
        distance = _edit_distance(output, label)
        if distance < min_distance:
            min_distance = distance
            optimized_output = label
    return optimized_output


def _edit_distance(s1, s2):
    """
    Calculate the editing distance between two strings
    """
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1]) + 1
    return dp[m][n]


if __name__ == "__main__":
    fire.Fire(main)
