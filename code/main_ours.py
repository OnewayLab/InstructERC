from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoConfig,
    get_linear_schedule_with_warmup,
)
import torch
import deepspeed
import json
import math
import os
import random
from torch.utils.data import DataLoader, SequentialSampler
import numpy as np
from data_utils.data_utils import *
from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam
from peft import LoraConfig, get_peft_model
import argparse
from deepspeed.utils.zero_to_fp32 import load_state_dict_from_zero_checkpoint

# packages for ERC
from sklearn import metrics
from sklearn.metrics import accuracy_score, f1_score


# evaluation function for ERC
def get_labels_attr(dataset):
    label_list_set = {
        "iemocap": ["happy", "sad", "neutral", "angry", "excited", "frustrated"],
        "meld": ["neutral", "surprise", "fear", "sad", "joyful", "disgust", "angry"],
        "EmoryNLP": ["Joyful", "Mad", "Peaceful", "Neutral", "Sad", "Powerful", "Scared"],
        "dialydailog": ["happy", "neutral", "angry", "sad", "fear", "surprise", "disgust"],
    }
    label_str_set = {
        "iemocap": "'happy', 'sad', 'neutral', 'angry', 'excited', 'frustrated'",
        "meld": "'neutral', 'surprise', 'fear', 'sad', 'joyful', 'disgust', 'angry'",
        "EmoryNLP": "'Joyful','Mad','Peaceful', 'Neutral','Sad','Powerful','Scared'",
        "dialydailog": "'happy', 'neutral', 'angry', 'sad', 'fear', 'surprise','disgust'",
    }

    emotional_label_dict = {text_label: num_label for num_label, text_label in enumerate(label_list_set[dataset])}
    emotional_label_str = label_str_set[dataset]
    return emotional_label_dict, emotional_label_str


def report_score(dataset, golds, preds, mode="test"):
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
    res["mode"] = mode
    for k, v in res.items():
        if isinstance(v, float):
            res[k] = round(v * 100, 3)

    res_matrix = metrics.classification_report(golds, preds, target_names=target_names, digits=digits)

    return res, res_matrix


def match_text(text, word_set_):
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


def edit_distance(s1, s2):
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


def optimize_output(output, label_set):
    """
    Calculate output and label_ Set the editing distance of each label in the set and return the label corresponding to the minimum editing distance
    """
    min_distance = float("inf")
    optimized_output = None
    for label in label_set:
        distance = edit_distance(output, label)
        if distance < min_distance:
            min_distance = distance
            optimized_output = label
    return optimized_output


device = torch.device("cuda")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

world_size = int(os.getenv("WORLD_SIZE", "1"))

parser = argparse.ArgumentParser()
parser.add_argument(
    "--dataset",
    type=str,
    required=True,
    choices=["iemocap", "meld", "EmoryNLP"],
    help="Datasets that need to be evaluated",
)
parser.add_argument(
    "--data_dir",
    type=str,
    required=True,
    help="The input data dir. Should contain the source and target files for the task.",
)
parser.add_argument(
    "--model_name_or_path",
    type=str,
    required=True,
    help="Path to pretrained model or model identifier from huggingface.co/models",
)
parser.add_argument(
    "--checkpoint_dir",
    type=str,
    default=None,
    help="Path to the fine-tuned model checkpoint.",
)
parser.add_argument("--gradient_checkpointing", action="store_true")
parser.add_argument(
    "--output_dir",
    type=str,
    required=True,
    help="Path to save trained model.",
)
parser.add_argument("--mode", type=str, default="sft")
parser.add_argument(
    "--deepspeed_config",
    type=str,
    default="./data_utils/deepspeed_config.json",
    help="Path to save trained model.",
)
parser.add_argument(
    "--num_train_epochs",
    default=10,
    type=int,
    help="Number of training epochs.",
)
parser.add_argument(
    "--gradient_accumulation_steps",
    default=1,
    type=int,
    help="gradient accumulation steps",
)
parser.add_argument(
    "--warmup_ratio",
    default=0.1,
    type=float,
    help="The ratio of warmup.",
)
parser.add_argument("--local_rank", default=-1)
parser.add_argument("--warmup_steps", default=None, type=int)
parser.add_argument("--learning_rate", default=3e-5, type=float)
parser.add_argument(
    "--max_seq_length",
    default=256,
    type=int,
    help="Max output seq length",
)
parser.add_argument(
    "--max_length",
    default=2048,
    type=int,
    help="Max output seq length",
)
parser.add_argument("--weight_decay", default=0.0, type=float, help="weight decay when updating parameters.")
parser.add_argument(
    "--save_steps",
    default=1000,
    type=int,
)
parser.add_argument(
    "--zero_shot",
    default=True,
    # action='store_true',
)
parser.add_argument(
    "--lora",
    default=False,
    # type=bool,
)
parser.add_argument(
    "--lora_dim",
    type=int,
    default=16,
)
parser.add_argument(
    "--lora_alpha",
    type=int,
    default=16,
)
parser.add_argument(
    "--lora_dropout",
    type=float,
    default=0.05,
)
parser.add_argument(
    "--lora_module_name",
    type=str,
    default="q_proj,k_proj,v_proj,query_key_value",
)
parser.add_argument("--batch_size", default=32, type=int)
parser.add_argument("--eval_batch_size", default=4, type=int)
parser.add_argument("--top_k", default=None, type=int)
parser.add_argument("--num_beams", default=1, type=int)
parser.add_argument("--seed", default=42, type=int)
parser.add_argument("--top_p", type=float, default=None)
parser.add_argument("--clip_norm", type=float, default=1.0)
parser.add_argument("--temp", type=float, default=None, help="Temperature for model generation.")
parser.add_argument(
    "--do_train",
    default=False
)
parser.add_argument(
    "--do_eval",
    default=False
)
parser.add_argument("--few_shot", type=bool, default=False)
parser.add_argument("--statistic_mode", default="True")
parser.add_argument("--offload_optimizer", action="store_true")
parser.add_argument("--data_percent", default=1.0, type=float)
args = parser.parse_args()
do_sample = args.top_k is not None or args.top_p is not None or args.num_beams > 1 or args.temp is not None
"""
args.top_k is not None (top_k is a parameter that limits the number of possible tokens to consider at each step)
args.top_p is not None (top_p is a parameter that limits the cumulative probability of the next token)
args.num_beams > 1(num_beams is a parameter that controls the number of beams to use for beam search)
args.temp is not None (temp is a parameter that controls the temperature of softmax distribution used for sampling)

# if any of those conditions are true, then do_sample will be set to True.
# Otherwise, it will be set to False.
# This variable is likely being used later in the code to determine whether or not to use sampling when generating text.
"""

if args.do_train == "True":
    args.do_train = True
else:
    args.do_train = False

if args.do_eval == "True":
    args.do_eval = True
else:
    args.do_eval = False
if args.lora == "True":
    args.lora = True
else:
    args.lora = False

model_args = {
    "dataset": args.dataset,
    "model_name_or_path": args.model_name_or_path,
    "checkpoint_dir": args.checkpoint_dir,
    "data_dir": args.data_dir,
    "max_seq_length": args.max_seq_length,
    "batch_size": args.batch_size,
    "eval_batch_size": args.eval_batch_size,
    "gradient_accumulation_steps": args.gradient_accumulation_steps,
    "learning_rate": args.learning_rate,
    "num_train_epochs": args.num_train_epochs,
    "save_steps": args.save_steps,
    "output_dir": args.output_dir,
    "max_length": args.max_length,
    "warmup_ratio": args.warmup_ratio,
    "warmup_steps": args.warmup_steps,
    "weight_decay": args.weight_decay,
    "data_dir": args.data_dir,
    "lora": args.lora,
    "lora_dim": args.lora_dim,
    "lora_dropout": args.lora_dropout,
    "lora_alpha": args.lora_alpha,
    "lora_module_name": args.lora_module_name,
    "num_beams": args.num_beams,
    "top_k": args.top_k,
    "top_p": args.top_p,
    "do_sample": do_sample,
    "seed": args.seed,
    "do_train": args.do_train,
    "do_eval": args.do_eval,
    "offload_optimizer": args.offload_optimizer,
    "deepspeed_config": args.deepspeed_config,
    "zero_shot": args.zero_shot,
    "statistic_mode": args.statistic_mode,
    "mode": args.mode,
    "emotion_prediction": False,  # 为了兼容原代码
    "beta": 0.1,  # 为了兼容原代码
    "gradient_checkpointing": args.gradient_checkpointing,
    "data_percent": args.data_percent,
}
args = ModelArgs()
args.update(model_args)

print(args)

if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir, exist_ok=True)

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

lora_config = LoraConfig(
    task_type="CAUSAL_LM",
    r=args.lora_dim,
    lora_alpha=args.lora_alpha,
    lora_dropout=args.lora_dropout,
    target_modules=args.lora_module_name.split(","),
    bias="none",
)
with open(args.deepspeed_config, "r", encoding="utf-8") as f:
    deepspeed_config = json.load(f)
deepspeed_config["train_batch_size"] = args.batch_size
deepspeed_config["gradient_accumulation_steps"] = args.gradient_accumulation_steps
if deepspeed_config["zero_optimization"]["stage"] == 3:
    deepspeed_config["zero_optimization"]["mics_shard_size"] = world_size


def getOptimizerGroup(model):
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in model.named_parameters() if (not any(nd in n for nd in no_decay) and p.requires_grad)
            ],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if (any(nd in n for nd in no_decay) and p.requires_grad)],
            "weight_decay": 0.0,
        },
    ]

    return optimizer_grouped_parameters


def _get_input_dict(batch):
    input_ids, labels, attention_mask, type_token_ids = (
        batch["input_ids"],
        batch["labels"],
        batch["attention_mask"],
        batch["type_token_ids"],
    )

    return {"input_ids": input_ids.to(device), "labels": labels.to(device), "attention_mask": attention_mask.to(device)}


## prepare model
config = AutoConfig.from_pretrained(args.model_name_or_path, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, trust_remote_code=True)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.unk_token
if tokenizer.eos_token is None:
    tokenizer.eos_token = tokenizer.pad_token
if args.model_type == "decoder":
    tokenizer.padding_side = "left"
if args.lora:
    model = get_peft_model(model, lora_config)
if args.gradient_checkpointing:
    if "chatglm" in args.model_name_or_path:
        model.supports_gradient_checkpointing = True
        model.transformer.gradient_checkpointing = True
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

if args.checkpoint_dir is not None:
    from deepspeed.utils.zero_to_fp32 import load_state_dict_from_zero_checkpoint

    model = load_state_dict_from_zero_checkpoint(model, args.checkpoint_dir)


num_parameters = get_parameter_number(model)
with open(os.path.join(args.output_dir, "model_params.json"), "w", encoding="utf-8") as f:
    f.write(json.dumps(num_parameters, indent=5))

## prepare data
train_file = os.path.join(args.data_dir, "train.json")
dev_file = os.path.join(args.data_dir, "test.json")
if not os.path.exists(dev_file):
    print("*****    Desire to evaluate, but test.json file not found *****")
    args.do_eval = False
train_dataset, dev_dataset = None, None
train_collator, dev_collator = None, None
if args.do_train:
    df_train = read_data(train_file, percent=args.data_percent, random_seed=args.seed)
    train_dataset = Seq2SeqDataset(args, df_train, mode="train")
    train_collator = Seq2SeqCollator(args, tokenizer, mode="train")
if args.do_eval:
    dev_datasets = []
    df_dev = read_data(dev_file, percent=1.0, random_seed=args.seed)
    dev_dataset = Seq2SeqDataset(args, df_dev, mode="dev")
    dev_collator = Seq2SeqCollator(args, tokenizer, mode="dev")

stop_word_list = ["sep_token_id", "eos_token_id", "pad_token_id"]
stop_ids = []
for stop_word in stop_word_list:
    id_ = getattr(tokenizer, stop_word)
    if id_ is not None:
        stop_ids.append(id_)
stop_criteria = KeywordsStoppingCriteria(stop_ids)
## prepare deepspeed model training
if args.do_train:
    t_total = math.ceil(len(train_dataset) / args.batch_size) * args.num_train_epochs
    warmup_steps = math.ceil(t_total * args.warmup_ratio) if args.warmup_steps is None else args.warmup_steps
    args.warmup_steps = warmup_steps

    # if args.offload_optimizer or args.zero_shot:
    if args.offload_optimizer:
        deepspeed_config["zero_optimization"]["offload_optimizer"]["device"] = "cpu"

    optimizer_grouped_parameters = getOptimizerGroup(model=model)

    optimizer_class = (
        DeepSpeedCPUAdam if deepspeed_config["zero_optimization"]["offload_optimizer"]["device"] == "cpu" else FusedAdam
    )
    optimizer = optimizer_class(optimizer_grouped_parameters, lr=args.learning_rate, betas=[0.9, 0.95])
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer, num_training_steps=t_total, num_warmup_steps=warmup_steps
    )

    model_engine, optimizer, train_dataloader, lr_scheduler = deepspeed.initialize(
        model=model,
        training_data=train_dataset,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        config=deepspeed_config,
        collate_fn=train_collator,
    )
    model = model_engine
    should_save = True
elif args.do_eval:
    if not args.zero_shot:
        model = load_state_dict_from_zero_checkpoint(model, args.output_dir)
    dtype = torch.half if args.model_type == "decoder" else torch.float32
    model_engine = deepspeed.init_inference(
        model,
        mp_size=world_size,
        replace_with_kernel_inject=True,
        dtype=dtype,
    )
    model = model_engine.module

if __name__ == "__main__":
    eval_score_list = []
    emotional_label_dict, emotional_label_str = get_labels_attr(dataset=args.dataset)
    if args.do_train:
        global_steps = 0
        best_f1_score = 0
        for epoch in range(args.num_train_epochs):
            model.train()
            batch_iterator = tqdm(
                train_dataloader,
                desc=f"Runing epoch{epoch} / {args.num_train_epochs}",
                disable=False,
                mininterval=0,
            )
            for step, batch in enumerate(batch_iterator):
                batch = _get_input_dict(batch)
                outputs = model(**batch)
                loss = outputs.loss
                model.backward(loss)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_norm)
                model.step()

                current_loss = loss.item()
                batch_iterator.set_description(
                    f"Epochs {epoch}/{args.num_train_epochs}. Running Loss: {current_loss:9.4f}"
                )

                if step % args.gradient_accumulation_steps == 0:
                    global_steps += 1
                    should_save = True
                if global_steps % args.save_steps == 0 and should_save:
                    model.save_checkpoint(args.output_dir)
                    should_save = False

            # model starts to evaluation
            model.eval()
            targets = list(df_dev["output"])
            eval_sampler = SequentialSampler(dev_dataset)
            eval_dataloader = DataLoader(
                dev_dataset,
                batch_size=args.eval_batch_size,
                sampler=eval_sampler,
                collate_fn=dev_collator,
                num_workers=8,
            )
            all_outputs = []

            preds_for_eval_path = os.path.join(args.output_dir, f"preds_for_eval_{epoch}.text")
            print("\n*****    Evaluating  *****\n")
            eval_inputs_iter = []
            for eval_step, eval_batch in enumerate(tqdm(eval_dataloader)):
                eval_batch = eval_batch.to(device)
                eval_inputs_iter.extend(eval_batch["input_ids"])
                max_length_this_batch = eval_batch["input_ids"].size(-1) if args.model_type == "decoder" else 0
                with torch.no_grad():
                    if "chatglm" in args.model_name_or_path:
                        outputs = model.generate(
                            **eval_batch,
                            num_beams=args.num_beams,
                            max_length=args.max_length,
                            do_sample=args.do_sample,
                            top_p=args.top_p,
                            top_k=args.top_k,
                        )
                    else:
                        if "token_type_ids" in eval_batch:
                            token_type_ids = eval_batch.pop("token_type_ids")
                        outputs = model.generate(
                            **eval_batch,
                            num_beams=args.num_beams,
                            top_k=args.top_k,
                            top_p=args.top_p,
                            early_stopping=True,
                            max_length=args.max_length,
                            length_penalty=2.0,
                            repetition_penalty=1.0,
                            num_return_sequences=1,
                        )
                outputs[outputs[:, :] < 0] = tokenizer.pad_token_id
                all_outputs.extend(outputs)
            eval_inputs_iter = [
                tokenizer.decode(e_id, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                for e_id in eval_inputs_iter
            ]
            # all_outputs
            outs = [
                tokenizer.decode(o_id, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                for o_id in all_outputs
            ]
            preds_for_eval = []
            all_answers = []  # 模型预测结果
            for index, o in enumerate(outs):
                this_input = eval_inputs_iter[index]
                if args.model_type == "decoder":
                    if this_input in o:
                        answer = o.replace(this_input, "").strip().rstrip()
                    else:
                        output_ids = all_outputs[index][args.max_seq_length :]
                        answer = tokenizer.decode(
                            output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
                        )
                else:
                    answer = o
                answer = answer.strip().rstrip()
                all_answers.append(answer)

                this_eval_instance = {
                    "index": index,
                    "input": this_input,
                    "output": answer,
                    "target": targets[index],
                }
                preds_for_eval.append(this_eval_instance)
            if args.statistic_mode == "True":
                preds = []
                golds = []
                confuse_case = []
                for index, answer in enumerate(all_answers):
                    golds += [emotional_label_dict[targets[index].split("\n")[-1]]]  # 真实标签（数字）
                    answer = answer.strip().split("\n")[-1]
                    match_res = match_text(answer, list(emotional_label_dict.keys()))
                    if match_res:
                        preds += [emotional_label_dict[match_res[0]]]
                    else:
                        preds += [emotional_label_dict[optimize_output(answer, list(emotional_label_dict.keys()))]]
                        confuse_case += [index]

                if len(preds) == len(all_answers):
                    score, res_matrix = report_score(dataset=args.dataset, golds=golds, preds=preds)

                # statisics of model's output
                with open(preds_for_eval_path, "w", encoding="utf-8") as f:
                    f.write(json.dumps(score))
                    f.write(f"\n{res_matrix}")
                    f.write(f"\nconfuse_case:{confuse_case} \n\n")
                    f.write(json.dumps(preds_for_eval, indent=5, ensure_ascii=False))

                if best_f1_score < score["F1_SA"]:
                    best_f1_score = score["F1_SA"]
                    tokenizer.save_pretrained(args.output_dir)
                    config.save_pretrained(args.output_dir)
                    args.save(args.output_dir)
                    #     model.save_checkpoint(args.output_dir)
                    with open(os.path.join(args.output_dir, "deepspeed_config.json"), "w", encoding="utf-8") as f:
                        f.write(json.dumps(deepspeed_config, indent=5))
            if args.statistic_mode != "True":
                with open(preds_for_eval_path, "w", encoding="utf-8") as f:
                    f.write(json.dumps(preds_for_eval, indent=5, ensure_ascii=False))
                tokenizer.save_pretrained(args.output_dir)
                config.save_pretrained(args.output_dir)
                args.save(args.output_dir)
                model.save_checkpoint(args.output_dir)
                with open(os.path.join(args.output_dir, "deepspeed_config.json"), "w", encoding="utf-8") as f:
                    f.write(json.dumps(deepspeed_config, indent=5))

    if not args.do_train and args.do_eval:
        # model starts to evaluation
        model.eval()
        targets = list(df_dev["output"])
        eval_sampler = SequentialSampler(dev_dataset)
        eval_dataloader = DataLoader(
            dev_dataset, batch_size=args.eval_batch_size, sampler=eval_sampler, collate_fn=dev_collator, num_workers=8
        )
        all_outputs = []

        preds_for_eval_path = os.path.join(args.output_dir, f"preds_for_eval.text")
        print("\n*****    Evaluating  *****\n")
        eval_inputs_iter = []
        for eval_step, eval_batch in enumerate(tqdm(eval_dataloader)):
            eval_batch = eval_batch.to(device)
            eval_inputs_iter.extend(eval_batch["input_ids"])
            max_length_this_batch = eval_batch["input_ids"].size(-1) if args.model_type == "decoder" else 0
            with torch.no_grad():
                if "chatglm" in args.model_name_or_path:
                    outputs = model.generate(
                        **eval_batch,
                        num_beams=args.num_beams,
                        # max_length=max_length_this_batch + args.max_length,
                        max_length=args.max_length,
                        do_sample=args.do_sample,
                        top_p=args.top_p,
                        top_k=args.top_k,
                    )
                else:
                    if "token_type_ids" in eval_batch:
                        token_type_ids = eval_batch.pop("token_type_ids")
                    outputs = model.generate(
                        **eval_batch,
                        num_beams=args.num_beams,
                        top_k=args.top_k,
                        top_p=args.top_p,
                        early_stopping=True,
                        # max_length=max_length_this_batch + args.max_length,
                        max_length=args.max_length,
                        length_penalty=2.0,
                        repetition_penalty=1.0,
                        num_return_sequences=1,
                        # stopping_criteria=StoppingCriteriaList([stop_criteria])
                    )
            outputs[outputs[:, :] < 0] = tokenizer.pad_token_id
            all_outputs.extend(outputs)
        eval_inputs_iter = [
            tokenizer.decode(e_id, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            for e_id in eval_inputs_iter
        ]
        # all_outputs
        outs = [
            tokenizer.decode(o_id, skip_special_tokens=True, clean_up_tokenization_spaces=True) for o_id in all_outputs
        ]
        preds_for_eval = []
        all_answers = []
        for index, o in enumerate(outs):
            this_input = eval_inputs_iter[index]
            if args.model_type == "decoder":
                if this_input in o:
                    answer = o.replace(this_input, "").strip().rstrip()
                else:
                    output_ids = all_outputs[index][args.max_seq_length :]
                    answer = tokenizer.decode(output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            else:
                answer = o
            answer = answer.strip().rstrip()
            all_answers.append(answer)

            this_eval_instance = {
                "index": index,
                "input": this_input,
                "output": answer,
                "target": targets[index],
            }
            preds_for_eval.append(this_eval_instance)

        preds = []
        golds = []
        # confuse_index = len(emotional_label_dict)
        # bad_case = []
        confuse_case = []
        for index, answer in enumerate(all_answers):
            golds += [emotional_label_dict[targets[index]]]
            match_res = match_text(answer, list(emotional_label_dict.keys()))
            if match_res:
                preds += [emotional_label_dict[match_res[0]]]
            else:
                # preds += [confuse_index]
                preds += [emotional_label_dict[optimize_output(answer, list(emotional_label_dict.keys()))]]
                confuse_case += [index]

        if len(preds) == len(all_answers):
            score, res_matrix = report_score(dataset=args.dataset, golds=golds, preds=preds)
            eval_score_list.append(score)

        # statisics of model's output

        with open(preds_for_eval_path, "w", encoding="utf-8") as f:
            f.write(json.dumps(score))
            f.write(f"\n{res_matrix}")
            f.write(f"\nconfuse_case: {confuse_case}  \n")
            f.write(f"\nThe num of confuse_case is : {len(confuse_case)} \n")
            f.write(json.dumps(preds_for_eval, indent=5, ensure_ascii=False))

    for i in eval_score_list:
        print(i)
