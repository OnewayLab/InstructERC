import argparse
import pickle
import json
import os

EMOTIONAL_LABELS = {
    "iemocap": ["happy", "sad", "neutral", "angry", "excited", "frustrated"],
    "meld": ["neutral", "surprise", "fear", "sad", "joyful", "disgust", "angry"],
    "EmoryNLP": ["Joyful", "Mad", "Peaceful", "Neutral", "Sad", "Powerful", "Scared"],
}
EMOTIONAL_LABEL_TEXTS = {
    dataset: ", ".join([f'"{e}"' for e in emotions])
    for dataset, emotions in EMOTIONAL_LABELS.items()
}
LABEL_CLASS = {
    "positive": ["happy", "excited", "surprise", "joyful", "Joyful", "Powerful"],
    "neutral": ["neutral", "Peaceful", "Neutral"],
    "negative": ["sad", "angry", "frustrated", "fear", "disgust", "angry", "Mad", "Sad", "Scared"],
}
LABEL_CLASS = {f: coarse for coarse, fine in LABEL_CLASS.items() for f in fine}

INPUT_TEMPLATE = """\
You are an expert of sentiment and emotional analysis. Your task is to analyze the emotion of the last turn of a conversation.
{task_statement}\
What follows is a conversation involving several speakers.
<BEGIN OF THE CONVERSATION>
{conversation}\
<END OF THE CONVERSATION>
{biography}\
{instruction}
"""
TASK_STATEMENTS = {
    "2-grain": {
        "iemocap": """There are two levels of emotional categories. The coarse-grained categories include: "positive", "neutral" and "negative". Each coarse-grained category contains several fine-grained categories. Positive emotions include "happy" and "excited". Negative emotions include "sad", "angry" and "frustrated".\n""",
        "meld": """There are two levels of emotional categories. The coarse-grained categories include: "positive", "neutral" and "negative". Each coarse-grained category contains several fine-grained categories. Positive emotions include "surprise" and "joyful". Negative emotions include "fear", "sad", "disgust" and "angry".\n""",
        "EmoryNLP": """There are two levels of emotional categories. The coarse-grained categories include: "Positive", "Neutral" and "Negative". Each coarse-grained category contains several fine-grained categories. Positive emotions include "Joyful" and "Powerful". Neutral emotions include "Neutral" and "Peaceful". Negative emotions include "Mad", "Sad" and "Scared".\n""",
    },
    "coe": {
        dataset: f"""There are {len(EMOTIONAL_LABELS[dataset])} emotional labels: f{emotion_text}.\n"""
        for dataset, emotion_text in EMOTIONAL_LABEL_TEXTS.items()
    },
    "none": {
        dataset: f"""There are {len(EMOTIONAL_LABELS[dataset])} emotional labels: f{emotion_text}.\n"""
        for dataset, emotion_text in EMOTIONAL_LABEL_TEXTS.items()
    },
}
INSTRUCTION = {
    "2-grain": """Analyze the emotion for <Turn_{turn}(Speaker_{speaker}): {utterance}> in the above conversation. The first line of your output must be one of "positive", "neutral" or "negative", and the second line must be a fine-grained emotional category chosen from {emotions}.""",
    "coe": """Let's analyze the emotion for each tuen in the conversation and finally determine the emotional label of the last tuen.""",
    "none": "Please select the emotional label of <Turn_{turn}(Speaker_{speaker}): {utterance}> from {emotions}.",
}

def format_output(cot_type, speakers, emotions):
    if cot_type == "none":
        return emotions[-1]
    elif cot_type == "2-grain":
        return f"{LABEL_CLASS[emotions[-1]]}\n{emotions[-1]}"
    elif cot_type == "coe":
        output = ""
        for i, (speaker, emotion) in enumerate(zip(speakers, emotions)):
            output += f"Turn_{i}(Speaker_{speaker}): {emotion}\n"
        return output

def process_dataset(dataset, window=110, cot_type="none", bio=True):
    """
    `cot_type` can be "none", "2-grain"
    `bio` can be True or False
    """
    speaker_label_dict = {}
    content_target_dict = {}
    content_task_dict = {}
    sentence_dict = {}
    data = pickle.load(open(f"../original_data/{dataset}/{dataset}.pkl", "rb"))

    # Different datasets have different ways of handling speaker_label
    if dataset == "iemocap":
        all_conv_id = data[3] + data[4] + data[5]
        sentence_dict = data[2]
        for conv_id in all_conv_id:
            temp_speaker_list = []
            for speaker_label in data[0][conv_id]:
                if speaker_label == "M":
                    temp_speaker_list.append(0)
                else:
                    temp_speaker_list.append(1)
            speaker_label_dict[conv_id] = temp_speaker_list
    elif dataset == "meld":
        all_conv_id = data[4] + data[5] + data[6]
        sentence_dict = data[3]
        for conv_id in all_conv_id:
            temp_speaker_list = []
            for speaker_label in data[0][conv_id]:
                temp_speaker_list.append(speaker_label.index(1))
            speaker_label_dict[conv_id] = temp_speaker_list
    elif dataset == "EmoryNLP":
        all_conv_id = data[3] + data[4] + data[5]
        sentence_dict = data[2]
        for conv_id in all_conv_id:
            temp_speaker_list = []
            for speaker_label in data[0][conv_id]:
                temp_speaker_list.append(speaker_label.index(1))
            speaker_label_dict[conv_id] = temp_speaker_list

    # 对conversation的utterance进行处理，其中index_w用于处理窗口大小设置下面的起始index
    # Process the utterances in the conversation, where 'index_w' is used to handle the starting index under the window size setting.
    for conv_id in all_conv_id:
        for conv_turn in range(len(sentence_dict[conv_id])):
            conversation_str = ""
            index_w = max(conv_turn - window, 0)
            speakers = speaker_label_dict[conv_id][index_w : conv_turn + 1]
            sentences = sentence_dict[conv_id][index_w : conv_turn + 1]
            emotions = [EMOTIONAL_LABELS[dataset][emo_id] for emo_id in data[1][conv_id][index_w : conv_turn + 1]]
            for i, (speaker_label, sub_sent) in enumerate(zip(speakers, sentences)):
                conversation_str += f'Turn_{i+1}(Speaker_{speaker_label}): {sub_sent}\n'
            task_statement = TASK_STATEMENTS[cot_type][dataset]
            instruction = INSTRUCTION[cot_type].format(
                speaker=speakers[-1], utterance=sentences[-1], emotions=EMOTIONAL_LABEL_TEXTS[dataset], turn=i+1
            )
            content_task_dict[f"{conv_id}_{conv_turn}"] = INPUT_TEMPLATE.format(
                task_statement=task_statement, conversation=conversation_str, biography="", instruction=instruction
            )
            content_target_dict[f"{conv_id}_{conv_turn}"] = format_output(cot_type, speakers, emotions)

    if dataset == "iemocap":
        train_ids, test_ids, valid_ids = data[3], data[4], data[5]
    elif dataset == "meld":
        train_ids, test_ids, valid_ids = data[4], data[5], data[6]
    elif dataset == "EmoryNLP":
        train_ids, test_ids, valid_ids = data[3], data[4], data[5]

    new_train_id, new_test_id, new_valid_id = [], [], []
    for train_id in train_ids:
        for conv_turn in range(len(sentence_dict[train_id])):
            new_train_id.append(f"{train_id}_{conv_turn}")

    for test_id in test_ids:
        for conv_turn in range(len(sentence_dict[test_id])):
            new_test_id.append(f"{test_id}_{conv_turn}")

    for valid_id in valid_ids:
        for conv_turn in range(len(sentence_dict[valid_id])):
            new_valid_id.append(f"{valid_id}_{conv_turn}")

    # Save data to json files
    data_path = f"../processed_data/{dataset}/{cot_type}_bio{bio}"
    os.makedirs(data_path, exist_ok=True)

    with open(f"{data_path}/train.json", "w") as f_train:
        for train_id in new_train_id:
            f_train.write(
                json.dumps(
                    {"input": f"{content_task_dict[train_id]}", "target": f"{content_target_dict[train_id]}"},
                    ensure_ascii=False,
                )
                + "\n"
            )

    with open(f"{data_path}/test.json", "w") as f_test:
        for test_id in new_test_id:
            f_test.write(
                json.dumps(
                    {"input": f"{content_task_dict[test_id]}", "target": f"{content_target_dict[test_id]}"},
                    ensure_ascii=False,
                )
                + "\n"
            )

    with open(f"{data_path}/valid.json", "w") as f_valid:
        for valid_id in new_valid_id:
            f_valid.write(
                json.dumps(
                    {"input": f"{content_task_dict[valid_id]}", "target": f"{content_target_dict[valid_id]}"},
                    ensure_ascii=False,
                )
                + "\n"
            )

    with open(f"../processed_data/dataset_info.json", "r", encoding="utf-8") as f:
        dataset_info = json.load(f)
    dataset_info[f"{dataset}_{cot_type}_bio{bio}_train"] = f"{data_path}/train.json"
    dataset_info[f"{dataset}_{cot_type}_bio{bio}_valid"] = f"{data_path}/valid.json"
    dataset_info[f"{dataset}_{cot_type}_bio{bio}_test"] = f"{data_path}/test.json"

    return data_path


parser = argparse.ArgumentParser(description="Data processing script")
parser.add_argument("--dataset", type=str, default="iemocap", help="Dataset name or path")
parser.add_argument("--historical_window", type=int, default=20, help="Historical window size")
parser.add_argument("--cot_type", type=str, default="none", help="Choose from 'none', '2-grain'")
parser.add_argument("--bio", type=str, default="False", help="Add biography or not")
args = parser.parse_args()

if args.bio == "True":
    args.bio = True
elif args.bio == "False":
    args.bio = False

# Process data
processed_data_path = process_dataset(
    dataset=args.dataset, window=args.historical_window, cot_type=args.cot_type, bio=args.bio
)

print(processed_data_path)
