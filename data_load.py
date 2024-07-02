import json
from pathlib import Path
from datasets import Dataset
from transformers import T5Tokenizer
import pickle

# Define the data directory
data_dir = Path("resources/data/asset/dataset")
validation_file = data_dir / "asset.valid.jsonl"
test_file = data_dir / "asset.test.jsonl"

# Load the model and tokenizer
model_name = "google/flan-t5-base"
tokenizer = T5Tokenizer.from_pretrained(model_name)

# Function to load data from a JSONL file
def load_jsonl(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return [json.loads(line.strip()) for line in f]

# Load validation and test data
validation_data = load_jsonl(validation_file)
test_data = load_jsonl(test_file)

# Convert data to Hugging Face Dataset format
def prepare_dataset(data):
    complex_sentences = [item["complex"] for item in data]
    simple_sentences = [item["simple"] for item in data]
    return Dataset.from_dict({"complex": complex_sentences, "simple": simple_sentences})

validation_dataset = prepare_dataset(validation_data)
test_dataset = prepare_dataset(test_data)

def find_max_token_length(dataset, is_input=True):
    max_length = 0
    for example in dataset:
        if is_input:
            lengths = [len(tokenizer(example["complex"]).input_ids)]
        else:
            lengths = [len(tokenizer(simple_sentence).input_ids) for simple_sentence in example["simple"]]
        max_length = max(max_length, *lengths)
    return max_length

# Save datasets to disk
# with open('validation_dataset.pkl', 'wb') as f:
#     pickle.dump(validation_dataset, f)
#
# with open('test_dataset.pkl', 'wb') as f:
#     pickle.dump(test_dataset, f)

# Find maximum token lengths for both inputs and labels
max_length_input = max(find_max_token_length(validation_dataset, is_input=True),
                       find_max_token_length(test_dataset, is_input=True))
max_length_label = max(find_max_token_length(validation_dataset, is_input=False),
                       find_max_token_length(test_dataset, is_input=False))

print(f"Maximum token length for inputs: {max_length_input}")
print(f"Maximum token length for labels: {max_length_label}")

# Maximum token length for inputs: 79
# Maximum token length for labels: 104
# Validation dataset size: 2000
# Test dataset size: 359


# Print the number of samples in each dataset to verify
print(f"Validation dataset size: {len(validation_dataset)}")
print(f"Test dataset size: {len(test_dataset)}")

print("\n Validation dataset samples:")
for i in range(3):
    print(f"Complex: {validation_dataset[i]['complex']}")
    print(f"Simple: {validation_dataset[i]['simple']}")

print("\n Test dataset samples:")
for i in range(3):
    print(f"Complex: {test_dataset[i]['complex']}")
    print(f"Simple: {test_dataset[i]['simple']}")