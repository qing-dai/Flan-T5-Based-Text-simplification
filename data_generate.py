import json
import pickle
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
from bert_score import score as bert_score
from evaluate import load
import random

# Load the tokenizer and model
model_name_or_path = "/home/ubuntu/projects/TS/result/flan-t5-finetuned0"
tokenizer = T5Tokenizer.from_pretrained(model_name_or_path)
model = T5ForConditionalGeneration.from_pretrained(model_name_or_path)

# Check if CUDA (GPU) is available and move the model to the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Load test dataset
with open('test_dataset.pkl', 'rb') as f:
    test_dataset = pickle.load(f)

# Define the prompt template
prompt_template = (
    "Please rewrite the following complex sentence in order to make it easier to understand "
    "by non-native speakers of English. You can do so by replacing complex words with simpler synonyms "
    "(i.e. paraphrasing), deleting unimportant information (i.e. compression), and/or splitting a long complex "
    "sentence into several simpler ones. The final simplified sentence needs to be grammatical, fluent, and retain "
    "the main ideas of its original counterpart without altering its meaning.\n\n"
    "Complex: {complex_sentence}\nSimple:"
)


def generate_simplification(complex_sentence):
    # Prepare the input prompt
    input_prompt = prompt_template.format(complex_sentence=complex_sentence)
    # Tokenize the input
    inputs = tokenizer.encode(input_prompt, return_tensors="pt", max_length=512, truncation=True).to(device)
    # Generate the output
    outputs = model.generate(inputs, max_length=150, num_beams=5, early_stopping=True)
    # Decode the output
    decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return decoded_output


# Prepare data for metric calculation
predictions = []
output_data = []

for i in range(len(test_dataset)):
    complex_sentence = test_dataset[i]["complex"]
    prediction = generate_simplification(complex_sentence)
    refs = test_dataset[i]["simple"]

    output_data.append({
        "complex": complex_sentence,
        "prediction": prediction,
        "simple": refs
    })

# Save the output data to a JSON file
with open('output_data_with_predictions.json', 'w', encoding='utf-8') as f:
    json.dump(output_data, f, ensure_ascii=False, indent=4)

# Load the saved JSON file to verify the format
with open('output_data_with_predictions.json', 'r', encoding='utf-8') as f:
    loaded_data = json.load(f)

# Print a sample from the loaded data to verify the format
print(json.dumps(loaded_data[:2], indent=4, ensure_ascii=False))
