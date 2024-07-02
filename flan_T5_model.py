import pickle
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
from easse.sari import get_corpus_sari_operation_scores
from easse.bertscore import corpus_bertscore
import random
import json
import numpy as np

# Load the tokenizer and model
model_name_or_path = "google/flan-t5-base"
tokenizer = T5Tokenizer.from_pretrained(model_name_or_path)
model = T5ForConditionalGeneration.from_pretrained(model_name_or_path)

# Check if CUDA (GPU) is available and move the model to the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Load test dataset
with open('test_dataset.pkl', 'rb') as f:
    test_dataset = pickle.load(f)

# Load validation dataset
with open('validation_dataset.pkl', 'rb') as f:
    validation_dataset = pickle.load(f)

# Randomly select 3 samples from the validation dataset
random.seed(42)
examples = random.sample(list(validation_dataset['complex']), 3)

# Create the few-shot examples for the prompt
few_shot_examples = ""
for example in examples:
    index = validation_dataset['complex'].index(example)
    few_shot_examples += (
        f"Complex: {example}\n"
        f"Simple: {validation_dataset['simple'][index][0]}\n"  # Use the first simplification as the example
    )

# Define the prompt template
# Define the prompt template


prompt_template = (
    "Please rewrite the following complex sentence in order to make it easier to understand "
    "by people who need assistive technology for reading. These people may have dyslexia or autism, "
    "etc. You can do so by replacing complex words with simpler synonyms"
    "(i.e. paraphrasing), deleting unimportant information (i.e. compression), and/or splitting a long complex "
    "sentence into several simpler ones. The final simplified sentence needs to be grammatical, fluent, and retain "
    "the main ideas of its original counterpart without altering its meaning.\n\n"
    f"{few_shot_examples}"
    "Complex: {complex_sentence}\nSimple:"
)


def generate_simplification_batch(complex_sentences, seed):
    # Prepare the input prompts
    input_prompts = [prompt_template.format(complex_sentence=cs) for cs in complex_sentences]
    # Tokenize the inputs
    inputs = tokenizer(input_prompts, return_tensors="pt", max_length=512, truncation=True, padding=True).to(device)
    # Set the seed for reproducibility
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Generate the outputs
    with torch.no_grad():  # Disable gradient calculation
        outputs = model.generate(
            inputs["input_ids"],
            max_length=100,
            do_sample=True,
            top_p=0.9,
            temperature=1.0,
            num_return_sequences=1
        )
    # Decode the outputs
    decoded_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    return decoded_outputs


# Prepare data for metric calculation
results = []
batch_size = 8  # Adjust batch size based on memory constraints
seeds = [42, 43, 44]

for i in range(0, len(test_dataset['complex']), batch_size):
    batch = {'complex': test_dataset['complex'][i:i + batch_size], 'simple': test_dataset['simple'][i:i + batch_size]}
    complex_sentences = batch['complex']
    batch_predictions = generate_simplification_batch(complex_sentences, seeds[0])

    for j in range(len(complex_sentences)):
        # Extract predictions for this complex sentence
        prediction = batch_predictions[j]
        refs = batch['simple'][j]

        results.append({
            "complex": complex_sentences[j],
            "prediction": prediction,
            "simple": refs
        })

# Save the results for qualitative check
with open("output_data_flan_t5_base.json", "w") as outfile:
    json.dump(results, outfile, indent=4)
