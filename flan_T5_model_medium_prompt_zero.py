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




# Define the prompt template
prompt_template = (
    "Please rewrite the following complex sentence to make it easier to understand "
    "by non-native speakers of English. Follow these guidelines to ensure a high-quality simplification:\n\n"
    "1. **Replace Difficult Words**: Substitute complex words with simpler synonyms at or below the B2 level in the CEFR. Avoid technical terms, idiomatic expressions, and jargon.\n"
    "2. **Simplify Sentence Structure**: Break long sentences into shorter ones. Each sentence should convey a single idea. Use simple sentence structures and avoid nested clauses.\n"
    "3. **Enhance Clarity**: Use direct and straightforward language. Avoid passive voice and ambiguous references. Ensure the subject and verb are clearly connected.\n"
    "4. **Maintain Coherence**: Ensure the logical flow of the original text is preserved. Use transitional words to link sentences and ideas where necessary.\n"
    "5. **Retain Essential Information**: Keep the main ideas and important details intact. Do not omit information that alters the overall meaning or context.\n"
    "6. **Improve Readability**: Keep sentences concise (10-15 words). Use familiar and common words. Ensure the text is easy to read and understand.\n"
    "7. **Provide Examples**: Where applicable, include examples or analogies to clarify complex ideas.\n\n"
    "Here are some examples of simplified sentences:\n\n"
    "\nComplex: {complex_sentence}\nSimple:"
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
with open("output_data_flan_t5_medium_zero.json", "w") as outfile:
    json.dump(results, outfile, indent=4)


