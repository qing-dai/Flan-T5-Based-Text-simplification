import pickle
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer, Trainer, TrainingArguments, DataCollatorForSeq2Seq
from datasets import DatasetDict

# Load datasets from disk
with open('validation_dataset.pkl', 'rb') as f:
    validation_dataset = pickle.load(f)

with open('test_dataset.pkl', 'rb') as f:
    test_dataset = pickle.load(f)

# Load the Flan-T5-base model and tokenizer
model_name = "google/flan-t5-base"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

# Check if CUDA (GPU) is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Define the prompt template
prompt_template = (
    "Please rewrite the following complex sentence in order to make it easier to understand "
    "by non-native speakers of English. You can do so by replacing complex words with simpler synonyms "
    "(i.e. paraphrasing), deleting unimportant information (i.e. compression), and/or splitting a long complex "
    "sentence into several simpler ones. The final simplified sentence needs to be grammatical, fluent, and retain "
    "the main ideas of its original counterpart without altering its meaning.\n\n"
    "Complex: {complex_sentence}\nSimple:"
)

# Tokenize the dataset
def preprocess_function(examples):
    inputs = [prompt_template.format(complex_sentence=cs) for cs in examples["complex"]]
    targets = [simplified[0] for simplified in examples["simple"]]
    model_inputs = tokenizer(inputs, max_length=100, truncation=True, padding="max_length")
    labels = tokenizer(targets, max_length=120, truncation=True, padding="max_length")
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# Apply preprocess function to the validation data
tokenized_validation_dataset = validation_dataset.map(preprocess_function, batched=True)

# Split the dataset into 80% training and 20% evaluation
split_dataset = tokenized_validation_dataset.train_test_split(test_size=0.2)
train_dataset = split_dataset['train']
eval_dataset = split_dataset['test']

# Data collator to handle padding and batching
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

# Set up training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",  # Enable evaluation at the end of each epoch
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
)

# Fine-tune the model with evaluation dataset
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,  # Providing evaluation dataset
    data_collator=data_collator,
)

trainer.train()

# Debug print statement to confirm the save process
print("Training complete. Saving model and tokenizer...")

# Save the model and tokenizer
model.save_pretrained("./result/flan-t5-finetuned")
tokenizer.save_pretrained("./result/flan-t5-finetuned")

