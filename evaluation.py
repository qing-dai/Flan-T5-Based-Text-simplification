import json
from easse.sari import get_corpus_sari_operation_scores
from easse.bertscore import corpus_bertscore

# Load the JSON file
# with open("output_data_flan_t5_base.json", "r") as infile:
#     data = json.load(infile)

#output_data_flan_t5_complica_zero.json
# with open("output_data_flan_t5_complica_zero.json", "r") as infile:
#     data = json.load(infile)

# "output_data_flan_t5_complica.json"
# with open("output_data_flan_t5_complica.json", "r") as infile:
#     data = json.load(infile)

# output_data_flan_t5_base_lexical_zero.json
# with open("output_data_flan_t5_base_lexical_zero.json", "r") as infile:
#     data = json.load(infile)

# output_data_flan_t5_base_lexical_zero_non_native.json
# with open("output_data_flan_t5_base_lexical_zero_non_native.json", "r") as infile:
#     data = json.load(infile)

# output_data_flan_t5_base_lexical_shots.json
# with open("output_data_flan_t5_base_lexical_shots.json", "r") as infile:
#     data = json.load(infile)

# # output_data_flan_t5_base_syntactic_zero.json
with open("output_data_flan_t5_base_syntactic_zero.json", "r") as infile:
    data = json.load(infile)

# output_data_flan_t5_base_syntactic_zero_non_native.json
# with open("output_data_flan_t5_base_syntactic_zero_non_native.json", "r") as infile:
#     data = json.load(infile)

# output_data_flan_t5_base_semantic_shots.json
# with open("output_data_flan_t5_base_semantic_shots.json", "r") as infile:
#     data = json.load(infile)

# output_data_flan_t5_base_semantic_zero.json
# with open("output_data_flan_t5_base_semantic_zero.json", "r") as infile:
#     data = json.load(infile)

# output_data_flan_t5_base_semantic_zero_non_native.json
# with open("output_data_flan_t5_base_semantic_zero_non_native.json", "r") as infile:
#     data = json.load(infile)


# output_data_flan_t5_base_syntactic_shots.json
# with open("output_data_flan_t5_base_syntactic_shots.json", "r") as infile:
#     data = json.load(infile)

# output_data_flan_t5_teacher.json
# with open("output_data_flan_t5_teacher.json", "r") as infile:
#     data = json.load(infile)

# output_data_flan_t5_medium_zero.json
# with open("output_data_flan_t5_medium_zero.json", "r") as infile:
#     data = json.load(infile)

# # output_data_flan_t5_medium.json
# with open("output_data_flan_t5_medium.json", "r") as infile:
#     data = json.load(infile)

# output_data_flan_t5_base_dyslexia_one_shot.json
# with open("output_data_flan_t5_base_dyslexia_one_shot.json", "r") as infile:
#     data = json.load(infile)

#output_data_flan_t5_base_dyslexia_two_shot.json
# with open("output_data_flan_t5_base_dyslexia_two_shot.json", "r") as infile:
#     data = json.load(infile)

# output_data_flan_t5_base_zero.json
#output_data_flan_T5_fine_tune.json
# with open("output_data_flan_T5_fine_tune.json", "r") as infile:
#     data = json.load(infile)

# with open("output_data_flan_t5_base_zero.json", "r") as infile:
#     data = json.load(infile)

# output_data_flan_t5_base_dyslexia.json
# with open("output_data_flan_t5_base_dyslexia.json", "r") as infile:
#     data = json.load(infile)

# output_data_flan_t5_base_dyslexia_zero_shot.json
# with open("output_data_flan_t5_base_dyslexia_zero_shot.json", "r") as infile:
#     data = json.load(infile)

# Function to compute SARI score for a single pair
def compute_sari(complex_sentence, prediction, references):
    formatted_references = [[ref] for ref in references]
    sari_add, sari_keep, sari_del = get_corpus_sari_operation_scores(
        [complex_sentence],
        [prediction],
        formatted_references
    )
    sari_score = (sari_add + sari_keep + sari_del) / 3
    return sari_add, sari_keep, sari_del, sari_score


# Aggregate SARI and BERT scores
total_sari_add = 0
total_sari_keep = 0
total_sari_del = 0
total_sari_score = 0
total_bert_f1 = 0
num_samples = len(data)

# Prepare data for BERTScore calculation
source_sentences = []
predictions = []
references = []

for item in data:
    complex_sentence = item["complex"]
    prediction = item["prediction"]
    refs = item["simple"]

    sari_add, sari_keep, sari_del, sari_score = compute_sari(complex_sentence, prediction, refs)
    total_sari_add += sari_add
    total_sari_keep += sari_keep
    total_sari_del += sari_del
    total_sari_score += sari_score

    source_sentences.append(complex_sentence)
    predictions.append(prediction)
    references.append(refs)

# Ensure references are correctly formatted for BERTScore
formatted_references = list(map(list, zip(*references)))

# Calculate BERTScore
P, R, F1 = corpus_bertscore(sys_sents=predictions, refs_sents=formatted_references)
average_bert_f1 = F1

# Calculate average SARI scores
average_sari_add = total_sari_add / num_samples
average_sari_keep = total_sari_keep / num_samples
average_sari_del = total_sari_del / num_samples
average_sari_score = total_sari_score / num_samples

print(f"Average SARI Add: {average_sari_add:.4f}")
print(f"Average SARI Keep: {average_sari_keep:.4f}")
print(f"Average SARI Delete: {average_sari_del:.4f}")
print(f"Average SARI Score: {average_sari_score:.4f}")
print(f"Average BERTScore F1: {average_bert_f1:.4f}")
