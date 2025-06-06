import json

# Paths
pred_file = "medlfqa_kqa_golden_test_persona_ideal_olaph_ccc-pubmed-fda-uniprot-ner-nih-ctgov-elsevier_10_pred.json"
evidence_file = "kqa_golden_test_evidence_factscore.jsonl"
output_file = "factscore_input.jsonl"

# 1. Load evidence file into a dict for fast lookup
evidence_dict = {}
with open(evidence_file, "r") as f:
    for line in f:
        entry = json.loads(line)
        evidence_dict[entry["Question"]] = {
            "topic": entry["Topic"],
            "evidence": entry["Evidence"]
        }

# 2. Read your predictions and write FactScore input
with open(pred_file, "r") as f:
    preds = json.load(f)

with open(output_file, "w") as out:
    for entry in preds:
        question = entry["qa"]["question"]
        answer = entry["prediction_result"]["predicted_answer"]["content"]
        # Lookup topic and evidence
        if question not in evidence_dict:
            print(f"Warning: Question not found in evidence file: {question}")
            continue
        topic = evidence_dict[question]["topic"]
        evidence = evidence_dict[question]["evidence"]
        out.write(json.dumps({
            "input": question,
            "output": answer,
            "topic": topic,
            "evidence": evidence
        }) + "\n")

print(f"Wrote FactScore input to {output_file}") 