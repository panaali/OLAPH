# How to Run FActScore on Your Model Predictions

This guide shows you how to evaluate your trained LLM using FActScore with the biomedical evidence from the OLAPH dataset.

## Quick Start

### Option 1: Simple Conversion (Recommended)

```bash
# Convert your predictions to FActScore format
python simple_factscore_runner.py your_predictions.json

# Then run FActScore
python -m factscore.factscorer \
    --input_path your_predictions_factscore_input.jsonl \
    --knowledge_source biomedical \
    --openai_key your_openai_key.txt \
    --verbose
```

### Option 2: Complete Pipeline with Examples

```bash
# See detailed instructions and examples
python example_usage.py your_predictions.json
```

### Option 3: Full-Featured Script

```bash
# Convert and run in one go (requires dependencies)
python run_factscore_on_predictions.py \
    --predictions_file your_predictions.json \
    --model_name retrieval+ChatGPT \
    --knowledge_source biomedical \
    --openai_key your_openai_key.txt
```

## Prerequisites

1. **Install FActScore dependencies:**
   ```bash
   pip install --upgrade factscore
   python -m spacy download en_core_web_sm
   ```

2. **Get OpenAI API Key:**
   - Go to https://platform.openai.com/api-keys
   - Create a new API key
   - Save it to a text file (e.g., `openai_key.txt`)

3. **Your prediction file format should be:**
   ```json
   [
       {
           "qa": {
               "question": "Your question here"
           },
           "prediction_result": {
               "predicted_answer": {
                   "content": "Your model's answer here"
               }
           }
       }
   ]
   ```

## What FActScore Does

1. **Breaks down** your generated answer into atomic facts
2. **Retrieves** relevant evidence from biomedical literature
3. **Checks** each fact against the evidence using ChatGPT/LLaMA
4. **Computes** the percentage of supported facts

## Important Notes

- **Cost**: ~$1 per 100 sentences using OpenAI API
- **Evidence**: Uses top 20 passages from the 100 biomedical evidence passages
- **Knowledge Source**: Use `biomedical` for medical questions (not Wikipedia)
- **Questions**: Must match those in `kqa_golden_test_evidence_factscore.jsonl`

## Output

FActScore will give you:
- **FActScore**: Percentage of factually correct atomic facts
- **Respond ratio**: Percentage of non-abstained responses  
- **Atomic facts per response**: Average number of facts generated

## Files Provided

1. **`simple_factscore_runner.py`** - Quick conversion script
2. **`example_usage.py`** - Detailed example with instructions
3. **`run_factscore_on_predictions.py`** - Full-featured script with all options

## Troubleshooting

**Q: "Question not found in evidence"**
A: Your questions must exactly match those in the evidence file. Check for extra spaces or punctuation.

**Q: "Module not found" errors**
A: Install dependencies: `pip install --upgrade factscore && python -m spacy download en_core_web_sm`

**Q: OpenAI API errors**
A: Check your API key file and ensure you have sufficient credits.

**Q: Want to use different evidence count?**
A: Modify `top_k` parameter (default is 20 out of 100 evidence passages).

## Example Command

```bash
# Full example
python simple_factscore_runner.py raw_pred.json

# Then run FActScore
python -m factscore.factscorer \
    --input_path raw_pred.jsonl \
    --knowledge_source biomedical \
    --openai_key openai_key.txt \
    --verbose
```

This will output your FActScore results showing how factually accurate your model's biomedical answers are!