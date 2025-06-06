#!/usr/bin/env python3
"""
Complete example of how to run FActScore on your predictions.
This script demonstrates the full pipeline.
"""

import json
import os

def convert_and_run_factscore(predictions_file, openai_key_file=None):
    """
    Complete pipeline to convert predictions and run FActScore.
    
    Args:
        predictions_file: Path to your predictions JSON file
        openai_key_file: Path to file containing OpenAI API key (optional for demo)
    """
    
    evidence_file = "kqa_golden_test_evidence_factscore.jsonl"
    
    print("="*60)
    print("FACTSCORE EVALUATION PIPELINE")
    print("="*60)
    
    # Step 1: Load evidence data
    print("Step 1: Loading evidence data...")
    question_to_topic = {}
    question_to_evidence = {}
    
    with open(evidence_file, 'r') as f:
        for line in f:
            data = json.loads(line)
            question = data['Question']
            topic = data['Topic']
            evidence = data['Evidence'][:20]  # Use top 20 evidence passages
            
            question_to_topic[question] = topic
            question_to_evidence[question] = evidence
    
    print(f"Loaded evidence for {len(question_to_topic)} questions")
    
    # Step 2: Load and convert predictions
    print("\nStep 2: Loading and converting predictions...")
    with open(predictions_file, 'r') as f:
        predictions = json.load(f)
    
    factscore_data = []
    missing_questions = []
    
    for i, pred in enumerate(predictions):
        question = pred['qa']['question']
        predicted_answer = pred['prediction_result']['predicted_answer']['content']
        
        if question in question_to_topic:
            topic = question_to_topic[question]
            evidence = question_to_evidence[question]
            
            factscore_entry = {
                "input": question,
                "output": predicted_answer,
                "topic": topic,
                "evidence": evidence
            }
            
            factscore_data.append(factscore_entry)
        else:
            missing_questions.append(question)
    
    print(f"Successfully converted {len(factscore_data)} predictions")
    if missing_questions:
        print(f"Warning: {len(missing_questions)} questions not found in evidence file")
    
    # Step 3: Save converted data
    output_file = predictions_file.replace('.json', '_factscore_input.jsonl')
    with open(output_file, 'w') as f:
        for entry in factscore_data:
            f.write(json.dumps(entry) + '\n')
    
    print(f"Saved FActScore input to: {output_file}")
    
    # Step 4: Show example of converted data
    print("\nStep 3: Example of converted data:")
    print("-" * 40)
    if factscore_data:
        example = factscore_data[0]
        print(f"Input: {example['input'][:100]}...")
        print(f"Output: {example['output'][:100]}...")
        print(f"Topic: {example['topic']}")
        print(f"Evidence passages: {len(example['evidence'])}")
    
    # Step 5: Instructions for running FActScore
    print("\n" + "="*60)
    print("NEXT STEPS: HOW TO RUN FACTSCORE")
    print("="*60)
    
    print("\n1. Install dependencies (if not already done):")
    print("   pip install --upgrade factscore")
    print("   python -m spacy download en_core_web_sm")
    
    print("\n2. Get OpenAI API key:")
    print("   - Go to https://platform.openai.com/api-keys")
    print("   - Create a new API key")
    print("   - Save it to a text file (e.g., 'openai_key.txt')")
    
    print("\n3. Run FActScore:")
    print(f"   python -m factscore.factscorer \\")
    print(f"       --input_path {output_file} \\")
    print(f"       --knowledge_source biomedical \\")
    print(f"       --openai_key openai_key.txt \\")
    print(f"       --verbose")
    
    print("\n4. Alternative: Use the preprocessing approach from OLAPH:")
    print("   python preprocess.py \\")
    print(f"       --input_file {predictions_file} \\")
    print(f"       --output_file {predictions_file.replace('.json', '_preprocessed.jsonl')} \\")
    print("       --top_k 20")
    
    print("\nNote: FActScore evaluation costs approximately $1 per 100 sentences using OpenAI API")
    
    return output_file

def show_sample_prediction_format():
    """Show what your prediction format should look like."""
    
    print("\n" + "="*60)
    print("EXPECTED PREDICTION FORMAT")
    print("="*60)
    
    sample_format = {
        "qa": {
            "id": "0",
            "question": "Your question here",
            "question_type": "long-form-answer",
            "question_topics": None,
            "answers": [
                # These are not used by FActScore, only for other evaluations
                {"content": "Reference answer", "type": "ideal"},
                {"content": ["Must have fact 1", "Must have fact 2"], "type": "must_have"},
                {"content": ["Nice to have fact 1"], "type": "nice_to_have"}
            ]
        },
        "prediction_result": {
            "predicted_answer": {
                "content": "Your model's generated answer here",  # This is what FActScore evaluates
                "type": "ideal"
            },
            "total_tokens": 1000.0,
            "execution_time": 2.5
        }
    }
    
    print("Your JSON file should contain a list of objects like this:")
    print(json.dumps(sample_format, indent=2))
    print("\nKey points:")
    print("- FActScore only uses: qa.question and prediction_result.predicted_answer.content")
    print("- The 'answers' field is not used by FActScore (it's for other evaluations)")
    print("- Make sure your questions match those in kqa_golden_test_evidence_factscore.jsonl")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python example_usage.py <predictions_file> [openai_key_file]")
        print("\nThis script will:")
        print("1. Convert your predictions to FActScore format")
        print("2. Show you how to run FActScore evaluation")
        print("3. Explain the expected format")
        
        show_sample_prediction_format()
        sys.exit(1)
    
    predictions_file = sys.argv[1]
    openai_key_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    if not os.path.exists(predictions_file):
        print(f"Error: File not found: {predictions_file}")
        sys.exit(1)
    
    # Run the conversion
    output_file = convert_and_run_factscore(predictions_file, openai_key_file)
    
    print(f"\n✅ Conversion complete! FActScore input saved to: {output_file}")
    print("Follow the instructions above to run the actual FActScore evaluation.")