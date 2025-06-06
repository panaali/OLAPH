#!/usr/bin/env python3
"""
Simple script to run FActScore on your model predictions.
Usage: python simple_factscore_runner.py your_predictions.json
"""

import json
import sys
import os

def main():
    if len(sys.argv) != 2:
        print("Usage: python simple_factscore_runner.py <predictions_file>")
        print("Example: python simple_factscore_runner.py raw_pred.json")
        sys.exit(1)
    
    predictions_file = sys.argv[1]
    evidence_file = "kqa_golden_test_evidence_factscore.jsonl"
    
    if not os.path.exists(predictions_file):
        print(f"Error: Predictions file not found: {predictions_file}")
        sys.exit(1)
    
    if not os.path.exists(evidence_file):
        print(f"Error: Evidence file not found: {evidence_file}")
        print("Make sure you're running this script from the Factscore directory")
        sys.exit(1)
    
    print("Loading evidence data...")
    # Load evidence lookup
    question_to_topic = {}
    question_to_evidence = {}
    
    with open(evidence_file, 'r') as f:
        for line in f:
            data = json.loads(line)
            question = data['Question']
            topic = data['Topic']
            evidence = data['Evidence'][:20]  # Use top 20 evidence
            
            question_to_topic[question] = topic
            question_to_evidence[question] = evidence
    
    print("Loading predictions...")
    # Load your predictions
    with open(predictions_file, 'r') as f:
        predictions = json.load(f)
    
    print("Converting to FActScore format...")
    # Convert to FActScore format
    factscore_data = []
    
    for pred in predictions:
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
            print(f"Warning: Question not found in evidence: {question}")
    
    # Save converted data
    output_file = predictions_file.replace('.json', '_factscore_input.jsonl')
    with open(output_file, 'w') as f:
        for entry in factscore_data:
            f.write(json.dumps(entry) + '\n')
    
    print(f"Converted {len(factscore_data)} predictions")
    print(f"Saved FActScore input to: {output_file}")
    print()
    print("Now run FActScore with:")
    print(f"python -m factscore.factscorer --input_path {output_file} --knowledge_source biomedical --openai_key YOUR_API_KEY.txt")
    print()
    print("Note: You'll need:")
    print("1. OpenAI API key in a text file")
    print("2. Install dependencies: pip install --upgrade factscore && python -m spacy download en_core_web_sm")

if __name__ == "__main__":
    main()