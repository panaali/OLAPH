#!/usr/bin/env python3
"""
Script to run FActScore on model predictions.
Converts prediction format to FActScore input format and runs evaluation.
"""

import json
import argparse
import os
import sys
from pathlib import Path

def load_evidence_data(evidence_file):
    """Load the evidence data and create lookup dictionaries."""
    question_to_topic = {}
    question_to_evidence = {}
    
    with open(evidence_file, 'r') as f:
        for line in f:
            data = json.loads(line)
            question = data['Question']
            topic = data['Topic']
            evidence = data['Evidence']
            
            question_to_topic[question] = topic
            question_to_evidence[question] = evidence
    
    return question_to_topic, question_to_evidence

def convert_predictions_to_factscore_format(predictions_file, evidence_file, output_file, top_k=20):
    """Convert prediction format to FActScore input format."""
    
    # Load evidence data
    question_to_topic, question_to_evidence = load_evidence_data(evidence_file)
    
    # Load predictions
    with open(predictions_file, 'r') as f:
        predictions = json.load(f)
    
    # Convert to FActScore format
    factscore_data = []
    
    for pred in predictions:
        question = pred['qa']['question']
        predicted_answer = pred['prediction_result']['predicted_answer']['content']
        
        # Look up topic and evidence for this question
        if question in question_to_topic:
            topic = question_to_topic[question]
            evidence = question_to_evidence[question][:top_k]  # Use top_k evidence
            
            factscore_entry = {
                "input": question,
                "output": predicted_answer,
                "topic": topic,
                "evidence": evidence
            }
            
            factscore_data.append(factscore_entry)
        else:
            print(f"Warning: Question not found in evidence file: {question}")
    
    # Save converted data
    with open(output_file, 'w') as f:
        for entry in factscore_data:
            f.write(json.dumps(entry) + '\n')
    
    print(f"Converted {len(factscore_data)} predictions to FActScore format")
    print(f"Saved to: {output_file}")
    
    return output_file

def run_factscore(factscore_input_file, model_name="retrieval+ChatGPT", 
                  knowledge_source="biomedical", openai_key="api.key", 
                  data_dir=".cache/factscore", cache_dir=".cache/factscore"):
    """Run FActScore evaluation."""
    
    # Import FActScore (do this here to avoid import errors if dependencies aren't installed)
    try:
        from factscore.factscorer import FactScorer
    except ImportError as e:
        print(f"Error importing FActScore: {e}")
        print("Please install required dependencies:")
        print("pip install --upgrade factscore")
        print("python -m spacy download en_core_web_sm")
        return None
    
    # Initialize FactScorer
    fs = FactScorer(
        model_name=model_name,
        data_dir=data_dir,
        cache_dir=cache_dir,
        openai_key=openai_key
    )
    
    # Load data
    topics, generations, evidences = [], [], []
    
    with open(factscore_input_file, 'r') as f:
        for line in f:
            dp = json.loads(line)
            topics.append(dp["topic"])
            evidences.append(dp["evidence"])
            generations.append(dp["output"])
    
    print(f"Running FActScore on {len(generations)} predictions...")
    print(f"Model: {model_name}")
    print(f"Knowledge source: {knowledge_source}")
    
    # Run FActScore
    try:
        out = fs.get_score(
            topics=topics,
            evidences=evidences,
            generations=generations,
            knowledge_source=knowledge_source,
            verbose=True
        )
        
        # Print results
        print("\n" + "="*50)
        print("FACTSCORE RESULTS")
        print("="*50)
        print(f"FActScore = {100*out['score']:.1f}%")
        if "init_score" in out:
            print(f"FActScore w/o length penalty = {100*out['init_score']:.1f}%")
        print(f"Respond ratio = {100*out['respond_ratio']:.1f}%")
        print(f"# Atomic facts per valid response = {out['num_facts_per_response']:.1f}")
        
        # Save results
        output_file = factscore_input_file.replace(".jsonl", "_factscore_results.json")
        with open(output_file, 'w') as f:
            json.dump(out, f, indent=2)
        
        print(f"\nDetailed results saved to: {output_file}")
        
        return out
        
    except Exception as e:
        print(f"Error running FActScore: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description="Run FActScore on model predictions")
    
    parser.add_argument("--predictions_file", type=str, required=True,
                        help="Path to your predictions JSON file")
    parser.add_argument("--evidence_file", type=str, 
                        default="kqa_golden_test_evidence_factscore.jsonl",
                        help="Path to evidence JSONL file")
    parser.add_argument("--output_dir", type=str, default="./factscore_outputs",
                        help="Directory to save outputs")
    parser.add_argument("--top_k", type=int, default=20,
                        help="Number of top evidence passages to use")
    parser.add_argument("--model_name", type=str, default="retrieval+ChatGPT",
                        choices=["retrieval+ChatGPT", "retrieval+llama", "retrieval+ChatGPT+npm"],
                        help="FActScore model to use")
    parser.add_argument("--knowledge_source", type=str, default="biomedical",
                        help="Knowledge source (biomedical or enwiki-20230401)")
    parser.add_argument("--openai_key", type=str, default="api.key",
                        help="Path to OpenAI API key file")
    parser.add_argument("--data_dir", type=str, default=".cache/factscore",
                        help="Data directory for FActScore")
    parser.add_argument("--cache_dir", type=str, default=".cache/factscore",
                        help="Cache directory for FActScore")
    parser.add_argument("--convert_only", action="store_true",
                        help="Only convert format, don't run FActScore")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Generate output filename
    pred_filename = Path(args.predictions_file).stem
    factscore_input_file = os.path.join(args.output_dir, f"{pred_filename}_factscore_input.jsonl")
    
    # Step 1: Convert predictions to FActScore format
    print("Step 1: Converting predictions to FActScore format...")
    convert_predictions_to_factscore_format(
        args.predictions_file, 
        args.evidence_file, 
        factscore_input_file,
        args.top_k
    )
    
    if args.convert_only:
        print("Conversion complete. Use --convert_only=False to run FActScore.")
        return
    
    # Step 2: Run FActScore
    print("\nStep 2: Running FActScore evaluation...")
    results = run_factscore(
        factscore_input_file,
        model_name=args.model_name,
        knowledge_source=args.knowledge_source,
        openai_key=args.openai_key,
        data_dir=args.data_dir,
        cache_dir=args.cache_dir
    )
    
    if results:
        print("\nFActScore evaluation completed successfully!")
    else:
        print("\nFActScore evaluation failed. Check error messages above.")

if __name__ == "__main__":
    main()