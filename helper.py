import os
import pandas as pd
import glob

def convert_pride_csvs():
    input_dir = 'sem1/results/pride_optimized_csv_results'
    output_dir = 'organized/pride_csv'

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Find all CSV files in the input directory
    csv_files = glob.glob(os.path.join(input_dir, '*.csv'))
    
    if not csv_files:
        print(f"No CSV files found in {input_dir}")
        return

    output_columns = [
        'question_id', 'permutation_idx', 'prob_A', 'prob_B', 'prob_C', 'prob_D',
        'predicted_answer', 'correct_position', 'correct_answer', 'model',
        'temperature', 'is_correct_fixed', 'debiased_predicted_answer', 'debiased_is_correct'
    ]

    for file_path in csv_files:
        df = pd.read_csv(file_path)

        # 1. Map 'original_correct' to 'correct_answer'
        if 'original_correct' in df.columns:
            df['correct_answer'] = df['original_correct']
        else:
            df['correct_answer'] = None # Fallback if missing

        # 2. Add missing 'temperature' column 
        # (Change this if you want to extract it from the filename or set a specific default)
        if 'temperature' not in df.columns:
            df['temperature'] = 0.5  

        # Ensure all required columns exist (to prevent KeyErrors)
        for col in output_columns:
            if col not in df.columns:
                df[col] = pd.NA

        # Extract only the targeted columns in the specified order
        df_out = df[output_columns]

        # Save to the new directory
        filename = os.path.basename(file_path)
        out_path = os.path.join(output_dir, filename)
        
        df_out.to_csv(out_path, index=False)
        print(f"Converted and saved: {out_path}")

if __name__ == "__main__":
    convert_pride_csvs()
