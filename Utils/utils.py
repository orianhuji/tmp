import os
import pandas as pd

def save_df_to_dir(results_df, base_dir, sub_dirs, file_name_format, add_context, model_name):
    # Get the root directory of the project
    root_dir = os.path.dirname(os.path.abspath(__file__))

    # Construct the output directory path
    output_dir = os.path.join(root_dir, base_dir, *sub_dirs)
    os.makedirs(output_dir, exist_ok=True)

    # Construct the file name
    file_name = file_name_format.format(model_name=model_name,
                                        context="with_context" if add_context else "without_context")

    # Construct the full file path
    file_path = os.path.join(output_dir, file_name)

    # Save the DataFrame to CSV
    results_df.to_csv(file_path, index=False)
