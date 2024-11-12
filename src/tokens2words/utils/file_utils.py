import os
import re


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


def parse_string_list_from_file(file_path, delimiter=None):
    """
    Parses a list of strings from a file, handling various list formats.

    Args:
        file_path (str): Path to the file containing the list.

    Returns:
        list: A list of parsed strings.
    """
    with open(file_path, 'r') as file:
        content = file.read()

    if delimiter is None:
        # Remove newlines and excess whitespace
        content = re.sub(r'\s+', ' ', content.strip())

        # Handle different delimiters and list formats
        # Removes common list notations like commas, brackets, quotes, etc.
        items = re.split(r'[,\[\]\(\)\{\}"\'\s]+', content)
    else:
        if delimiter == "newline":  # TODO fix this
            delimiter = "\n"
        items = [item.strip() for item in content.split(delimiter)]

    # Filter out any empty strings from the list
    return [item for item in items if item]