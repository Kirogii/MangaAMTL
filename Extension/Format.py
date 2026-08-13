import os
import argparse

def format_files_to_prompt(directory_path, output_filename="prompt_output.txt"):
    """
    Reads all files in a directory and formats them as: Filename "Content"
    """
    formatted_parts = []
    
    # Get the absolute path for cleaner output
    directory_path = os.path.abspath(directory_path)
    output_file_path = os.path.join(directory_path, output_filename)

    try:
        items = os.listdir(directory_path)
    except FileNotFoundError:
        print(f"Error: The directory '{directory_path}' was not found.")
        return

    for item in items:
        item_path = os.path.join(directory_path, item)
        
        # Process only files, skip directories and the output file itself
        if os.path.isfile(item_path) and item != output_filename:
            try:
                # Try utf-8 first, fallback to latin-1 for weird encodings
                try:
                    with open(item_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                except UnicodeDecodeError:
                    with open(item_path, 'r', encoding='latin-1') as f:
                        content = f.read()

                formatted_parts.append(f'{item} "{content}"')
                
            except Exception as e:
                print(f"Could not read file {item}: {e}")

    final_output = " ".join(formatted_parts)

    try:
        with open(output_file_path, 'w', encoding='utf-8') as out_file:
            out_file.write(final_output)
        print(f"\n[+] Success! Output saved to: {output_file_path}")
    except Exception as e:
        print(f"Error saving output file: {e}")

    print("\n--- Preview ---")
    if len(final_output) > 500:
        print(final_output[:500] + "...")
    else:
        print(final_output)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Format files in a directory into a single prompt string.")
    
    # Changed this line: nargs="?" makes it optional, default="." uses current folder
    parser.add_argument("directory", nargs="?", default=".", help="Path to the directory (defaults to current folder)")
    parser.add_argument("-o", "--output", default="prompt_output.txt", help="Name of the output txt file")
    
    args = parser.parse_args()
    
    format_files_to_prompt(args.directory, args.output)