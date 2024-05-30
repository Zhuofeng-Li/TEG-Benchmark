import argparse
import os
import csv

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder_path", "-fp", help="The path to the folder containing the log files.")
    parser.add_argument("--file_suffix", "-fs", help="The suffix of the files to be processed.")
    args = parser.parse_args()
    
     # Define folder path and file suffix
    folder_path = args.folder_path
    file_suffix = args.file_suffix

    # Initialize a dictionary to store the max AUC and F1 score for each file
    max_scores = {}

    # Iterate over all files in the folder
    for file_name in os.listdir(folder_path):
        # Ensure the file is a text file
        if file_name.endswith(".txt") and file_suffix in file_name:
            file_path = os.path.join(folder_path, file_name)
            # Initialize the max AUC and F1 score for the current file
            max_auc = 0
            max_f1 = 0
            # Read the content of the file
            with open(file_path, "r") as file:
                lines = file.readlines()
                # Find the max AUC and F1 score in the file
                for line in lines:
                    if "Validation micro AUC:" in line:
                        auc = float(line.split(":")[-1].strip())
                        max_auc = max(max_auc, auc)
                    elif "F1 score" in line:
                        f1 = float(line.split(":")[-1].strip())
                        max_f1 = max(max_f1, f1)
            # Store the max AUC and F1 score for the current file
            max_scores[file_name] = {"Max micro AUC": max_auc, "Max F1 score": max_f1}

    # Define the order to sort the file names, considering only the model name
    order = ["MLP", "GraphSAGE", "GeneralConv", "GINE", "EdgeConv", "GraphTransformer"]

    # Filter and sort the file names based on the defined order, ignoring 'summary.txt'
    filtered_keys = [key for key in max_scores.keys() if key != 'summary.txt']
    sorted_files = sorted(filtered_keys, key=lambda x: order.index(x.split('_')[0]))

    # Define the output CSV file path
    output_csv_path = os.path.join(folder_path, f"{file_suffix}_max_scores_summary.csv")

    # Write the results to the CSV file
    with open(output_csv_path, mode='w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        # Write the header
        writer.writerow(["File", "Max micro AUC", "Max F1 score"])
        # Write the max scores for each file
        for file_name in sorted_files:
            writer.writerow([file_name, max_scores[file_name]["Max micro AUC"], max_scores[file_name]["Max F1 score"]])

    # Print confirmation message
    print(f"Results have been written to {output_csv_path}")
