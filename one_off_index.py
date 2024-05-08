from Bio import SeqIO
import csv

def load_protein_sequence(fasta_file):
    for record in SeqIO.parse(fasta_file, "fasta"):
        return str(record.seq)

def load_negatively_selected_positions(csv_file):
    site_column_name = "Site"
    positions = []
    with open(csv_file, 'r', encoding='utf-8-sig') as file:
        csv_reader = csv.DictReader(file)
        for row in csv_reader:
            position = int(row[site_column_name]) - 1
            positions.append(position)
    return positions

def get_matches(start_position, end_position, target_positions, mer_size):
    matches = []
    # Adjust the search window based on mer_size
    for amino_acid_position in range(start_position, end_position - mer_size + 1):
        # Adjust matching_positions to consider the dynamic mer_size
        matching_positions = [pos for pos in range(amino_acid_position, amino_acid_position + mer_size) if pos in target_positions]
        if len(matching_positions) >= 3:
            # Report positions using 1-based indexing
            matches.append((amino_acid_position + 1, [pos + 1 for pos in matching_positions]))
    return matches

def main():
    protein_sequence = load_protein_sequence("final_omicron_spike.fasta")
    negatively_selected_positions = load_negatively_selected_positions("newsites3.19.csv")

    
    for mer_size in range(8, 13):  
        output_file_path = f"{mer_size}mersOutfile.csv"
        with open(output_file_path, 'w', newline='') as outfile:
            writer = csv.writer(outfile)
            writer.writerow(["Start Position", "Epitope", "Matches"])  
            for start_position in range(len(protein_sequence) - mer_size + 1): 
                end_position = start_position + mer_size
                matches = get_matches(start_position, end_position, negatively_selected_positions, mer_size)
                if matches:
                    epitope_sequence = protein_sequence[start_position:end_position]
                    writer.writerow([matches[0][0], epitope_sequence, matches])  # Write to CSV using the first match's start position
                    print(f"Mer Size: {mer_size}, Writing: Start Position: {matches[0][0]}, Epitope: {epitope_sequence}, Matches: {matches}")

if __name__ == "__main__":
    main()