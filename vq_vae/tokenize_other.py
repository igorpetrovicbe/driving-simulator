import h5py

def export_h5_dataset_to_text(h5_file_path, dataset_name, output_text_file):
    with h5py.File(h5_file_path, 'r') as h5_file:
        # Open the dataset
        dataset = h5_file[dataset_name]

        # Open the text file for writing
        with open(output_text_file, 'w') as txt_file:
            # Loop through all the data in the dataset
            for data in dataset:
                # Write each element to the text file, separated by new lines
                txt_file.write(str(data[0]) + '\n')

    print(f"Data from {dataset_name} has been exported to {output_text_file}")

# Example usage
h5_file_path = 'H:\\PycharmProjects\\VQGAN-pytorch-main\driving_images4.h5'
out_folder = 'out_dataset'

with h5py.File(h5_file_path, 'r') as h5_file:
    for i in range(9999):
        dataset_name = f'angle_{i}'
        output_text_file = f'{out_folder}/{i+1}_angle.txt'
        if dataset_name in h5_file:
            export_h5_dataset_to_text(h5_file_path, dataset_name, output_text_file)
        else:
            break

with h5py.File(h5_file_path, 'r') as h5_file:
    for i in range(9999):
        dataset_name = f'speed_{i}'
        output_text_file = f'{out_folder}/{i+1}_speed.txt'
        if dataset_name in h5_file:
            export_h5_dataset_to_text(h5_file_path, dataset_name, output_text_file)
        else:
            break
