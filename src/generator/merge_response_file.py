import argparse
import os
import json

def merge_jsonl_files(input_folder, output_file):
    data_list = []
    jsonl_files = [f for f in os.listdir(input_folder) if f.endswith('.jsonl')]
    jsonl_files.sort()  

    for filename in jsonl_files:
            file_path = os.path.join(input_folder, filename)
            with open(file_path, 'r') as infile:
                for line in infile:
                    data = json.loads(line)
                    data['_source_file'] = filename
                    data_list.append(data)


    with open(output_file, 'w', encoding='utf-8') as outfile:
        for entry in data_list:
            json.dump(entry, outfile)
            outfile.write('\n')

    print(f"finish and file is saved in:{output_file}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_data_path', type=str,
                        default=None,
                        help="The folder where the chunk files you want to merge are located.",
                        )
    parser.add_argument('--output_file', type=str,
                        default=None,
                        )

    args = parser.parse_args()
    merge_jsonl_files(args.input_folder, args.output_file)


if __name__ == "__main__":
    main()