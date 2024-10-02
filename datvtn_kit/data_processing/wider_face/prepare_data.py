import os
import json
from pathlib import Path
import numpy as np
from typing import List, Tuple

def process_annotations(input_path: Path, output_path: Path) -> None:
    """Processes the annotation data from a text file and saves it as a JSON file.

    Args:
        input_path (Path): Path to the input file.
        output_path (Path): Path to the output file.
    """
    result = []
    temp_annotation = {}

    # Indices of valid annotation landmarks
    valid_annotation_indices = np.array([0, 1, 3, 4, 6, 7, 9, 10, 12, 13])

    with input_path.open("r") as file:
        for line_id, line in enumerate(file):
            if line.startswith("#"):
                if line_id != 0:
                    result.append(temp_annotation)

                # Start a new annotation block
                temp_annotation = {
                    "file_name": line.replace("#", "").strip(),
                    "annotations": []
                }
            else:
                points = list(map(int, line.strip().split()[:4]))
                x_min, y_min, width, height = points
                x_max = x_min + width
                y_max = y_min + height

                # Ensure minimum bounding box size
                x_min = max(x_min, 0)
                y_min = max(x_min + 1, x_max)
                y_min = max(y_min, 0)
                y_max = max(y_min + 1, y_max)

                # Parse landmarks if available
                landmarks = np.array([float(coord) for coord in line.strip().split()[4:]])
                if landmarks.size > 0:
                    landmarks = landmarks[valid_annotation_indices].reshape(-1, 2).tolist()
                else:
                    landmarks = []

                temp_annotation["annotations"].append({
                    "bbox": [x_min, y_min, x_max, y_max],
                    "landmarks": landmarks
                })

        # Append the last annotation block
        if temp_annotation:
            result.append(temp_annotation)

    # Write the results to the output file
    with output_path.open("w") as file:
        json.dump(result, file, indent=2)


def process_labels_file(txt_path: str, image_dir: str) -> Tuple[List[str], List[List[List[float]]]]:
    """Processes the label text file to extract image paths and their corresponding labels.

    Args:
        txt_path (str): Path to the text file containing labels and image names.

    Returns:
        tuple: A tuple containing two lists:
            - List of image paths.
            - List of lists where each sublist contains labels for each image.
    """
    imgs_path = []
    words = []
    labels = []

    with open(txt_path, 'r') as f:
        lines = f.readlines()

    is_first = True

    for line in lines:
        line = line.rstrip()
        if line.startswith('#'):
            if is_first:
                is_first = False
            else:
                words.append(labels.copy())
                labels.clear()

            # Update image path based on text file reference
            path = line[2:]
            image_path = os.path.join(image_dir, path)
            imgs_path.append(image_path)
        else:
            label = [float(x) for x in line.split(' ')]
            labels.append(label)

    # Append the last set of labels
    words.append(labels)

    return imgs_path, words
