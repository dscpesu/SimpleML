# Augmentation
## Overview

This script balances an image dataset with data augmentation and saves it to a specified output folder. It can be used to address the problem of imbalanced datasets in machine learning, where some classes are represented by many more samples than others.

## Usage

To use the script, simply run it with the following command:

```python
python balance_dataset.py --input_folder <input_folder> --output_folder <output_folder> --min_threshold <min_threshold> --max_threshold <max_threshold> --plot_file <plot_file>
```
The following arguments are required:

* `input_folder`: The path to the input folder containing the imbalanced dataset.
* `output_folder`: The path to the output folder where the balanced dataset will be saved.

The following arguments are optional:

* `min_threshold`: The minimum number of samples for each class in the balanced dataset. If a class has fewer than this number of samples, the script will use data augmentation to generate additional samples.
* `max_threshold`: The maximum number of samples for each class in the balanced dataset. If a class has more than this number of samples, the script will randomly select a subset of samples to include in the balanced dataset.
* `plot_file`: The path to a file where the script will save a frequency bar plot of the class counts in the input dataset.

## Example

To balance an image dataset with a minimum threshold of 100 samples per class and a maximum threshold of 200 samples per class, and save the balanced dataset to the folder `balanced_dataset`, the following command would be used:

python
python balance_dataset.py --input_folder dataset/asl_dataset --output_folder balanced_dataset --min_threshold 100 --max_threshold 200


## Output

The script will save the balanced dataset to the specified output folder. The balanced dataset will be organized in a subfolder for each class, with each image saved as a PNG file.

If the `plot_file` argument is specified, the script will also save a frequency bar plot of the class counts in the input dataset to the specified file.

## Data augmentation

The script uses the following data augmentation techniques to generate additional samples for underrepresented classes:

* Random rotation (40 degrees)
* Random horizontal flip
* Random vertical flip
* Color jitter (brightness, contrast, saturation, and hue)

## Conclusion

This script can be used to balance an image dataset with data augmentation, which can improve the performance of machine learning models trained on the dataset.


To view the Markdown syntaxes, you can open the README.md file in a text editor or in a Markdown previewer.
