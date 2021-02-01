# Vision system counting holes


[![image](https://img.shields.io/badge/license-MIT-green.svg)](https://github.com/HeZhang1994/gif-creator/blob/master/LICENSE)
[![image](https://img.shields.io/badge/python-3.8-blue.svg)]()
[![image](https://img.shields.io/badge/status-stable-brightgreen.svg)]()
[![image](https://img.shields.io/badge/version-1.0.0-informational)]()

This is an implementation of the university project in **Python** that recognizes the objects in an image and counts the holes in each of them.

## Functions

- Specify the directory with images to be processed, its data and output directory with a simple **console argument** parser.

- Returns the **total number of holes** in each object in the image and the **number of all sub-objects** included in it.

- Uses **JSON** files for both input and output files.

## Dependency

* __opencv-contrib-python 4.5.1.48__
* __numpy 1.20.0__
* __setuptools 47.1.0__

## Usage

1. Prepare a set of images with objects consisting of blocks with holes (e.g., `01.jpg`, `02.jpg`, ...) in a folder (e.g., `imgs/`) or at any location on your computer.

#### Sample pictures:
<p><center><img src="imgs\img_001-preview.jpg" width="30%">
<img src="imgs\img_002-preview.jpg" width="30%"></center></p>

2. Prepare a file with information about the number of sub-objects for each larger object in the photo in JSON format.

#### Example:

```json
{
    "img_001": [{
        "red": "0",
        "blue": "2",
        "white": "1",
        "grey": "1",
        "yellow": "1"
    }]
}
```

3. Run the script by passing three arguments:
   + I. The path to the image directory
   + II. The path to the input data file
   + II. Path to the directory where the output file is to be created

#### Example:
```console
python vsch\vsch.py imgs\ data\input.json data\output.json
```


## Results
- **The output file** will contain the number of holes for each object in the image in the order corresponding to that of the input list.

#### Example:
```json
{
    "img_001": [
        35,
        10,
        17,
        10,
        17,
        17,
        10,
        35,
        35,
        29,
        28,
        10
    ]
}
```
