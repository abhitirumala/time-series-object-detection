# Time Series Object Detection in Python

This repository explores the usage of LSTM and Convolutional Neural Networks to develop a novel object detection algorithm to increase speed and reliability.

This algorithm is trained using the [Core50 Dataset](https://vlomonaco.github.io/core50/) by Vincenzo Lomonaco & Davide Maltoni

## Algorithm Structure and Pseudocode

In a nutshell the current algorithm follows these steps:

1. Read and classify a set of video frames and corresponding bounding boxes to develop a base history for the model

2. For each incoming video frame read:

   1. Create an array containing each bounding box's coordinates from the last 15 frames

   2. Feed this array into the LSTM model and predict the bounding box coordinates for the current image

   3. Generate boxes around the predicted box to account for spontaneous changes

      <div>
         <img src="https://raw.githubusercontent.com/abhitirumala/time-series-object-detection/master/public/one_box.png" alt="one_box" height="400" />
         <img src="https://raw.githubusercontent.com/abhitirumala/time-series-object-detection/master/public/multi_box.png" alt="multiple boxes" height="400" />
      <div>

   4. Run the CNN model and predict the classes for each of the generated bounding boxes

   5. Select the bounding box with the greatest accuracy and append its coordinates to the an array containing each frame's bounding box coordinates

   6. Draw the generated box on the image using OpenCV and add to the list of modified video frame images

3. Write the list of modifies video frames to a new .mp4 file

## Installation and Running Current Model

Git clone the repository to your local drive.

    $ git clone https://github.com/abhitirumala/continual-object-detection.git

Go to the [Core50 Dataset](https://vlomonaco.github.io/core50/) by Vincenzo Lomonaco & Davide Maltoni:

Download [full-size_350x350_images.zip](http://bias.csr.unibo.it/maltoni/download/core50/core50_350x350.zip) and [core50_train.csv](https://vlomonaco.github.io/core50/data/core50_train.csv)

Place the folders in this file structure:

    Computer Vision (Or some folder name)
    |
    |- continual-object-detection (This repository)
    |  |- README.md
    |  |  ...
    |
    |- data
    |  |- core50_350x350
    |  |  |  ...
    |  |
    |  |- core50_train.csv

Install dependencies (Make sure python version is <=3.6)

    $ pip3 install -r requirements.txt

Run the model

    $ python3 run_model.py

Edited video file will save to "test_video_autobox.mp4"

## Attribution

Algorithm - designed and written by Abhiram Tirumala

Training set - [Core50 Dataset](https://vlomonaco.github.io/core50/) by Vincenzo Lomonaco & Davide Maltoni
