# ProHand Programs
Collection of programs used for the data acquistion for the ProHand project

In this repo you'll find:
  - The EPS32 code used to acquire the data
  - The software for acquisition to use with the aforementioned code
  - The matlab programs used to relabel the data
  - Some programs written just to visualized the data incoming from the electrodes

Some of the programs can work also with MyoArmband from ThalmicLabs, but only on Linux. With Windows they tend to get stuck.
Since the codes are very similar in structure and there was a lot of copy and paste between them, it should be difficult to add the MyoArmband functionality to the other files as well.

To fully use these programs, you'll need a custom setup like this one:
<p align="middle">
  <img src="https://github.com/SartoratoGiulio/ProHand_Programs/blob/main/readme_img/setup1.jpg" width=40% height=40%>
  &nbsp; &nbsp;
  <img src="https://github.com/SartoratoGiulio/ProHand_Programs/blob/main/readme_img/setup2.jpg" width=40% height=40%>
</p>

I'll upload the board schematics another time.

## Live visualization programs
 Radial Graph | Round Plot
 :---------:|:----------:
![Radial Graph](https://github.com/SartoratoGiulio/ProHand_Programs/blob/main/readme_img/radial_graph.gif) | ![Round Plot](https://github.com/SartoratoGiulio/ProHand_Programs/blob/main/readme_img/round_graph.gif)

## Acquisition Program
To be used with the code found in the **esp32_code** folder.
The poses of the exercise can be changed by just changing the names in the list. The photos are loaded directly from the **Pose** folder, but they need to have the exact same name of the poses in the exercise list. The photos should all be square.
<p align="center">
  <img src="https://github.com/SartoratoGiulio/ProHand_Programs/blob/main/readme_img/pose_display.PNG" width=50% height=50%>
</p>

## Relabeling programs
In the folder **Relabeling** there are all the matlab files that I wrote for this project. The main file is **emg_relabel.m**. It needs a csv file with this structure:

EMG Channel 1-8| Repetition | Stimulus
:--:|:--:|:--:

After the relabeling process is adviced to check the data and fix some errors, since the algorithm is not 100% accurate.

The other files are mainly leftovers from previous iteration and debugging.

## Training Programs
In this folder there are various programs used to train and test the models in various ways. This was done with Jupyter notebooks for two reasons: Google Colab Compatibility and so that in case a training failed it was possible to restart the program without starting everything again. Most of the programs are kinda useless since the test were meant for my thesis work. I'll add a basic training program that contains just the dataframe manipulation, the models training and the models evaluiation.

## STL  Files
This folder contains the STL files for the wrist ring and the support for the sensors, plus the encorder ring and sensor mount that would be used with the Pro-Hand 3.0 prosthesis. You'll need a lot of 2mmx1mm magnets to fill both the wrist ring and the prosthesis encoder. The magnetic sensors used are the US1881 for the arm support and the US5881 for the prosthesis encoder. Note that for the wrist ring the magnets need to be mounted in with alternating polarities, while for the prosthesis ring they need to be mounted with the south pole facing outward, since the US5881 sensors are unipolar.

## Dependencies
 - pygame
 - pygame_widgets
 - multiprocessing
 - numpy
 - pandas
 - pyomyo
 - pyserial
