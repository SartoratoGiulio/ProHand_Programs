# ProHand_Programs
Collection of programs used for the acquistion of data for the ProHand project

In this repo you'll find:
  - The EPS32 code used to acquire the data
  - The software for acquisition to use with the aforementioned code
  - The matlab programs used to relabel the data
  - Some programs written just to visualized the data incoming from the electrodes

Some of the programs can work also with MyoArmband from ThalmicLabs, but only on Linux. With Windows they tend to get stuck.
Since the codes are very similar in structure and there was a lot of copy and paste between them, it should be difficult to add the MyoArmband functionality to the other files as well.

## Visualization files
 Radial Graph | Round Plot
 :---------:|:----------:
![Radial Graph](https://github.com/SartoratoGiulio/ProHand_Programs/blob/main/readme_img/radial_graph.gif) | ![Round Plot](https://github.com/SartoratoGiulio/ProHand_Programs/blob/main/readme_img/round_graph.gif)

## Dependencies
 - pygame
 - pygame_widgets
 - multiprocessing
 - numpy
 - pandas
 - pyomyo
 - pyserial
