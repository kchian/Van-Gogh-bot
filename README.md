# Vincent Van-Gogh-bot

Bridging the gap between post-impressionism and abstract expressionism. Just a bit more abstract than I'd like.

Video:
https://youtu.be/3__Pqb_T0eM

- backbone.py trains the cnn backbone
- rcnn.py trains the rcnn using the backbone
- learn.py learns a dmp from a trajectory
- img_lib.py contains a set of useful transforms when working with lab images
- main.py executes the full inference path (on a lab machine, which has odd python version dependencies out of my control)
- rcnn_main.py executes RCNN inference
- vis.py contains useful visualization functions
- save_trajectory.py performs demonstration recordings

Each shape has its own folder + demo. The `strokes/` folder contains drawings to help train the CNN models. Certain items may be missing from the `notebooks/` directory, and paths may be broken as the repository was refactored for some semblance of cleanliness.