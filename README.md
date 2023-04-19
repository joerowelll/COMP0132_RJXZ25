# COMP0132 MSc Robotics and Computation Project
## Semantic Validation in Structure From Motion
A project to incorporate semantics into SfM using DeepLab semantic segmentation, for point cloud filtering.
### To clone (including submodules)
```
git clone --recurse-submodules --remote-submodules https://github.com/joerowelll/COMP0132_RJXZ25.git

```
<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li><a href="#about-the-project">About The Project</a>
    <li><a href="#project-structure">Project Structure</a></li>
    <li><a href="#data-sets">Data-sets</a>
    <li><a href="#basic-usage">Basic Usage</a>
    <li><a href="#structure-from-motion-with-colmap">Structure from Motion with COLMAP</a></li>
    <li><a href="#semantic-segmentation-with-deeplab">Semantic Segmentation with DeepLab</a></li>
    <li><a href="#planar-reconstruction">Planar Reconstruction</a></li>
    <li><a href="#database-manipulation">Database Manipulation</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>


### About the Project
[![Project Introduction Video](https://github.com/joerowelll/COMP0132/blob/main/images/thumbnail.jpeg)](https://www.youtube.com/watch?v=hrHsb8gOGck&t=5s)
### Project Structure
```
COMP0132_RJXZ25
├── data
│   └── out90.png ...etc
├── database
│   ├── semantics.py
│   ├── plane_intersection.py
│   ├── visualise_model.py
│   └── readBlobData.py
├── featureDescriptors
│   ├── SURF.m
│   ├── sift_descriptor.asv
│   └── sift_descriptor.m
├── images
├── research/deeplab
├── semanticSegmentation
├── structureFromMotion
├── 3rdParty
│   ├── ceres-solver
│   ├── colmap
│   ├── PlaneRCNN
│   └── DeepLab
├── report
│   ├── latex.tex files
│   └── report.pdf
├── LICENSE.TXT
└── README.md

```


### Data-sets 
Find the focus key data-set Brunswick Square Brighton on [Google Drive](https://drive.google.com/drive/folders/1CNxIw8gyTOldooBWsJqVdKNtusy9eF5a?usp=sharing).
File Structure:

```
COMP0132
├── louvre
│   └── 
├── brighton
│   ├── semantic_labelled_SIFT_keypoints
│   │   ├── labelled3Dpoints.csv
│   │   └── brightonKeypoints.csv
│   ├── segmentedImages
│   │   └── out1.png ...etc
│   ├── segmentation_videos
│   │   └── semantic_segmentation_video.mp4
│   ├── segmentation_video_frames_1
│   │   └── semantic_segmentation0001.png ...etc
│   ├── images
│   │   └── out1.png ...etc
│   ├── colmap_output
│   │   ├── undistorted
│   │   │   ├── stereo
│   │   │   │   ├── normal_maps
│   │   │   │   ├── depth_maps
│   │   │   │   ├── consistency_graphs
│   │   │   │   ├── patch-match.cfg
│   │   │   │   └── fusion.cfg
│   │   │   ├── sparse
│   │   │   │   ├── points3D.bin
│   │   │   │   ├── images.bin
│   │   │   │   └── cameras.bin
│   │   │   ├── images
│   │   │   ├── run-colmap-photometric.sh
│   │   │   └── run-colmap-geometric.sh
│   │   ├── sparse
│   │   │   └── 0
│   │   │       ├── project.ini
│   │   │       ├── points3D.bin
│   │   │       ├── images.bin
│   │   │       ├── cameras.bin
│   │   │       ├── brighton.ply
│   │   │       └──  brighton.nvm
│   │   ├── mpi
│   │   ├── database.db-wal
│   │   ├── database.sb-shm
│   │   ├── database.db
│   │   └── colmap_output.txt
│   ├── segmentation_brighton.zip
│   └── brightonImages.zip


```
### Basic Usage
#### Dependencies:

The code depends on the following third-party libraries:
-Eigen
-Ceres
All of these libraries are added to this repository as submodules, or directly as source files.

#### Preparation
- Clone the repository to your computer including all submodules.
- Build Ceres in folders `3rdParty/build-ceres-solver/`.
- Compile the code using the `CMakeLists.txt` file:
    ```bash
    mkdir build
    cd build
    cmake ..
    make -j4
    cd ..
    ```
- Make conda environment:
```
git clone https://github.com/joerowelll/COMP0132_RJXZ25.git
cd COMP0132_RJXZ25
conda env create --name comp0132 --file=environments.yml
```
#### Semantic Labelling of 3D Points
Example Usage:
```
cd COMP0132/databases
python database.py  --database_path ~/COMP0132/brighton_workspace/database.db
```
##### Planar-Orthogonal Scene Abstraction

Example usage:

```bash
    cd orthogonal-planes/ply_detect_refine # go to scene abstraction directory
    bin/PLY_PPDetectRefine --img test_single.ply # run code
```

### Structure from Motion with COLMAP
[COLMAP](https://colmap.github.io/)

### Semantic Segmentation with DeepLab
Semantic Segmentation on Brighton Data-set Video Demo

[![Semantic Segmentation Demo](https://github.com/joerowelll/COMP0132/blob/main/images/thumbnail2.png)](https://www.youtube.com/watch?v=UwfRyR7IwWU&t=55s)
[DeepLab](https://github.com/tensorflow/models/tree/master/research/deeplab) 

### Planar Reconstruction

### Database Manipulation 
USAGE:
```
$ python semantics.py  --database_path PATH_TO_DATABASE.db
```
Example usage:
```
$ python semantics.py  --database_path ~/COMP0132/brighton_workspace/database.db
```
Reccommended file structure is as follows:
```
COMP0132

```
### Licence

### Contact
mailto:ucabcrr@ucl.ac.uk
### Acknowledgments
Supervisors: Professor Simon Julier, Ziwen Lu \
Planar reconstruction depends on [Detection and Refinement of Orhtogonal Plane Pairs and Derived Orthogonality Primitives](https://github.com/c-sommer/orthogonal-planes)



