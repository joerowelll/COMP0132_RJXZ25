#!/bin/bash

# The project folder must contain a folder "images" with all the images.
COMMON_DATASET_PATH=$PWD/Common

# Check if we have to do the common matching operations

COMMON_DATABASE_FILE=$COMMON_DATASET_PATH/database.db
COMMON_IMAGE_FILE_DIR=$COMMON_DATASET_PATH/images

echo === Common database ===

if [[ -f "$COMMON_DATABASE_FILE" ]]
then
    # File exists; nothing to do
    echo The file $COMMON_DATABASE_FILE exists
    echo Assuming feature extraction / exhaustive matching has been completed and will not run it again.
else
    # File doesn't exist, so process
    echo The file $COMMON_DATABASE_FILE does not exist
    echo Assuming feature extraction / exhaustive matching needs to be carried out.
    
    colmap feature_extractor \
	   --database_path $COMMON_DATABASE_FILE \
	   --image_path $COMMON_IMAGE_FILE_DIR
    
    colmap exhaustive_matcher \
	   --database_path $COMMON_DATABASE_FILE

    # Make read only for safety
    chmod -R a-w $COMMON_DATASET_PATH
fi

echo === Output directory ===

# If a directory was specified, use it. Otherwise generate
# automatically from the current date / time.
if [ $# -eq 0 ]
then
    OUTPUT_DIR=`date +%Y%m%d-%H%M%S`
    echo Creating new directory $OUTPUT_DIR
    mkdir -p $OUTPUT_DIR
else
    OUTPUT_DIR=${1%/}
    if [[ -d "$OUTPUT_DIR" ]]
    then
	echo Using existing directory $OUTPUT_DIR
    else
	echo Creating new directory $OUTPUT_DIR
	mkdir -p $OUTPUT_DIR
    fi
fi

echo === Output directory database ===

# Now generate the output-specific database file; this is obtained
# from the common one by copying.

DATABASE_FILE=$OUTPUT_DIR/database.db

if [[ -f "$DATABASE_FILE" ]]
then
    # File exists; nothing to do
    echo The file $DATABASE_FILE exists
    echo Assuming feature extraction / exhaustive matching has been completed and will not run it again.
else
    # File doesn't exist, so process
    echo The file $DATABASE_FILE does not exist
    echo Assuming feature extraction / exhaustive matching needs to be carried out
    cp $COMMON_DATABASE_FILE $DATABASE_FILE
fi

echo === Sparse reconstruction ===

# Check if we have to compute the sparse solution

SPARSE_DIR=$OUTPUT_DIR/sparse

mkdir -p $SPARSE_DIR

SPARSE_OUTPUT_FILE=$SPARSE_DIR/0/cameras.bin

if [[ -f "$SPARSE_OUTPUT_FILE" ]]
then
    # File exists; nothing to do
    echo The file $SPARSE_OUTPUT_FILE exists
    echo Assuming the sparse solution is available and does not need to be computed again
else
    # File doesn't exist, so process
    echo The file $SPARSE_OUTPUT_FILE does not exist
    echo Assuming the sparse solution is not available and must be computed
    
    colmap mapper \
	   --database_path $DATABASE_FILE \
	   --image_path $COMMON_DATASET_PATH/images \
	   --output_path $SPARSE_DIR
fi

DENSE_DIR=$OUTPUT_DIR/dense

mkdir -p $DENSE_DIR

UNDISTORTED_IMAGES_DIR=$DENSE_DIR/images

echo === Undistorting images ===

# Check if we have to undistort the images; just use a simple check to
# see if the directory exists.

if [[ -d "$UNDISTORTED_IMAGES_DIR" ]]
then
    # Directory exists; nothing to do
    echo The directory $UNDISTORTED_IMAGES_DIR exists
    echo Assuming the images have been undistorted and do not need to be processed.
else
    # Directory doesn't exist, so process
    echo The directory $UNDISTORTED_IMAGES_DIR does not exist
    echo Assuming the images need to be undistorted

    mkdir -p $UNDISTORTED_IMAGES_DIR
    
    echo colmap image_undistorter \
	   --image_path $COMMON_DATASET_PATH/images \
	   --input_path $OUTPUT_PATH/sparse/0 \
	   --output_path $DENSE_DIR \
	   --output_type COLMAP \
	   --max_image_size 2000

    colmap image_undistorter \
	   --image_path $COMMON_DATASET_PATH/images \
	   --input_path $OUTPUT_DIR/sparse/0 \
	   --output_path $DENSE_DIR \
	   --output_type COLMAP \
	   --max_image_size 2000
fi

echo === Stereo patch matching ===

# I'm not entirely sure what the unified output file name is for dense
# reconstruction. At the moment we get the last file out of the images
# folder and hope that works
LAST_FILE=`ls $COMMON_IMAGE_FILE_DIR | sort -V | tail -n 1`

LAST_DEPTH_MAP_FILE=${DENSE_DIR}/stereo/depth_maps/${LAST_FILE}.photometric.bin

if [[ -f "$LAST_DEPTH_MAP_FILE" ]]
then
    # File exists; nothing to do
    echo The file $LAST_DEPTH_MAP_FILE exists
    echo Assuming the stereo patch matching has been completed.
else
    # File doesn't exist, so process
    echo The file $LAST_DEPTH_MAP_FILE does not exist
    echo Assuming the stereo patch matching needs to be carried out
    
    colmap patch_match_stereo \
	   --workspace_path $DENSE_DIR \
	   --workspace_format COLMAP \
	   --PatchMatchStereo.geom_consistency true
fi

echo === Stereo Fusion ===

FUSED_PLY_FILE=$DENSE_DIR/fused.ply

if [[ -f "$FUSED_PLY_FILE" ]]
then
    # File exists; nothing to do
    echo The file $FUSED_PLY_FILE exists
    echo Assuming stereo fusion has been carried out.
else
    # File doesn't exist, so process
    echo The file $FUSED_PLY_FILE does not exist
    echo Assuming stereo fusion needs to be carried out
    
    colmap stereo_fusion \
	   --workspace_path $DENSE_DIR \
	   --workspace_format COLMAP \
	   --input_type geometric \
	   --output_path $FUSED_PLY_FILE
fi

echo === Poisson Mesher ===

MESHED_POISSON_PLY_FILE=$DENSE_DIR/meshed-poisson.ply

if [[ -f "$MESHED_POISSON_PLY_FILE" ]]
then
    # File exists; nothing to do
    echo The file $MESHED_POISSON_PLY_FILE exists
    echo Assuming Poisson meshing has been carried out
else
    # File doesn't exist, so process
    echo The file $MESHED_POISSON_PLY_FILE does not exist
    echo Assuming Poisson meshing needs to be carried out

    colmap poisson_mesher \
	   --input_path  $FUSED_PLY_FILE \
	   --output_path $MESHED_POISSON_PLY_FILE
fi

echo === Delaunay Mesher ===

MESHED_DELAUNAY_PLY_FILE=$DENSE_DIR/meshed-delaunay.ply

if [[ -f "$MESHED_DELAUNAY_PLY_FILE" ]]
then
    # File exists; nothing to do
    echo The file $MESHED_DELAUNAY_PLY_FILE exists
    echo Assuming Delaunay meshing has been carried out
else
    # File doesn't exist, so process
    echo The file $MESHED_DELAUNAY_PLY_FILE does not exist
    echo Assuming Delaunay meshing needs to be carried out

    colmap delaunay_mesher \
	   --input_path $DENSE_DIR \
	   --output_path $MESHED_DELAUNAY_PLY_FILE
fi
