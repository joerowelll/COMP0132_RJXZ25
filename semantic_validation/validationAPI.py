# Copyright 2022 Joseph Rowell

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#API to run semantic valication on colmap sfm model using deeplab images


# The parsed project folder must contain a folder "images" with all the images.
import argparse

class DeepLabModel(object):
  """Class to load deeplab model and run inference."""

  INPUT_TENSOR_NAME = 'ImageTensor:0'
  OUTPUT_TENSOR_NAME = 'SemanticPredictions:0'
  # INPUT_SIZE = 513
  # INPUT_SIZE = 1025
  # INPUT_SIZE = 1242
  INPUT_SIZE = 1226
  FROZEN_GRAPH_NAME = 'frozen_inference_graph'

  def __init__(self, tarball_path):
    """Creates and loads pretrained deeplab model."""
    self.graph = tf.Graph()

    graph_def = None
    # Extract frozen graph from tar archive.
    tar_file = tarfile.open(tarball_path)
    for tar_info in tar_file.getmembers():
      if self.FROZEN_GRAPH_NAME in os.path.basename(tar_info.name):
        file_handle = tar_file.extractfile(tar_info)
        graph_def = tf.GraphDef.FromString(file_handle.read())
        break

    tar_file.close()

    if graph_def is None:
      raise RuntimeError('Cannot find inference graph in tar archive.')

    with self.graph.as_default():
      tf.import_graph_def(graph_def, name='')

    self.sess = tf.Session(graph=self.graph)

  def run(self, image):
    """Runs inference on a single image.

    Args:
      image: A PIL.Image object, raw input image.

    Returns:
      resized_image: RGB image resized from original input image.
      seg_map: Segmentation map of `resized_image`.
    """
    width, height = image.size
    print (image.size)
    resize_ratio = 1.0 * self.INPUT_SIZE / max(width, height)
    print (resize_ratio)
    target_size = (int(resize_ratio * width), int(resize_ratio * height))
    resized_image = image.convert('RGB').resize(target_size, Image.ANTIALIAS)
    batch_seg_map = self.sess.run(
        self.OUTPUT_TENSOR_NAME,
        feed_dict={self.INPUT_TENSOR_NAME: [np.asarray(resized_image)]})
    seg_map = batch_seg_map[0]
    return resized_image, seg_map


def create_pascal_label_colormap():
  """Creates a label colormap used in PASCAL VOC segmentation benchmark.

  Returns:
    A Colormap for visualizing segmentation results.
  """
  colormap = np.zeros((256, 3), dtype=int)
  ind = np.arange(256, dtype=int)

  for shift in reversed(range(8)):
    for channel in range(3):
      colormap[:, channel] |= ((ind >> channel) & 1) << shift
    ind >>= 3

  return colormap

def create_cityscapes_label_colormap():
  """Creates a label colormap used in CITYSCAPES segmentation benchmark.
  Returns:
    A colormap for visualizing segmentation results.
  """
  colormap = np.zeros((256, 3), dtype=np.uint8)
  colormap[0] = [128, 64, 128]
  colormap[1] = [244, 35, 232]
  colormap[2] = [70, 70, 70]
  colormap[3] = [102, 102, 156]
  colormap[4] = [190, 153, 153]
  colormap[5] = [153, 153, 153]
  colormap[6] = [250, 170, 30]
  colormap[7] = [220, 220, 0]
  colormap[8] = [107, 142, 35]
  colormap[9] = [152, 251, 152]
  colormap[10] = [70, 130, 180]
  colormap[11] = [220, 20, 60]
  colormap[12] = [255, 0, 0]
  colormap[13] = [0, 0, 142]
  colormap[14] = [0, 0, 70]
  colormap[15] = [0, 60, 100]
  colormap[16] = [0, 80, 100]
  colormap[17] = [0, 0, 230]
  colormap[18] = [119, 11, 32]
  return colormap

def label_to_color_image(label):
  """Adds color defined by the dataset colormap to the label.

  Args:
    label: A 2D array with integer type, storing the segmentation label.

  Returns:
    result: A 2D array with floating type. The element of the array
      is the color indexed by the corresponding element in the input label
      to the PASCAL color map.

  Raises:
    ValueError: If label is not of rank 2 or its value is larger than color
      map maximum entry.
  """
  if label.ndim != 2:
    raise ValueError('Expect 2-D input label')
  # Change this to relevant pretrained dataset for optimal visualisation
  ############colormap = create_pascal_label_colormap()
  colormap = create_cityscapes_label_colormap()

  if np.max(label) >= len(colormap):
    raise ValueError('label value too large.')

  return colormap[label]

def vis_segmentation(image, seg_map, image_count):

  """Visualizes input image, segmentation map and overlay view."""
  plt.figure(figsize=(15, 5))
  grid_spec = gridspec.GridSpec(1, 4, width_ratios=[6, 6, 6, 1])

  plt.subplot(grid_spec[0])
  plt.imshow(image)
  plt.axis('off')
  plt.title('Input Image')

  plt.subplot(grid_spec[1])
  seg_image = label_to_color_image(seg_map).astype(np.uint8)
  plt.imshow(seg_image)
  plt.axis('off')
  plt.title('Segmentation Map')

  plt.subplot(grid_spec[2])
  plt.imshow(image)
  plt.imshow(seg_image, alpha=0.7)
  plt.axis('off')
  plt.title('Segmentation Overlay')

  unique_labels = np.unique(seg_map)
  ax = plt.subplot(grid_spec[3])
  plt.imshow(
      FULL_COLOR_MAP[unique_labels].astype(np.uint8), interpolation='nearest')
  ax.yaxis.tick_right()
  plt.yticks(range(len(unique_labels)), LABEL_NAMES[unique_labels])
  plt.xticks([], [])
  ax.tick_params(width=0.0)
  plt.grid('off')
  

  
  # Make this save as zero padded frame number, check ffmpeg requirements.
  
  #plt.savefig("/content/drive/MyDrive/COMP0132/" + dataset + "/segmentation_video/" + "semantic_segmentation" + (str(1102-image_count)).zfill(5) + ".png", dpi='figure', format= "PNG")
  plt.savefig("/content/drive/MyDrive/COMP0132/" + dataset + "/segmentation_video/" + "semantic_segmentation" + (str(image_counter)).zfill(5) + ".png", dpi='figure', format= "PNG")
  plt.show()

def convert(img, target_type_min, target_type_max, target_type):
  imin = img.min()
  imax = img.max()

  a = (target_type_max - target_type_min) / (imax - imin)
  b = target_type_max - a * imax
  new_img = (a * img + b).astype(target_type)
  return new_img
  
  
  #Pascal VOC
# LABEL_NAMES = np.asarray([
#     'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
#     'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
#     'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tv'
# ])

#Cityscapes
LABEL_NAMES = np.asarray([
    'unlabeled', 'ego vehicle', 'out of roi', 'static', 'dynamic', 'ground', 'road',
    'sidewalk', 'parking', 'rail track', 'building', 'wall', 'fence', 'guard rail', 'bridge',
    'tunnel', 'pole', 'polegroup', 'traffic light', 'traffic sign', 'vegetation', 'terrain',
     'sky', 'person', 'rider', 'car', 'truck', 'bus', 'caravan', 'trailer', 'train', 
     'motorcycle', 'bicycle', 'license plate'
])
#ADE20K
# LABEL_NAMES = np.asarray([
#     'road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 'traffic light',
#     'traffic sign', 'vegetation',  'terrain',
#      'sky', 'person', 'rider', 'car', 'truck', 'bus', 'train', 
#      'motorcycle', 'bicycle'
# ])

FULL_LABEL_MAP = np.arange(len(LABEL_NAMES)).reshape(len(LABEL_NAMES), 1)
FULL_COLOR_MAP = label_to_color_image(FULL_LABEL_MAP)
def run_colmap_sparse(images_path):
    import subprocess

    subprocess.run([ # to run cli stuff
      "DATASET_PATH={images_path}",
     "colmap feature_extractor \
        --database_path $DATASET_PATH/database.db \
        --image_path $DATASET_PATH/images",

     "colmap exhaustive_matcher \
        --database_path $DATASET_PATH/database.db",

      "mkdir $DATASET_PATH/sparse",

      "colmap mapper \
        --database_path $DATASET_PATH/database.db \
        --image_path $DATASET_PATH/images \
        --output_path $DATASET_PATH/sparse",

    ])

def run_colmap_dense(images_path):
  import subprocess
  subprocess.run[(
  "DATASET_PATH={images_path}",
  "mkdir $DATASET_PATH/dense",

  "colmap image_undistorter \
    --image_path $DATASET_PATH/images \
    --input_path $DATASET_PATH/sparse/0 \
    --output_path $DATASET_PATH/dense \
    --output_type COLMAP \
    --max_image_size 2000",

  "colmap patch_match_stereo \
    --workspace_path $DATASET_PATH/dense \
    --workspace_format COLMAP \
    --PatchMatchStereo.geom_consistency true",

  "colmap stereo_fusion \
    --workspace_path $DATASET_PATH/dense \
    --workspace_format COLMAP \
    --input_type geometric \
    --output_path $DATASET_PATH/dense/fused.ply",

  "colmap poisson_mesher \
    --input_path $DATASET_PATH/dense/fused.ply \
    --output_path $DATASET_PATH/dense/meshed-poisson.ply",

  "colmap delaunay_mesher \
    --input_path $DATASET_PATH/dense \
    --output_path $DATASET_PATH/dense/meshed-delaunay.ply")]


def run_deeplab():
  import os
  from io import BytesIO  
  import tarfile
  import tempfile
  from six.moves import urllib

  from matplotlib import gridspec 
  from matplotlib import pyplot as plt
  import numpy as np
  from PIL import Image
  import globimport cv2
  import natsort

  #tensorflow_version 1.x
  import tensorflow as tf

  # TODO Deeplab API
  _DOWNLOAD_URL_PREFIX = 'http://download.tensorflow.org/models/'
  _MODEL_URLS = {
      'mobilenetv2_coco_voctrainaug':
          'deeplabv3_mnv2_pascal_train_aug_2018_01_29.tar.gz',
      'mobilenetv2_coco_voctrainval':
          'deeplabv3_mnv2_pascal_trainval_2018_01_29.tar.gz',
      'xception_coco_voctrainaug':
          'deeplabv3_pascal_train_aug_2018_01_04.tar.gz',
      'xception_coco_voctrainval':
          'deeplabv3_pascal_trainval_2018_01_04.tar.gz',
          'deeplabv3_cityscapes_train':
          'deeplabv3_cityscapes_train_2018_02_06.tar.gz',
          'xception65_ade20k_train':
          'deeplabv3_xception_ade20k_train_2018_05_29.tar.gz'
  }
  _TARBALL_NAME = 'deeplab_model.tar.gz'
  model_dir = tempfile.mkdtemp()
  tf.gfile.MakeDirs(model_dir)

  download_path = os.path.join(model_dir, _TARBALL_NAME)
  print('downloading model, this might take a while...')
  urllib.request.urlretrieve(_DOWNLOAD_URL_PREFIX + _MODEL_URLS[MODEL_NAME],
                     download_path)
  print('download completed! loading DeepLab model...')

  MODEL = DeepLabModel(download_path) 
  print('model loaded successfully!')
  imageset=glob.glob("/content/drive/MyDrive/COMP0132/"+dataset+"/images/*.png")
#imageset_sorted = sorted(imageset)
imageset_sorted = natsort.natsorted(imageset) #sort images without zero padding in name
image_count = len(imageset_sorted) #decreasing
image_counter = 0                  #increasing
############"$dataset"
# !mkdir /tmp/00/segmentation
# !mkdir /content/drive/MyDrive/"$dataset"/segmentation

import matplotlib.image as mpimg

for img in imageset_sorted:
    cv_img = cv2.imread(img)
    imgu8 = convert(cv_img, 0, 255, np.uint8)
    
    # try:
    # f = urllib.request.urlopen(url)
    # jpeg_str = f.read()
    # original_im = Image.open(BytesIO(jpeg_str))
    # original_im=Image.open('/content/drive/MyDrive/experiment_mono/1520430162157909301_undistort.png')
    # original_im=Image.open('/content/drive/MyDrive/experiment_mono/1520430162157909301.png')
    # original_im=Image.open('/tmp/dataset-outdoors2_1024_16/mav0/cam0/data/1520430162157909301.png')

    # original_im = Image.open('/home/ziwen/Downloads/DeeplabV3/expriment_mono/1520430162157909301.png')
    # pix = np.array(original_im)
    # imgu8 = convert(pix, 0, 255, np.uint8)
    # undistorted = cv2.remap(imgu8, mapx, mapy, cv2.INTER_LINEAR)
    # plt.imshow(original_im)
    # pix = np.array(original_im)
    # plt.imshow(pix)
    # pix = np.array(original_im)

    # plt.imshow(undistorted)
    # original_im=undistorted

    # color_coverted = cv2.cvtColor(original_im, cv2.COLOR_BGR2RGB)
    # original_im=Image.fromarray(color_coverted)
    # imgu8 = convert(pix, 0, 255, np.uint8)
    original_im = Image.fromarray(imgu8, "RGB")
    
    # except IOError:
    # # print('Cannot retrieve image. Please check url: ' + url)
    #   return

    print('running deeplab on image ...' )
    resized_im, seg_map = MODEL.run(original_im)
    # print(seg_map.dtype)
    #seg_img_int8 = (seg_map * 255).astype(np.uint8)
    seg_img_int8 = np.uint8(seg_map)
    seg_im = Image.fromarray(seg_img_int8)
    print(os.path.basename(img))
    # vis_segmentation(resized_im, seg_map)
    # tmp/dataset-outdoors2_1024_16/mav0/cam0/data/
    
    # ## SIFT Feature Overlay (optional)
    # gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    # # create SIFT feature extractor
    # sift = cv2.SIFT_create()
    # # detect features from the image
    # keypoints, descriptors = sift.detectAndCompute(img, None) 
    # # draw the detected key points
    # sift_image = cv2.drawKeypoints(gray, keypoints, img)
    
    #vis_segmentation(img, seg_map, image_counter)
   

    # seg_im.save(('/tmp/'+dataset+'/segmentation/'+os.path.basename(img)),"PNG") try this
    ######################seg_im.save("/content/drive/MyDrive/" + dataset + "/segmentation_video/" + str(1099-image_count),"PNG")
    vis_segmentation(cv_img, seg_map, image_counter)#####################################################
    # try and get the original image displayed, not 8 bit version
    
    # seg_im.save(('/content/drive/MyDrive/tum_vi/segmentation/'+os.path.basename(img)),"PNG")

    # cv2.imwrite(("/content/drive/MyDrive/tum_vi/segmentation/"+os.path.basename(img)), imgu8)
    image_count = image_count - 1 
    image_counter = image_counter + 1
    # print(("./int8_result/"+os.path.basename(img)))
    print("{} {}".format(image_count, "images left"))



  return None

def run_validation():
  return None

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run COLMAP")
    parser.add_argument("--images_path", help="path to colmap input Images file", default = "images")
    parser.add_argument("--segmented_images_path", help="path to deeplab output file", default = "images")
    args = parser.parse_args()
    images_path = args.images_path
    
    run_colmap_sparse(images_path)
    run_colmap_dense(images_path)
    #run_deeplab(images_path)
    #run_validation()
