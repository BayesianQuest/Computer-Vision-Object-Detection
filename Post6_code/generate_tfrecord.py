'''

Script for preprocessing the images for the TFODAPI



usage
python generate_tfrecord.py -i /media/acer/7DC832E057A5BDB1/JMJTL/Tomslabs/BayesianQuest/Pothole/TFODAPI/workspace/training_pothole/images/train -a /media/acer/7DC832E057A5BDB1/JMJTL/Tomslabs/BayesianQuest/Pothole/pothole_df.csv -l /media/acer/7DC832E057A5BDB1/JMJTL/Tomslabs/BayesianQuest/Pothole/TFODAPI/workspace/training_pothole/annotations/label_map.pbtxt -o /media/acer/7DC832E057A5BDB1/JMJTL/Tomslabs/BayesianQuest/Pothole/TFODAPI/workspace/training_pothole/annotations/train.record

python generate_tfrecord.py -i /media/acer/7DC832E057A5BDB1/JMJTL/Tomslabs/BayesianQuest/Pothole/TFODAPI/workspace/training_pothole/images/test -a /media/acer/7DC832E057A5BDB1/JMJTL/Tomslabs/BayesianQuest/Pothole/pothole_df.csv -l /media/acer/7DC832E057A5BDB1/JMJTL/Tomslabs/BayesianQuest/Pothole/TFODAPI/workspace/training_pothole/annotations/label_map.pbtxt -o /media/acer/7DC832E057A5BDB1/JMJTL/Tomslabs/BayesianQuest/Pothole/TFODAPI/workspace/training_pothole/annotations/test.record

# Export the model use the following
python exporter_main_v2.py --input_type image_tensor --pipeline_config_path ./models/my_ssd_resnet50_v1_fpn_old/pipeline.config --trained_checkpoint_dir ./models/my_ssd_resnet50_v1_fpn_old/ --output_directory ./exported-models/my_model_ssd

'''

import os
import glob
import pandas as pd
import io
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'    # Suppress TensorFlow logging (1)
import tensorflow.compat.v1 as tf
import argparse
from PIL import Image
from object_detection.utils import dataset_util, label_map_util


# Define the argument parser
arg = argparse.ArgumentParser()
arg.add_argument("-l","--labels-path",help="Path to the labels .pbxtext file",type=str)
arg.add_argument("-o","--output-path",help="Path to the output .tfrecord file",type=str)
arg.add_argument("-i","--image_dir",help="Path to the folder where the input image files are stored. ", type=str, default=None)
arg.add_argument("-a","--anot_file",help="Path to the folder where the annotation file is stored. ", type=str, default=None)

args = arg.parse_args()

# Load the labels files
label_map = label_map_util.load_labelmap(args.labels_path)
label_map_dict = label_map_util.get_label_map_dict(label_map)

# Function to extract information from the images
def create_tf_example(path,annotRecords):
    with tf.gfile.GFile(path, 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = Image.open(encoded_jpg_io)
    width, height = image.size
    # Get the filename from the path
    filename = path.split("/")[-1].encode('utf8')
    image_format = b'jpeg'
    # Get all the lists to store the records
    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    classes_text = []
    classes = []
    # Iterate through the annotation records and collect all the records
    for index, row in annotRecords.iterrows():
        xmins.append(row['xmin'] / width)
        xmaxs.append(row['xmax'] / width)
        ymins.append(row['ymin'] / height)
        ymaxs.append(row['ymax'] / height)
        classes_text.append(row['class'].encode('utf8'))
        classes.append(label_map_dict[row['class']])
    # Store all the examples in the format we want
    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))

    return tf_example


def main(_):

    # Create the writer object
    writer = tf.python_io.TFRecordWriter(args.output_path)
    # Get the annotation file from the arguments
    annotFile = pd.read_csv(args.anot_file)
    # Get the path to the image directory
    path = os.path.join(args.image_dir)
    # Get the list of all files in the image directory
    imgFiles = glob.glob(path + "/*.jpeg")
    # Read each of the file and then extract the details
    for imgFile in imgFiles:
        # Get the file name from the path
        fname = imgFile.split("/")[-1]
        # Get all the records for the filename from the annotation file
        annotRecords = annotFile.loc[annotFile.filename==fname,:]
        tf_example =  create_tf_example(imgFile,annotRecords)
        # Write the records to the required format
        writer.write(tf_example.SerializeToString())
    writer.close()
    print('Successfully created the TFRecord file: {}'.format(args.output_path))
if __name__ == '__main__':
    tf.app.run()


