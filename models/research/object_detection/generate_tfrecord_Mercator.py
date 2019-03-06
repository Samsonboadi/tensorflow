"""
Usage:
  # From tensorflow/models/
  # Create train data:
  python generate_tfrecord.py --csv_input=images/train_labels.csv --image_dir=images/train --output_path=train.record

  # Create test data:
  python generate_tfrecord.py --csv_input=images/test_labels.csv  --image_dir=images/test --output_path=test.record
"""
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import io
import pandas as pd
import tensorflow as tf

from PIL import Image
from object_detection.utils import dataset_util
from collections import namedtuple, OrderedDict

flags = tf.app.flags
flags.DEFINE_string('csv_input', '', 'Path to the CSV input')
flags.DEFINE_string('image_dir', '', 'Path to the image directory')
flags.DEFINE_string('output_path', '', 'Path to output TFRecord')
FLAGS = flags.FLAGS


# TO-DO replace this with label map
def class_text_to_int(row_label):
    if row_label == 'Arnold Zijlstra':
        return 1
    if row_label == 'Bart Lissens':
        return 2
    if row_label == 'Van Hoecke':
        return 3
    if row_label == 'Bob Mijwaard':
        return 4
    if row_label == 'Christian Vossers':
        return 5
    if row_label == 'Chirstopher De Bryn':
        return 6
    if row_label == 'Craig Schultz':
        return 7
    if row_label == 'Dave Dionne':
        return 8
    if row_label == 'Didier Bastin':
        return 9
    if row_label == 'Erik van Perre':
        return 10
    if row_label == 'Feike Kramer':
        return 11
    if row_label == 'Fokke Jacobi':
        return 12
    if row_label == 'Frank Schouten':
        return 13
    if row_label == 'Froukje Kuijk':
        return 14
    if row_label == 'Geert De Coensel':
        return 15
    if row_label == 'Isabelle Gooris':
        return 16
    if row_label == 'Isabelle Huisman':
        return 17
    if row_label == 'Jan Stuckens':
        return 18
    if row_label == 'Jasper Roest':
        return 19
    if row_label == 'Jasper Van Nieuland':
        return 20
    if row_label == 'Jelle Bockstal':
        return 21
    if row_label == 'Jeroem De Wilde':
        return 22
    if row_label == 'Jianyun Zhou':
        return 23
    if row_label == 'Johan Peeters':
        return 24
    if row_label == 'Kristof Nevelsteen':
        return 25
    if row_label == 'Luc De Heyn':
        return 26
    if row_label == 'Maarten Vanopstal':
        return 27
    if row_label == 'Mascha van Ringen':
        return 28
    if row_label == 'Maureen van Driel-Rengelink':
        return 29
    if row_label == 'Mike van der Woning':
        return 30
    if row_label == 'Nicholas Bellin':
        return 31
    if row_label == 'Olivier Damanet':
        return 32
    if row_label == 'Olivier Verhamme':
        return 33
    if row_label == 'Pierre Vos':
        return 34
    if row_label == 'Pim van der Kleij':
        return 35
    if row_label == 'Rob Vab Rijen':
        return 36
    if row_label == 'Robin De Nutte':
        return 37
    if row_label == 'Ronnie Lassche':
        return 38
    if row_label == 'Saliba Antar':
        return 39
    if row_label == 'Sam Delefortrie':
        return 40
    if row_label == 'Searp Wijbenga':
        return 41
    if row_label == 'Stephan Deckers':
        return 42
    if row_label == 'Wijnand van Riel':
        return 43
    if row_label == 'Wim Blanken':
        return 44
    if row_label == 'Wouter Sanders':
        return 45
    if row_label == 'Yannick Neuteleers':
        return 46
    if row_label == 'Miguel Hartogs':
        return 47
    else:
        return 0


def split(df, group):
    data = namedtuple('data', ['filename', 'object'])
    gb = df.groupby(group)
    return [data(filename, gb.get_group(x)) for filename, x in zip(gb.groups.keys(), gb.groups)]


def create_tf_example(group, path):
    with tf.gfile.GFile(os.path.join(path, '{}'.format(group.filename)), 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = Image.open(encoded_jpg_io)
    width, height = image.size

    filename = group.filename.encode('utf8')
    image_format = b'jpg'
    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    classes_text = []
    classes = []

    for index, row in group.object.iterrows():
        xmins.append(row['xmin'] / width)
        xmaxs.append(row['xmax'] / width)
        ymins.append(row['ymin'] / height)
        ymaxs.append(row['ymax'] / height)
        classes_text.append(row['class'].encode('utf8'))
        classes.append(class_text_to_int(row['class']))

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
    writer = tf.python_io.TFRecordWriter(FLAGS.output_path)
    path = os.path.join(os.getcwd(), FLAGS.image_dir)
    examples = pd.read_csv(FLAGS.csv_input)
    grouped = split(examples, 'filename')
    for group in grouped:
        tf_example = create_tf_example(group, path)
        writer.write(tf_example.SerializeToString())

    writer.close()
    output_path = os.path.join(os.getcwd(), FLAGS.output_path)
    print('Successfully created the TFRecords: {}'.format(output_path))


if __name__ == '__main__':
    tf.app.run()
