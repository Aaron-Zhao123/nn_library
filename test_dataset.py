import tensorflow as tf
import preprocessing as ult
from datasets.imagenet_dataset import ImagenetData


FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('subset', 'train',
"""Either 'train' or 'validation'.""")

tf.app.flags.DEFINE_string('data_dir', '/tmp/mydata',
                           """Path to the processed data, i.e. """
                           """TFRecord of Example protos.""")


def distort_image(image, height, width, bbox, thread_id=0, scope=None):
  """Distort one image for training a network.

  Distorting images provides a useful technique for augmenting the data
  set during training in order to make the network invariant to aspects
  of the image that do not effect the label.

  Args:
    image: 3-D float Tensor of image
    height: integer
    width: integer
    bbox: 3-D float Tensor of bounding boxes arranged [1, num_boxes, coords]
      where each coordinate is [0, 1) and the coordinates are arranged
      as [ymin, xmin, ymax, xmax].
    thread_id: integer indicating the preprocessing thread.
    scope: Optional scope for name_scope.
  Returns:
    3-D float Tensor of distorted image used for training.
  """
  with tf.name_scope(values=[image, height, width, bbox], name=scope,
                     default_name='distort_image'):
    # Each bounding box has shape [1, num_boxes, box coords] and
    # the coordinates are ordered [ymin, xmin, ymax, xmax].

    # Display the bounding box in the first thread only.
    if not thread_id:
      image_with_box = tf.image.draw_bounding_boxes(tf.expand_dims(image, 0),
                                                    bbox)
      tf.summary.image('image_with_bounding_boxes', image_with_box)

  # A large fraction of image datasets contain a human-annotated bounding
  # box delineating the region of the image containing the object of interest.
  # We choose to create a new bounding box for the object which is a randomly
  # distorted version of the human-annotated bounding box that obeys an allowed
  # range of aspect ratios, sizes and overlap with the human-annotated
  # bounding box. If no box is supplied, then we assume the bounding box is
  # the entire image.
    sample_distorted_bounding_box = tf.image.sample_distorted_bounding_box(
        tf.shape(image),
        bounding_boxes=bbox,
        min_object_covered=0.1,
        aspect_ratio_range=[0.75, 1.33],
        area_range=[0.05, 1.0],
        max_attempts=100,
        use_image_if_no_bounding_boxes=True)
    bbox_begin, bbox_size, distort_bbox = sample_distorted_bounding_box
    if not thread_id:
      image_with_distorted_box = tf.image.draw_bounding_boxes(
          tf.expand_dims(image, 0), distort_bbox)
      tf.summary.image('images_with_distorted_bounding_box',
                       image_with_distorted_box)

    # Crop the image to the specified bounding box.
    distorted_image = tf.slice(image, bbox_begin, bbox_size)

    # This resizing operation may distort the images because the aspect
    # ratio is not respected. We select a resize method in a round robin
    # fashion based on the thread number.
    # Note that ResizeMethod contains 4 enumerated resizing methods.
    resize_method = thread_id % 4
    distorted_image = tf.image.resize_images(distorted_image, [height, width],
                                             method=resize_method)
    # Restore the shape since the dynamic slice based upon the bbox_size loses
    # the third dimension.
    distorted_image.set_shape([height, width, 3])
    if not thread_id:
      tf.summary.image('cropped_resized_image',
                       tf.expand_dims(distorted_image, 0))

    # Randomly flip the image horizontally.
    distorted_image = tf.image.random_flip_left_right(distorted_image)

    # Randomly distort the colors.
    distorted_image = distort_color(distorted_image, thread_id)

    if not thread_id:
      tf.summary.image('final_distorted_image',
                       tf.expand_dims(distorted_image, 0))
    return distorted_image


def decode_jpeg(image_buffer, scope=None):
  """Decode a JPEG string into one 3-D float image Tensor.

  Args:
    image_buffer: scalar string Tensor.
    scope: Optional scope for name_scope.
  Returns:
    3-D float Tensor with values ranging from [0, 1).
  """
  with tf.name_scope(values=[image_buffer], name=scope,
                     default_name='decode_jpeg'):
    # Decode the string as an RGB JPEG.
    # Note that the resulting image contains an unknown height and width
    # that is set dynamically by decode_jpeg. In other words, the height
    # and width of image is unknown at compile-time.
    image = tf.image.decode_jpeg(image_buffer, channels=3)

    # After this point, all image pixels reside in [0,1)
    # until the very end, when they're rescaled to (-1, 1).  The various
    # adjust_* ops all require this range for dtype float.
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    return image


def parse_example_proto(example_serialized):
  """Parses an Example proto containing a training example of an image.

  The output of the build_image_data.py image preprocessing script is a dataset
  containing serialized Example protocol buffers. Each Example proto contains
  the following fields:

    image/height: 462
    image/width: 581
    image/colorspace: 'RGB'
    image/channels: 3
    image/class/label: 615
    image/class/synset: 'n03623198'
    image/class/text: 'knee pad'
    image/object/bbox/xmin: 0.1
    image/object/bbox/xmax: 0.9
    image/object/bbox/ymin: 0.2
    image/object/bbox/ymax: 0.6
    image/object/bbox/label: 615
    image/format: 'JPEG'
    image/filename: 'ILSVRC2012_val_00041207.JPEG'
    image/encoded: <JPEG encoded string>

  Args:
    example_serialized: scalar Tensor tf.string containing a serialized
      Example protocol buffer.

  Returns:
    image_buffer: Tensor tf.string containing the contents of a JPEG file.
    label: Tensor tf.int32 containing the label.
    bbox: 3-D float Tensor of bounding boxes arranged [1, num_boxes, coords]
      where each coordinate is [0, 1) and the coordinates are arranged as
      [ymin, xmin, ymax, xmax].
    text: Tensor tf.string containing the human-readable label.
  """
  # Dense features in Example proto.
  feature_map = {
      'image/encoded': tf.FixedLenFeature([], dtype=tf.string,
                                          default_value=''),
      'image/class/label': tf.FixedLenFeature([1], dtype=tf.int64,
                                              default_value=-1),
      'image/class/text': tf.FixedLenFeature([], dtype=tf.string,
                                             default_value=''),
  }
  sparse_float32 = tf.VarLenFeature(dtype=tf.float32)
  # Sparse features in Example proto.
  feature_map.update(
      {k: sparse_float32 for k in ['image/object/bbox/xmin',
                                   'image/object/bbox/ymin',
                                   'image/object/bbox/xmax',
                                   'image/object/bbox/ymax']})

  features = tf.parse_single_example(example_serialized, feature_map)
  label = tf.cast(features['image/class/label'], dtype=tf.int32)

  xmin = tf.expand_dims(features['image/object/bbox/xmin'].values, 0)
  ymin = tf.expand_dims(features['image/object/bbox/ymin'].values, 0)
  xmax = tf.expand_dims(features['image/object/bbox/xmax'].values, 0)
  ymax = tf.expand_dims(features['image/object/bbox/ymax'].values, 0)

  # Note that we impose an ordering of (y, x) just to make life difficult.
  bbox = tf.concat(axis=0, values=[ymin, xmin, ymax, xmax])

  # Force the variable number of bounding boxes into the shape
  # [1, num_boxes, coords].
  bbox = tf.expand_dims(bbox, 0)
  bbox = tf.transpose(bbox, [0, 2, 1])

  return features['image/encoded'], label, bbox, features['image/class/text']

# def main():
#     dataset = ImagenetData(subset=FLAGS.subset)
#     data_files = dataset.data_files()
#     images, labels = ult.distorted_inputs(
#         dataset,
#         batch_size=FLAGS.batch_size,
#         num_preprocess_threads=FLAGS.num_preprocess_threads)
#
def traverse_train():
    dataset = ImagenetData(subset=FLAGS.subset)
    data_files = dataset.data_files()

    if data_files is None:
      raise ValueError('No data files found for this dataset')
    else:
      print("there are {} tfRecord files".format(len(data_files)))
    reader = dataset.reader()
    filename_queue = tf.train.string_input_producer(data_files,
                                                    shuffle=True,
                                                    capacity=16)
    # key,value = reader.read(filename_queue)
    _, example_serialized = reader.read(filename_queue)
    image_buffer, label_index, bbox, _ = parse_example_proto(
        example_serialized)
    image = decode_jpeg(image_buffer)

    height = FLAGS.image_size
    width = FLAGS.image_size
    image = distort_image(image, height, width, bbox)
    init = tf.global_variables_initializer()

    sess = tf.Session()
    sess.run(init)

    # Start the queue runners.
    for i in range(1024*10000):
      tf.train.start_queue_runners(sess=sess)
      value_run = sess.run([image])
      print(value_run)


#
#     # Create filename_queue
#     if train:
#         filename_queue = tf.train.string_input_producer(data_files,
#         shuffle=False,
#         capacity=16)
#     else:
#         filename_queue = tf.train.string_input_producer(data_files,
#         shuffle=False,
#         capacity=1)
#
#     if num_preprocess_threads is None:
#         num_preprocess_threads =  4
#         num_readers = 1
#
#     if num_readers > 1:
#         enqueue_ops = []
#         for _ in range(num_readers):
#             reader = dataset.reader()
#             _, value = reader.read(filename_queue)
#             enqueue_ops.append(examples_queue.enqueue([value]))
#
#             tf.train.queue_runner.add_queue_runner(
#             tf.train.queue_runner.QueueRunner(examples_queue, enqueue_ops))
#             example_serialized = examples_queue.dequeue()
#     else:
#         reader = dataset.reader()
#         key, example_serialized = reader.read(filename_queue)
#         print(key)
#
#     images_and_labels = []
#     for thread_id in range(num_preprocess_threads):
#         # Parse a serialized Example proto to extract the image and metadata.
#         image_buffer, label_index, bbox, _ = parse_example_proto(
#         example_serialized)
#         image = image_preprocessing(image_buffer, bbox, train, thread_id)
#         images_and_labels.append([image, label_index])
def main(argv=None):  # pylint: disable=unused-argument
  traverse_train()

if __name__ == '__main__':
  tf.app.run()
