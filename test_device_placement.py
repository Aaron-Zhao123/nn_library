# Creates a graph.
a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
c = tf.matmul(a, b)

w = tf.get_variable('w')
with tf.Graph().as_default(), tf.device('/cpu:0'):
    for i in xrange(2):
        with tf.device('/gpu:%d' % i):
            with tf.name_scope('%s_%d' % ('tower', i)) as scope:
                a_shape = [6]
                a_init = tf.constant_initializer(a, dtype=tf.float32)
                a_tf = tf.get_variable('a', a_shape, initializer = a_init)
