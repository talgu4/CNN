'''

need to restore model with:
    with tf.Session() as sess:
        new_saver = tf.train.import_meta_graph('CNN.ckpt.meta')
        new_saver.restore(sess, tf.train.latest_checkpoint('./'))

        # init restored variables
        tf.global_variables_initializer()
        sess.run(tf.global_variables_initializer())
        predict = sess.run([tf_p], feed_dict={ x: [image], p_keep_conv:1.0, p_keep_hidden:0.5})     print(predict)



'''