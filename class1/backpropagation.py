import tensorflow as tf

w = tf.Variable(tf.constant(5, dtype=tf.float32))  # set random initial value 5, and make it trainable
lr = 0.2    # learning rate
epoch = 40

for epoch in range(epoch):
    with tf.GradientTape() as tape:     # "with expression as variable"
        loss = tf.square(w + 1)

    grads = tape.gradient(loss, w)   # gradient function

    w.assign_sub(lr * grads)    # .assign_sub, self-decrement
    print("After %s epoch, w is %f, loss is %f" % (epoch+1, w.numpy(), loss))