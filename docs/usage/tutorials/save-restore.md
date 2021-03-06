# Save and Restore Training

AutoDist savers (wrappers around TF savers) are required for saving.
These wrapped savers will save the **original graph** instead of the transformed distributed graph executed during the training.
Users can then further fine-tune the model without AutoDist or under other distributed settings.

There are 2 AutoDist saving APIs, which are very similar to the native TensorFlow saving interface. One is `saver` another is `SavedModelBuilder`.

## saver

AutoDist provides a `saver`, which is a wrapper of the original Tensorflow Saver. It has the exactly same interface as the one in Tensorflow. **Note that this saver should be created before the `create_distributed_session` function.** Here is an example:

```
from autodist.checkpoint.saver import Saver as autodist_saver
...

# Build your model
model = get_your_model()

# Create the AutoDist Saver
saver = autodist_saver()

# Create the AutoDist session
sess = autodist.create_distributed_session()
for steps in steps_to_train:
    # some training steps
    ...

# Save the training result
saver.save(sess, checkpoint_name, global_step=step)
```

The saved checkpoint can be loaded without AutoDist, just like a normal TensorFlow Model.

```
with tf.compat.v1.Session() as sess:
    # Build your model
    model = get_your_model()

    # Create the saver
    tf_saver = tf.compat.v1.train.Saver()

    # Restore the variables
    tf_saver.restore(sess, tf.train.latest_checkpoint(checkpoint_dir))

# Fine-tuning
for steps in steps_to_train:
    # some training steps
    ...
```

More detailed usage with Keras can be found [here](https://github.com/petuum/autodist/blob/master/tests/checkpoint/test_keras_saver.py).

## SavedModelBuilder

The `SavedModelBuilder` API will not only save the trainable variables, but also some other training metadata, such as the Tensorflow MetaGraph and training operations. Like the `saver`, `SavedModelBuilder` also has the same interface as Tensorflow.
Also like the `saver`, a user still can load the saved model without AutoDist for fine-tuning or serving on a single node.

```
# create the AutoDist Saver
saver = autodist_saver()

# create the AudoDist session
sess = autodist.create_distributed_session()
for steps in steps_to_train:
    # some training steps
    ...

builder = SavedModelBuilder(EXPORT_DIR)
builder.add_meta_graph_and_variables(
    sess=sess,
    tags=[TAG_NAME],
    saver=saver,
    signature_def_map=signature_map)
builder.save()
```

The output of *SavedModelBuilder* is a binary including model weights, model graph, and some other training information.
Just like the `saver`, a user can load the saved model without AutoDist for fine-tuning or serving on a single node.

```
with tf.compat.v1.Session() as sess:
    # Load the model
    loaded = tf.compat.v1.saved_model.loader.load(sess, [TAG_NAME], EXPORT_DIR)

    # Get training operation
    train_op = tf.compat.v1.get_collection(TRAIN_OP_KEY)

    # Retrieve model feed and fetch
    serving_signature = loaded.signature_def["serving_default"]
    input_op_names, input_tensor_names = _get_input_tensor_and_op(
        serving_signature.inputs)
    output_op_names, output_tensor_names = _get_output_tensor_and_op(
        serving_signature.outputs)
    input_table = dict(zip(input_op_names, input_tensor_names))
    output_table = dict(zip(output_op_names, output_tensor_names))

    # Fine-tuning
    for _ in range(EPOCHS):
        l, _ = sess.run([output_table["loss"], train_op], feed_dict={input_table["input_data"]:inputs})
        print('loss: {}\n'.format(l))
```

We don't need to build our model in this case, because the model graph is loaded from the serialized data.
More detailed usage, can be found [here](https://github.com/petuum/autodist/blob/master/tests/checkpoint/test_saved_model.py).

## Save model on NFS

AudoDist provides a saving indicator for users who want to save their models on NFS(Network File System) to avoid model writing conflict. This is because some of the NFS systems don't support file locking and the saver in each node might save the model to the same file simultaneously. It works for both `saver` and `SavedModelBuilder` introduced above. Here is an example:

```
from autodist.autodist import IS_AUTODIST_CHIEF

# Some training code
...

if IS_AUTODIST_CHIEF:
    saver.save(session, checkpoint_name, global_step=epoch)
    print('Checkpoint saved at {%s}' % checkpoint_name)
else:
    print("Skip saving on worker nodes.")

```
