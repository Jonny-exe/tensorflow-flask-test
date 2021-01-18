from __future__ import absolute_import, division, print_function, unicode_literals
import pandas as pd
import tensorflow as tf

(train_images, train_labels), (
    test_images,
    test_labels,
) = tf.keras.datasets.mnist.load_data()


# CSV_COLUMN_NAMES = ["status", "type", "dead"]
NUMERIC_COLUMNS = ["status", "type"]
CATEGORICAL_COLUMNS = ["color"]
SPECIES = ["notDead", "dead"]
# Lets define some constants to help us later on

train = pd.read_csv(
    "https://raw.githubusercontent.com/Jonny-exe/tensorflow-text/main/src/test1_copy.csv")
test = pd.read_csv(
    "https://raw.githubusercontent.com/Jonny-exe/tensorflow-text/main/src/test1_copy.csv")

train_y = train.pop("dead")
test_y = test.pop("dead")

# train = pd.read_csv(train_path, names=CSV_COLUMN_NAMES, header=0)
# test = pd.read_csv(test_path, names=CSV_COLUMN_NAMES, header=0)
# Here we use keras (a module inside of TensorFlow) to grab our datasets and read them into a pandas dataframe

print("Train:", train)
# train_y = train.pop("dead")
# test_y = test.pop("dead")


def input_fn(features, labels, training=True, batch_size=256):
    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))

    # Shuffle and repeat if you are in training mode.
    if training:
        dataset = dataset.shuffle(1000).repeat()

    return dataset.batch(batch_size)


# Feature columns describe how to use the input.
my_feature_columns = []

# this is for categorical collumns to convert strings into unique ints
for feature_name in CATEGORICAL_COLUMNS:
    vocabulary = train[feature_name].unique()
    print("Vocabulary: ", vocabulary)
    categorical_column = tf.feature_column.categorical_column_with_vocabulary_list(
        key=feature_name, vocabulary_list=vocabulary, dtype=tf.string, default_value=-1, num_oov_buckets=0)
    # These must be wrapped https://stackoverflow.com/questions/48614819/items-of-feature-columns-must-be-a-featurecolumn-given-vocabularylistcategori
    my_feature_columns.append(
        tf.feature_column.indicator_column(categorical_column))


for feature_name in NUMERIC_COLUMNS:
    my_feature_columns.append(tf.feature_column.numeric_column(
        key=feature_name, dtype=tf.int8))

print("my feature columns: ", my_feature_columns)


classifier = tf.estimator.DNNClassifier(
    feature_columns=my_feature_columns,
    # Two hidden layers of 30 and 10 nodes respectively.
    hidden_units=[30, 10],
    # The model must choose between 3 classes.
    n_classes=3,
)


classifier.train(input_fn=lambda: input_fn(
    train, train_y, training=True), steps=5000)
# We include a lambda to avoid creating an inner function previously


def input_fn2(features, batch_size=256):
    # Convert the inputs to a Dataset without labels.
    return tf.data.Dataset.from_tensor_slices(dict(features)).batch(batch_size)


features = ["type", "color", "status"]
predict = {}

print("Please type numeric values as prompted.")
for feature in features:
    valid = True
    val = input(feature + ": ")
    if val.isdigit():
        val = int(val)

    predict[feature] = [val]


def evaluate():
    eval_result = classifier.evaluate(
        input_fn=lambda: input_fn(test, test_y, training=False))
    print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**eval_result))


predictions = classifier.predict(input_fn=lambda: input_fn2(predict))
for pred_dict in predictions:
    class_id = pred_dict["class_ids"][0]
    probability = pred_dict["probabilities"][class_id]
    print("Class id: ", class_id, SPECIES)
    print('Prediction is "{}" ({:.1f}%)'.format(
        SPECIES[class_id], 100 * probability))
    evaluate()
