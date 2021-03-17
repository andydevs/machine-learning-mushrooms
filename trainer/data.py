"""
Data input pipeline
"""
import tensorflow as tf
import pandas as pd

# Possible label categories
label_categories = ['e', 'p']

def get_data(data_file, batch_size, shuffle_buffer, repeat_num, train_frac, display_data):
    """
    Get data from data file
    """
    # Read dataset. Split features and labels. Convert labels to indeces
    df = pd.read_csv(data_file)
    labels = df.pop('class')
    labels = labels.apply(label_categories.index)
    dataset = tf.data.Dataset.from_tensor_slices((dict(df), labels))

    # One-hot encode labels.
    preprocess_labels = lambda labels: tf.one_hot(labels, depth=len(label_categories))
    dataset = dataset.map(lambda feats, labels: (feats, preprocess_labels(labels)))

    # Shuffle, batch, and repeat dataset
    dataset = dataset.shuffle(shuffle_buffer)
    dataset = dataset.batch(batch_size)
    dataset = dataset.repeat(repeat_num)

    # Display data if display data flag is on
    if display_data:
        for feature_batch, label_batch in dataset.take(1):
            print(f"'class': {label_batch}")
            print('features:')
            for feature, value in feature_batch.items():
                print(f'    {feature:30s}: {value}')

    # Split into training and testing 
    train_num = int(train_frac*len(dataset))
    train_dataset = dataset.take(train_num)
    test_dataset = dataset.skip(train_num)

    # Return dataset
    return train_dataset, test_dataset