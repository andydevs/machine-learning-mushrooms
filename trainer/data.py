"""
Data input pipeline
"""

def get_data(batch_size, shuffle_buffer, repeat_num, train_frac, display_data):
    """
    Get data from data file
    """
    # Read dataset. Split features and labels
    df = pd.read_csv(DATA_FILE)
    labels = df.pop('class')
    dataset = tf.data.Dataset.from_tensor_slices((dict(df), labels))

    # Preprocess labels
    def preprocess_label(features, labels):
        """
        Convert string-categorical labels into
        binary one-hot representation
        """
        true_label = tf.constant('p')
        labels = tf.math.equal(labels, true_label)
        labels = tf.cast(labels, tf.uint8)
        labels = tf.one_hot(labels, depth=2)
        return features, labels
    dataset = dataset.map(preprocess_label)

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