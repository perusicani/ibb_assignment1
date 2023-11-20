import random
from sklearn.model_selection import train_test_split

def split_dataset(txt_file):
    with open(txt_file, 'r') as file:
        lines = file.readlines()

    # Create a dictionary to store identities and corresponding image names
    identity_dict = {}
    for line in lines:
        image_name, identity = line.strip().split()
        if identity not in identity_dict:
            identity_dict[identity] = []
        identity_dict[identity].append(image_name)

    train_set = []
    test_set = []

    # Ensure each identity is represented in both sets
    for identity, image_names in identity_dict.items():
        # Shuffle the image names to randomize the selection
        random.shuffle(image_names)

        # Split the images into training and test sets
        train_images, test_images = train_test_split(image_names, test_size=0.3)

        # Append to the overall sets
        train_set.extend(train_images)
        test_set.extend(test_images)

    return train_set, test_set