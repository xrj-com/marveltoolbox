import numpy as np 
import collections

def get_attack_data_2labels(dataset, class_num=10, origin_target=0, fake_target=1, length=10, repeat=1):
    """
    Generate the input data to the attack algorithm.

    data: the images to attack
    samples: number of samples to use
    targeted: if true, construct targeted attacks, otherwise untargeted attacks
    start: offset into data to use
    inception: if targeted and inception, randomly sample 100 targets intead of 1000
    """
    images = []
    fake_targets = []
    origin_targets = []
    # if otarget is not None:

    for i in range(len(dataset)):
        if int(dataset[i][1]) == origin_target:
            images.append(dataset[i][0].numpy())
            fake_targets.append(np.eye(class_num)[fake_target])
            origin_targets.append(dataset[i][1])
            length -= 1
            if length == 0:
                images = np.array(images)
                fake_targets = np.array(fake_targets)
                images = np.repeat(images, repeat, 0)
                fake_targets = np.repeat(fake_targets, repeat, 0)
                origin_targets = np.repeat(origin_targets, repeat, 0)
                return images, fake_targets, origin_targets

def get_attack_data_all_labels(dataset, class_num=10, length=1, repeat=1):
    """
    Generate the input data to the attack algorithm.

    data: the images to attack
    samples: number of samples to use
    targeted: if true, construct targeted attacks, otherwise untargeted attacks
    start: offset into data to use
    inception: if targeted and inception, randomly sample 100 targets intead of 1000
    """
    images = []
    fake_targets = []
    origin_targets = []
    # if otarget is not None:
    image_dict = {}.fromkeys(range(10), 0)
    full = 0

    for i in range(len(dataset)):
        if image_dict[int(dataset[i][1])] < length:
            for j in range(10):
                if int(dataset[i][1]) != j:
                    images.append(dataset[i][0].numpy())
                    fake_targets.append(np.eye(class_num)[j])
                    origin_targets.append(dataset[i][1])
            image_dict[int(dataset[i][1])] += 1

    images = np.array(images)
    fake_targets = np.array(fake_targets)
    images = np.repeat(images, repeat, 0)
    fake_targets = np.repeat(fake_targets, repeat, 0)
    origin_targets = np.repeat(origin_targets, repeat, 0)
    return images, fake_targets, origin_targets

def get_attack_data_all_labels(dataset, class_num=10, length=1, repeat=1):
    """
    Generate the input data to the attack algorithm.
    data: the images to attack
    samples: number of samples to use
    targeted: if true, construct targeted attacks, otherwise untargeted attacks
    start: offset into data to use
    inception: if targeted and inception, randomly sample 100 targets intead of 1000
    """
    images = []
    fake_targets = []
    origin_targets = []
    # if otarget is not None:
    image_dict = {}.fromkeys(range(10), 0)
    full = 0

    for i in range(len(dataset)):
        if image_dict[int(dataset[i][1])] < length:
            for j in range(10):
                if int(dataset[i][1]) != j:
                    images.append(dataset[i][0].numpy())
                    fake_targets.append(np.eye(class_num)[j])
                    origin_targets.append(dataset[i][1])
            image_dict[int(dataset[i][1])] += 1

    images = np.array(images)
    fake_targets = np.array(fake_targets)
    images = np.repeat(images, repeat, 0)
    fake_targets = np.repeat(fake_targets, repeat, 0)
    origin_targets = np.repeat(origin_targets, repeat, 0)
    return images, fake_targets, origin_targets

def CW_generate_data(dataset, samples, class_num=10, targeted=True, start=0, inception=False):
    """
    Generate the input data to the attack algorithm.
    data: the images to attack
    samples: number of samples to use
    targeted: if true, construct targeted attacks, otherwise untargeted attacks
    start: offset into data to use
    inception: if targeted and inception, randomly sample 100 targets intead of 1000
    """
    inputs = []
    targets = []
    for i in range(samples):
        if targeted:
            if inception:
                seq = random.sample(range(1,1001), 10)
            else:
                seq = range(class_num)

            for j in seq:
                if (j == dataset[start+i][1]) and (inception == False):
                    continue
                inputs.append(dataset[start+i][0].numpy())
                targets.append(np.eye(class_num)[j])
        else:
            inputs.append(dataset[start+i][0].numpy())
            targets.append(np.eye(class_num)[dataset[start+i][1]])

    inputs = np.array(inputs)
    targets = np.array(targets)

    return inputs, targets


def get_attack_data_for_feature_attack(dataset, class_num=10, length=1):
    """
    Generate the input data to the attack algorithm.

    data: the images to attack
    samples: number of samples to use
    targeted: if true, construct targeted attacks, otherwise untargeted attacks
    start: offset into data to use
    inception: if targeted and inception, randomly sample 100 targets intead of 1000
    """
    images = []
    target_images = []
    fake_targets = []

    image_dict = collections.defaultdict(list)


    per_fake_target = []
    for i in range(len(dataset)):
        label_now = int(dataset[i][1])
        if len(image_dict[label_now]) < length:
            image_dict[label_now].append(dataset[i][0].numpy())
            per_fake_target.append(np.eye(2)[1])


    for j in range(class_num):
        for k in range(class_num):
            if j!=k:
                images += image_dict[j]
                target_images += image_dict[k]
                fake_targets += per_fake_target

    images = np.array(images)
    target_images = np.array(target_images)
    fake_targets = np.array(fake_targets)
    return images, target_images, fake_targets

