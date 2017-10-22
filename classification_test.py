#!/usr/bin/python3

from classification import train_classifier, classify
from numpy import zeros
from os.path import abspath, join
from sys import argv, exit
from keras.models import load_model

if len(argv) != 3:
    print('Usage: %s train_dir test_dir' % argv[0])
    exit(0)

train_dir = argv[1]
test_dir = argv[2]

def read_csv(filename):
    res = {}
    with open(filename) as fhandle:
        next(fhandle)
        for line in fhandle:
            parts = line.rstrip('\n').split(',')
            res[parts[0]] = int(parts[1])
    return res

def compute_accuracy(classified, gt):

    correct = 0
    total = len(classified)
    for filename, class_id in classified.items():
        if class_id == gt[filename]:
            correct += 1
    print (correct, total)
    return correct / total

train_gt = read_csv(join(train_dir, 'gt.csv'))
train_img_dir = join(train_dir, 'images')

train_classifier(train_gt, train_img_dir, fast_train=True)

#model = train_classifier(train_gt, train_img_dir)
#model.save('birds_model.hdf5')
model = load_model('best/birds_model.hdf5')
test_img_dir = join(test_dir, 'img_test')
img_classes = classify(model, test_img_dir)

test_gt = read_csv(join(test_dir, 'gt.csv'))
acc = compute_accuracy(img_classes, test_gt)
print('Accuracy: ', acc)
