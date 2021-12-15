import torch

from cogs.commands.misc.resnext import resnext_classify, preprocess_tilt, resnext_run, preprocess_flip_horizontal, preprocess_random_perspective

def gen_random_preprocess(num_perspective=6, num_tilt=3):
    perspective = [preprocess_random_perspective for i in range(0, num_perspective)]
    tilt = [preprocess_tilt for i in range(0, num_tilt)]
    flip = [preprocess_flip_horizontal]
    return perspective + tilt + flip

# calculation performed using method outlined in
# Classification Confidence Estimation with Test-Time Data-Augmentation
# Yuval Bahat, Gregory Shaknarovich
# June 30, 2020
# arXiv: https://arxiv.org/abs/2006.16705
# Basically, just averaging softmax outputs
def calc_confidence(fname):
    with open("imagenet_classes.txt", "r") as f:
        num_categories = len([s.strip() for s in f.readlines()])

    softmax_avg = torch.zeros(num_categories)
    preprocess = gen_random_preprocess()

    for p in preprocess:
        softmax_avg = softmax_avg.add(resnext_run(fname, preprocess=p))

    return softmax_avg.div(len(preprocess))

def calc_confidence_idx(fname, cat_idx):
    return calc_confidence(fname)[cat_idx].tolist()

def test():
    fname = "testimg/generic.jpeg"
    prediction = resnext_classify(fname)
    (certainty, img_classification) = (calc_confidence_idx(fname, prediction[2][0]), prediction[0][0])
    print(f"image classification: {img_classification}")
    print(f"confidence: {certainty*100:.1f}%")


def run_on_file(contents):
    prediction = resnext_classify(contents)
    (certainty, img_classification) = (calc_confidence_idx(contents, prediction[2][0]), prediction[0][0])
    return (f"image classification: {img_classification}", f"confidence: {certainty*100:.1f}%")


if __name__ == '__main__':
    test()