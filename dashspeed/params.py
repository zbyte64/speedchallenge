#not a classifier, but -1 to 1 for (normalized) speed
num_classes = 1

# Models to choose from [alexnet, vgg, squeezenet, inception]
model_name = "alexnet"

# Batch size for training (change depending on how much memory you have)
batch_size = 8

# Number of epochs to train for
num_epochs = 15

# Flag for feature extracting. When False, we finetune the whole model,
#   when True we only update the reshaped layer params
feature_extract = True
