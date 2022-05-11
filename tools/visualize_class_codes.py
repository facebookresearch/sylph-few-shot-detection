import torch
from detectron2.utils.file_io import PathManager
from detectron2.data import MetadataCatalog
import numpy as np
from sylph.runner.meta_fcos_runner import MetaFCOSRunner  # noqa

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import random


import os

# scale and move the coordinates so they fit [0; 1] range
def scale_to_01_range(x):
    # compute the distribution range
    value_range = (np.max(x) - np.min(x))

    # move the distribution so that it starts from zero
    # by extracting the minimal value from all its values
    starts_from_zero = x - np.min(x)

    # make the distribution fit [0; 1] by dividing by its range
    return starts_from_zero / value_range

all_synthesized_class_codes = "manifold://fai4ar/tree/liyin/few-shot/meta-fcos/test/20211105173218/e2e_train/inference/default/final/lvis_meta_val_all/0" #20211107130806
save_class_codes_path = os.path.join(all_synthesized_class_codes, "all_codes.pth")
save_class_biases_path = os.path.join(all_synthesized_class_codes, "all_biases.pth")

dataset_name = "lvis_meta_val_all"
classes = MetadataCatalog.get(dataset_name).thing_classes
id_map = MetadataCatalog.get(dataset_name).thing_dataset_id_to_contiguous_id

# this code is only used for one time
class_codes = []
class_biases = []
for code_name in classes:
    code_file = os.path.join(all_synthesized_class_codes, f"{code_name}.pth")
    local_file = PathManager.get_local_path(code_file)
    class_code = torch.load(local_file, map_location=torch.device("cpu"))
    class_codes.append(class_code["class_code"]["cls_conv"])
    class_biases.append(class_code["class_code"]["cls_bias"])

class_codes= torch.cat(class_codes, dim=0).view(len(class_codes), -1)
class_biases= torch.cat(class_biases, dim=0).view(len(class_biases), -1)
with PathManager.open(save_class_codes_path, "wb") as f:
    torch.save(class_codes, f)
with PathManager.open(save_class_biases_path, "wb") as f:
    torch.save(class_biases, f)
# end of code saving
colors = []
for i in range(len(classes)):
    color = (random.random(), random.random(), random.random())
    colors.append(color)
# with PathManager.open(save_class_codes_path, "rb") as f:
#     class_codes = torch.load(f, map_location=torch.device("cpu"))

# normalize class codes
class_codes = torch.nn.functional.normalize(class_codes, p=2, dim=1)


tsne = TSNE(n_components=2).fit_transform(class_codes.numpy())
# extract x and y coordinates representing the positions of the images on T-SNE plot
tx = tsne[:, 0]
ty = tsne[:, 1]

tx = scale_to_01_range(tx)
ty = scale_to_01_range(ty)

# initialize a matplotlib plot
fig = plt.figure()
ax = fig.add_subplot(111)

value = 1
for i, (color, class_name) in enumerate(zip(colors, classes)):
    # extract the coordinates of the points of this class only
    if np.random.randint(3, 1)[0] % 3 != 0:
        continue
    current_tx = np.take(tx, [i])
    current_ty = np.take(ty, [i])
    #color = np.array(color, dtype=np.float) / 255
    ax.scatter(current_tx, current_ty, c=color) #, label=class_name)
    ax.annotate(class_name, (current_tx, current_ty))


# for every class, we'll add a scatter plot separately
# for label in colors_per_class:
#     # find the samples of the current class in the data
#     indices = [i for i, l in enumerate(labels) if l == label]

#     # extract the coordinates of the points of this class only
#     current_tx = np.take(tx, indices)
#     current_ty = np.take(ty, indices)

#     # convert the class color to matplotlib format
#     color = np.array(colors_per_class[label], dtype=np.float) / 255

#     # add a scatter plot with the corresponding color and label
#     ax.scatter(current_tx, current_ty, c=color, label=label)



# build a legend using the labels we set previously
ax.legend(loc='best')

# finally, show the plot
plt.show()
plt.savefig('synthesized_class_codes.png')
