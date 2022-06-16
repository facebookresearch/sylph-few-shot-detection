"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

# fmt: off
# COCO
COCO_NOVEL_CLASSES = [1, 2, 3, 4, 5, 6, 7, 9, 16, 17, 18, 19, 20, 21, 44, 62, 63, 64, 67, 72]

COCO_BASE_CLASSES = [
    8, 10, 11, 13, 14, 15, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35,
    36, 37, 38, 39, 40, 41, 42, 43, 46, 47, 48, 49, 50, 51, 52, 53, 54,
    55, 56, 57, 58, 59, 60, 61, 65, 70, 73, 74, 75, 76, 77, 78, 79, 80,
    81, 82, 84, 85, 86, 87, 88, 89, 90,
]

LVIS_COCO_NOVEL_OVERLAP_SYNSET = ['airplane.n.01',
 'beef.n.01',
 'bicycle.n.01',
 'bird.n.01',
 'boat.n.01',
 'bottle.n.01',
 'bus.n.01',
 'car.n.01',
 'cat.n.01',
 'chair.n.01',
 'dining_table.n.01',
 'dog.n.01',
 'horse.n.01',
 'motorcycle.n.01',
 'person.n.01',
 'pot.n.04',
 'sheep.n.01',
 'sofa.n.01',
 'television_receiver.n.01',
 'train.n.01']

coco_dataset_splits = {}
# LVIS {'c': 461, 'f': 405, 'r': 337})
from detectron2.data.datasets.lvis_v1_categories import LVIS_CATEGORIES as LVIS_V1_CATEGORIES
LVIS_V1_COCO_NOVEL_NO_OVERLAP_CATEGORIES = [item for item in LVIS_V1_CATEGORIES if item['synset'] not in LVIS_COCO_NOVEL_OVERLAP_SYNSET]

"""
Benchmarking Splits:
    TFA: f+c as base and r as novel
    Ours: 305f+361c+237r=903 as base and 100f+100c+100r as novel classes

For demo:
    305f+461c+337r=1103 as base and 100f as novel classes

Available dataset names and corresponding num_classes include:
Pretraining:
    lvis_pretrain_train_basemix #703, lvis_pretrain_val_basemix
    lvis_pretrain_train_basev1 #1103, lvis_pretrain_val_basev1
    lvis_pretrain_train_basev2 #305, lvis_pretrain_val_basev2
Meta-training:
    lvis_meta_train_basemix #703, lvis_meta_val_novelmix #500
    lvis_meta_train_basev1 #1103, lvis_meta_val_novelv1 #100
    lvis_meta_train_basev2 #305, lvis_meta_val_novelv1 #100
    lvis_meta_train_basefc #405+461=766, lvis_meta_val_novelv1 #100

Other splits:
    basev1unknown #1104 classes with novelv1 classes label replaced with unknown
"""
"""
Random shuffle the ids:
with temp_seed(2021):
  np.random.shuffle(LVIS_CATEGORIES)

frequent_ids = [item['id'] for item in LVIS_CATEGORIES if item['frequency'] == 'f']
common_ids = [item['id'] for item in LVIS_CATEGORIES if item['frequency'] == 'c']
rare_ids= [item['id'] for item in LVIS_CATEGORIES if item['frequency'] == 'r']
"""
LVIS_V1_ID_CAT_MAP = {item["id"] : item for item in LVIS_V1_CATEGORIES}

LVIS_FREQUENT_IDS=[1037, 50, 962, 804, 728, 379, 1008, 837, 372, 430, 739, 1108, 160, 385, 556, 1197, 689, 36, 806, 276, 827, 90, 781, 709, 617, 299, 1100, 1083, 848, 1055, 716, 724, 114, 137, 87, 1103, 89, 1043, 477, 898, 75, 12, 559, 676, 112, 1097, 277, 816, 706, 916, 644, 1093, 947, 1198, 838, 1142, 175, 719, 77, 1104, 800, 828, 377, 1074, 923, 441, 297, 194, 104, 565, 1023, 715, 647, 422, 549, 232, 1078, 447, 964, 1136, 305, 1110, 951, 272, 865, 192, 1020, 1042, 330, 924, 734, 502, 285, 2, 80, 29, 653, 133, 993, 395, 621, 1172, 1077, 1188, 1052, 589, 178, 534, 927, 966, 392, 981, 409, 700, 1025, 818, 125, 440, 342, 1026, 685, 41, 217, 642, 229, 514, 1059, 544, 461, 3, 298, 378, 679, 1027, 1079, 1071, 703, 979, 641, 771, 624, 569, 79, 609, 303, 949, 225, 748, 338, 1024, 496, 789, 347, 96, 88, 1060, 521, 1098, 798, 56, 296, 35, 611, 782, 713, 655, 152, 118, 980, 547, 586, 59, 1156, 921, 169, 1202, 1035, 61, 83, 845, 628, 261, 570, 259, 799, 645, 885, 1017, 1121, 982, 692, 208, 4, 1056, 81, 1099, 995, 698, 444, 1186, 53, 832, 745, 687, 15, 836, 1109, 1155, 563, 880, 528, 631, 218, 756, 639, 11, 437, 735, 351, 195, 445, 701, 116, 401, 536, 226, 1095, 254, 253, 1105, 1096, 900, 566, 776, 1018, 344, 910, 1114, 1019, 99, 548, 793, 146, 578, 375, 183, 626, 591, 515, 896, 95, 415, 150, 968, 361, 986, 860, 919, 669, 708, 65, 668, 451, 915, 23, 948, 109, 177, 899, 181, 705, 185, 139, 230, 306, 390, 1061, 68, 1177, 189, 1123, 1122, 961, 204, 138, 127, 904, 1162, 1115, 442, 32, 943, 452, 817, 1178, 615, 811, 271, 694, 171, 726, 66, 143, 658, 369, 953, 48, 43, 469, 1102, 881, 955, 738, 394, 387, 1072, 459, 203, 1134, 429, 704, 510, 1045, 630, 592, 957, 252, 115, 34, 207, 358, 60, 373, 27, 675, 367, 1117, 94, 967, 76, 110, 1033, 965, 1070, 132, 595, 1161, 1021, 959, 1011, 19, 749, 284, 57, 86, 1173, 255, 309, 173, 605, 517, 498, 474, 814, 643, 1112, 911, 1091, 324, 421, 661, 835, 903, 976, 154, 876, 1133, 524, 757, 1064, 751, 1190, 319, 235, 1139, 659, 346, 766, 1000, 540, 500, 404, 912, 627, 1050, 1191, 633, 380, 411, 614, 350, 670, 45, 58, 1141]
LVIS_COMMON_IDS=[660, 946, 576, 1067, 283, 371, 1195, 33, 403, 512, 707, 523, 134, 470, 555, 612, 1068, 656, 158, 558, 765, 472, 28, 334, 826, 909, 1073, 889, 62, 10, 977, 135, 522, 1168, 243, 476, 443, 211, 1160, 216, 188, 861, 329, 884, 841, 423, 1181, 598, 290, 1176, 1081, 554, 637, 1046, 328, 493, 807, 465, 588, 1182, 122, 519, 221, 26, 1006, 468, 320, 922, 872, 54, 1001, 9, 1152, 228, 763, 723, 797, 91, 201, 613, 337, 156, 1196, 7, 1154, 711, 1192, 550, 721, 145, 490, 1170, 933, 750, 433, 339, 791, 1040, 1004, 1143, 893, 530, 790, 857, 495, 833, 460, 997, 198, 180, 989, 386, 1111, 100, 22, 213, 504, 842, 46, 507, 70, 219, 311, 761, 868, 654, 907, 340, 695, 111, 1094, 511, 121, 1085, 293, 418, 844, 1132, 464, 795, 680, 1076, 184, 1013, 450, 871, 206, 505, 537, 288, 839, 466, 667, 774, 1014, 932, 1151, 187, 573, 168, 37, 1039, 453, 67, 107, 1087, 6, 579, 935, 1041, 434, 1179, 220, 973, 531, 47, 483, 485, 875, 1038, 1128, 840, 533, 1101, 1189, 325, 286, 866, 148, 1163, 742, 248, 1125, 717, 584, 363, 1199, 697, 1092, 760, 239, 124, 813, 74, 1066, 102, 773, 341, 770, 867, 1171, 345, 581, 650, 928, 44, 163, 1, 268, 242, 494, 462, 720, 777, 424, 343, 736, 895, 463, 1131, 455, 539, 718, 786, 412, 830, 205, 526, 819, 846, 1063, 834, 84, 897, 847, 312, 666, 8, 359, 25, 699, 1138, 484, 356, 249, 682, 775, 999, 190, 227, 936, 562, 843, 601, 315, 740, 926, 1200, 906, 1086, 746, 525, 683, 1194, 454, 200, 384, 623, 564, 1065, 1090, 499, 1137, 984, 552, 1034, 5, 98, 1166, 1149, 166, 141, 762, 193, 940, 1007, 960, 162, 1036, 652, 457, 1147, 681, 314, 391, 241, 129, 1180, 649, 636, 1069, 157, 1175, 877, 396, 267, 607, 1002, 780, 1184, 1113, 677, 744, 408, 417, 767, 16, 1185, 186, 1009, 684, 383, 212, 191, 264, 768, 72, 608, 279, 934, 1187, 590, 725, 263, 487, 360, 882, 92, 753, 1130, 604, 165, 874, 24, 436, 475, 737, 55, 501, 732, 280, 308, 260, 1169, 1082, 1201, 825, 393, 863, 471, 878, 278, 273, 406, 425, 870, 176, 73, 520, 1088, 149, 336, 1164, 402, 971, 322, 901, 741, 1140, 963, 335, 497, 438, 128, 696, 854, 1022, 879, 978, 289, 593, 802, 1062, 1174, 256, 332, 318, 274, 1051, 996, 489, 1203, 731, 950, 327, 587, 954, 529, 1183, 821, 1089, 988, 419, 197, 174, 888, 970, 101, 629, 1106, 929, 1120, 673, 473, 794, 596, 97, 224, 370, 103, 108, 600, 801, 448, 120, 246, 18, 930, 553, 1107, 1044, 170, 399, 747, 64, 546, 1127, 199, 153, 622, 1153]
LVIS_RARE_IDS=[389, 400, 809, 987, 864, 376, 1116, 446, 326, 14, 506, 1058, 606, 355, 619, 223, 974, 93, 1144, 210, 551, 902, 769, 1016, 439, 1165, 920, 13, 918, 545, 117, 937, 1135, 509, 618, 752, 1145, 17, 1003, 998, 1193, 214, 281, 580, 620, 671, 792, 333, 714, 49, 1012, 492, 352, 131, 420, 382, 119, 805, 478, 301, 257, 316, 925, 251, 123, 287, 1158, 63, 247, 1053, 516, 209, 1048, 574, 236, 295, 727, 603, 486, 858, 663, 292, 317, 428, 300, 972, 944, 1005, 743, 331, 646, 815, 202, 567, 956, 304, 238, 891, 82, 913, 942, 155, 812, 126, 38, 1015, 985, 179, 887, 458, 449, 541, 40, 594, 785, 159, 759, 321, 245, 852, 413, 1049, 764, 678, 894, 368, 616, 467, 851, 859, 1084, 142, 691, 664, 42, 353, 231, 855, 1157, 803, 1148, 634, 1047, 561, 862, 914, 983, 479, 651, 161, 908, 302, 917, 233, 783, 883, 1159, 1075, 354, 105, 931, 796, 151, 856, 527, 722, 1028, 52, 508, 291, 850, 873, 779, 869, 690, 313, 310, 535, 577, 787, 78, 270, 244, 482, 348, 374, 890, 542, 140, 357, 638, 602, 693, 729, 648, 1054, 632, 712, 599, 1124, 686, 265, 635, 1118, 674, 733, 513, 991, 1031, 31, 381, 625, 294, 662, 365, 481, 518, 269, 585, 266, 407, 1150, 975, 788, 182, 51, 702, 905, 323, 969, 784, 1129, 557, 215, 810, 958, 349, 262, 282, 853, 941, 414, 491, 688, 85, 571, 572, 130, 275, 755, 758, 410, 1126, 222, 234, 1057, 39, 397, 672, 952, 1119, 730, 849, 824, 1146, 640, 938, 939, 532, 575, 240, 992, 69, 416, 808, 21, 1029, 172, 398, 665, 886, 772, 435, 164, 167, 136, 892, 106, 610, 196, 427, 20, 488, 820, 994, 990, 1080, 568, 364, 113, 710, 250, 543, 1010, 362, 582, 754, 144, 1167, 426, 431, 30, 560, 829, 432, 583, 147, 388, 480, 405, 307, 597, 456, 538, 778, 237, 1032, 822, 366, 1030, 823, 657, 945, 258, 71, 503, 831]

unknown_category = {'frequency': 'f', 'id': 1231, 'synset': "unknown", 'synonyms': ["unknown"], 'def': 'unknown categories', 'name': 'unknown'}
LVIS_FREQUENT_CATEGORIES = [LVIS_V1_ID_CAT_MAP[cid] for cid in LVIS_FREQUENT_IDS]
LVIS_COMMON_CATEGORIES = [LVIS_V1_ID_CAT_MAP[cid] for cid in LVIS_COMMON_IDS]
LVIS_RARE_CATEGORIES = [LVIS_V1_ID_CAT_MAP[cid] for cid in LVIS_RARE_IDS]


LVIS_ALL_CATEGORIES = LVIS_V1_CATEGORIES


# add basemix and novelmix, half and half split, in total 703:
base_len, common_len, rare_len = 305, 361, 237
# in test: 100, 100, 100 = 300 classes
LVIS_BASEMIX_CATEGORIES = (
    LVIS_FREQUENT_CATEGORIES[0:base_len]
    + LVIS_COMMON_CATEGORIES[0:common_len]
    + LVIS_RARE_CATEGORIES[0:rare_len]
)
LVIS_NOVELMIX_CATEGORIES = (
    LVIS_FREQUENT_CATEGORIES[base_len:]
    + LVIS_COMMON_CATEGORIES[common_len:]
    + LVIS_RARE_CATEGORIES[rare_len:]
)
"""
Add datasplits by categories. For base classes, add "base" in the datasplit name.
"""
datasplit_categories = {
    # split it into two subsets: half and half across f, c, r
    "all": sorted(
        LVIS_V1_CATEGORIES, key=lambda x: x["id"] #LVIS_V1_CATEGORIES
    ),
    # our benchmarking
    "basemix": sorted(
        LVIS_BASEMIX_CATEGORIES, key=lambda x: x["id"]
    ),  # half f, half c, half r
    "novelmix": sorted(LVIS_NOVELMIX_CATEGORIES, key=lambda x: x["id"]),
    # demo: base with 1103 mix, novel with only 100 frequent
    "basev1": sorted(
        LVIS_FREQUENT_CATEGORIES[0:305] + LVIS_COMMON_CATEGORIES + LVIS_RARE_CATEGORIES,
        key=lambda x: x["id"],
    ),
    # training with unknown categories, will only be registered in pretraining
    "basev1unknown": sorted(
        LVIS_FREQUENT_CATEGORIES[0:305] + LVIS_COMMON_CATEGORIES + LVIS_RARE_CATEGORIES+[unknown_category],
        key=lambda x: x["id"],
    ),
    "novelv1": sorted(LVIS_FREQUENT_CATEGORIES[305:], key=lambda x: x["id"]),
    # base with 305 frequent, novel with only 100 frequent
    "basev2": sorted(
        LVIS_FREQUENT_CATEGORIES[0:305],
        key=lambda x: x["id"],
    ),
    # "basefc": sorted(
    #     LVIS_FREQUENT_CATEGORIES[0:305]+ LVIS_COMMON_CATEGORIES ,
    #     key=lambda x: x["id"],
    # ),
    # TFA: The split of fc(866) as base class and r(337) as novel classes is used in TFA
    "basefc": sorted(
        LVIS_FREQUENT_CATEGORIES + LVIS_COMMON_CATEGORIES,
        key=lambda x: x["id"],
    ),
    "novelr": sorted(
        LVIS_RARE_CATEGORIES,
        key=lambda x: x["id"],
    ),
    "cnno": sorted(LVIS_V1_COCO_NOVEL_NO_OVERLAP_CATEGORIES, key=lambda x: x["id"]),
    "base350wcommon": sorted(LVIS_FREQUENT_CATEGORIES[50:400] + LVIS_COMMON_CATEGORIES, key=lambda x: x["id"]),
}

# Add for number of different classes in base classes
datasplit_categories["novel50"] = sorted(LVIS_FREQUENT_CATEGORIES[0:50], key=lambda x: x["id"])
for num_base_classes in [50, 100, 150, 200, 250, 300, 350]:
    datasplit_categories[f"base{num_base_classes}"] = sorted(LVIS_FREQUENT_CATEGORIES[50:50+num_base_classes], key=lambda x: x["id"])

# Add for different number of classes from common categories
for num_common_classes in [100, 200, 300, 400]:
    datasplit_categories[f"base350wcommon{num_common_classes}"] = sorted(LVIS_FREQUENT_CATEGORIES[50:400] + LVIS_COMMON_CATEGORIES[0:num_common_classes], key=lambda x: x["id"])

for num_rare_classes in [100, 200, 300]:
    datasplit_categories[f"base350wcommonwrare{num_rare_classes}"] = sorted(LVIS_FREQUENT_CATEGORIES[50:400] + LVIS_COMMON_CATEGORIES+LVIS_RARE_CATEGORIES[0:num_rare_classes], key=lambda x: x["id"])
