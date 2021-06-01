import numpy as np
from matplotlib import colors
import matplotlib.pyplot as plt
import tensorflow as tf
from utils.data_utils import get_labels

labels = get_labels()
trainid2label = { label.trainId : label for label in labels }
catid2label = { label.categoryId : label for label in labels }


def plot_iou_trainId(trainId_label_map, catId_label_map, n_classes, iou_class, model, iou_mean):

    categories = [trainId_label_map[i].category for i in range(1, n_classes)]
    
    cat_colors = {
        'void': colors.to_hex(list(np.array(catId_label_map[0].color)/255)),
        'flat': colors.to_hex(list(np.array(catId_label_map[1].color)/255)),
        'construction': colors.to_hex(list(np.array(catId_label_map[2].color)/255)),
        'object': colors.to_hex(list(np.array(catId_label_map[3].color)/255)),
        'nature': colors.to_hex(list(np.array(catId_label_map[4].color)/255)),
        'sky': colors.to_hex(list(np.array(catId_label_map[5].color)/255)),
        'human': colors.to_hex(list(np.array(catId_label_map[6].color)/255)),
        'vehicle': colors.to_hex(list(np.array(catId_label_map[7].color)/255))
    }
    _colors = [cat_colors[category] for category in categories]

    names = [trainId_label_map[i].name for i in range(1, n_classes)]

    fig, ax = plt.subplots(figsize=(14,10))
    hbars = ax.barh(names, iou_class, color=_colors)
    
    ax.set_xlabel("IoU Coefficient: ", fontsize=18)
    ax.set_ylabel("Class Name", fontsize=18)
    ax.set_title("Class Scores for {} - Mean IoU: {:.3f}".format(model.name, iou_mean), fontsize=22)
    ax.set_xlim([0, 1])
    
    ax.bar_label(hbars, fmt="%.2f", padding=3, fontsize=16)
    
    plt.savefig("plots/"+model.name+"_class_iou_scores.png")
    plt.show()
    
    
def plot_iou_catId(catId_label_map, n_classes, iou_class, model, iou_mean):

    categories = [catId_label_map[i+1].category for i in range(n_classes-1)]
    cat_colors = {
        'void': colors.to_hex(list(np.array(catId_label_map[0].color)/255)),
        'flat': colors.to_hex(list(np.array(catId_label_map[1].color)/255)),
        'construction': colors.to_hex(list(np.array(catId_label_map[2].color)/255)),
        'object': colors.to_hex(list(np.array(catId_label_map[3].color)/255)),
        'nature': colors.to_hex(list(np.array(catId_label_map[4].color)/255)),
        'sky': colors.to_hex(list(np.array(catId_label_map[5].color)/255)),
        'human': colors.to_hex(list(np.array(catId_label_map[6].color)/255)),
        'vehicle': colors.to_hex(list(np.array(catId_label_map[7].color)/255))
    }
    _colors = [cat_colors[category] for category in categories]
    
    plt.figure(figsize=(14,10))
    plt.barh(categories, iou_class, color=_colors)
    plt.xlabel("IoU Coefficient", fontsize=18)
    plt.ylabel("Category Name", fontsize=18)
    plt.title("Category IoU Scores for {} - Average: {:.3f}".format(model.name, iou_mean), fontsize=22)
    plt.xlim([0, 1])
    plt.savefig("plots/"+model.name+"_category_iou_scores.png")
    plt.show()

    
def label_to_rgb(mask):
    h = mask.shape[0]
    w = mask.shape[1]
    mask_rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for val, key in trainid2label.items():
        indices = mask == val
        mask_rgb[indices.squeeze()] = key.color 
    return mask_rgb


def display(display_list, title=True):
    plt.figure(figsize=(15, 5), dpi=150) # dpi=200
    if title:
        title = ['Input Image', 'True Mask', 'Predicted Mask']
    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i+1)
        if title:
            plt.title(title[i])
        plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]))
        plt.axis('off')
    plt.tight_layout()
    plt.show()

    
def create_mask(pred_mask):
    pred_mask = tf.squeeze(pred_mask)
    pred_mask = tf.argmax(pred_mask, axis=-1)
    pred_mask = pred_mask[..., tf.newaxis]
    pred_mask = label_to_rgb(pred_mask.numpy())
    return pred_mask
