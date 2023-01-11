from matplotlib import patches
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as transforms


def display(image):

    img, _ = image
    img = np.transpose(img.numpy(), (1, 2, 0))

    plt.figure()
    plt.imshow(img)
    plt.gca().get_xaxis().set_visible(False)
    plt.gca().get_yaxis().set_visible(False)
    plt.show()


def extract_images(images):
    return [resize_image(image[0]) for image in images]


def extract_bbox(images):
    return [transform_bbox(image) for image in images]


def extract_single_bbox(bbox_coor, image):
    bbox = {
        "xmin": bbox_coor[0],
        "xmax": bbox_coor[1],
        "ymin": bbox_coor[3],
        "ymax": bbox_coor[4],
    }
    input_image_size = {"width": image[0].shape[1], "height": image[0].shape[2]}
    out_bbox = {"cx": 0.0, "cy": 0.0, "width": 0.0, "height": 0.0}
    scale_width = float(224) / input_image_size["width"]
    scale_height = float(224) / input_image_size["height"]
    out_bbox["cx"] = scale_width * 0.5 * (bbox["xmin"] + bbox["xmax"]) / 224
    out_bbox["cy"] = scale_height * 0.5 * (bbox["ymin"] + bbox["ymax"]) / 224
    out_bbox["width"] = scale_width * float(bbox["xmax"] - bbox["xmin"]) / 224
    out_bbox["height"] = scale_height * float(bbox["ymax"] - bbox["ymin"]) / 224
    out = torch.Tensor(out_bbox)
    return out


def transform_bbox(bbox, input_image_size, image):
    if image[1]["bbox"].type != "list":
        return extract_single_bbox(image[1]["bbox"].numpy(), image)
    else:
        bbox_list = []
        for b in image[1]["bbox"]:
            bbox_list.append(extract_single_bbox(b.numpy()), image)
        out = torch.Tensor(bbox_list)
        return out


def resize_image(image):
    preprocess_image = transforms.Resize(224)
    return preprocess_image(image)


def plot_bbox(ax, bbox):
    cx, cy, width, height = bbox
    upl_x, upl_y = cx - width / 2.0, cy - height / 2.0
    p = patches.Rectangle(
        (upl_x, upl_y),
        width,
        height,
        fill=False,
        clip_on=False,
        edgecolor="yellow",
        linewidth=4,
    )
    ax.add_patch(p)
