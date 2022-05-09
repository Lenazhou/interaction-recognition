import numpy as np
from PIL import Image,ImageEnhance
import random
import json
import numpy as np
from skimage import transform as sktsf
import math
import cv2

def read_image(path, dtype=np.float32, color=True):
    """Read an image from a file.

    This function reads an image from given file. The image is CHW format and
    the range of its value is :math:`[0, 255]`. If :obj:`color = True`, the
    order of the channels is RGB.

    Args:
        path (str): A path of image file.
        dtype: The type of array. The default value is :obj:`~numpy.float32`.
        color (bool): This option determines the number of channels.
            If :obj:`True`, the number of channels is three. In this case,
            the order of the channels is RGB. This is the default behaviour.
            If :obj:`False`, this function returns a grayscale image.

    Returns:
        ~numpy.ndarray: An image.
    """
    f = Image.open(path)
    k = random.uniform(1.3, 2.3)
    color_enhance=random.choice([True,False])

    if color_enhance:
        im_1 = ImageEnhance.Color(f).enhance(k)

    try:
        if color:
            img = f.convert('RGB')
        else:
            img = f.convert('P')
        img = np.asarray(img, dtype=dtype)
    finally:
        if hasattr(f, 'close'):
            f.close()

    if img.ndim == 2:
        # reshape (H, W) -> (1, H, W)
        return img[np.newaxis]
    else:
        # transpose (H, W, C) -> (C, H, W)

        return img.transpose((2, 0, 1))


def resize_bbox(bbox, in_size, out_size):
    """Resize bounding boxes according to image resize.

    The bounding boxes are expected to be packed into a two dimensional
    tensor of shape :math:`(R, 4)`, where :math:`R` is the number of
    bounding boxes in the image. The second axis represents attributes of
    the bounding box. They are :math:`(y_{min}, x_{min}, y_{max}, x_{max})`,
    where the four attributes are coordinates of the top left and the
    bottom right vertices.

    Args:
        bbox (~numpy.ndarray): An array whose shape is :math:`(R, 4)`.
            :math:`R` is the number of bounding boxes.
        in_size (tuple): A tuple of length 2. The height and the width
            of the image before resized.
        out_size (tuple): A tuple of length 2. The height and the width
            of the image after resized.

    Returns:
        ~numpy.ndarray:
        Bounding boxes rescaled according to the given image shapes.

    """
    bbox = bbox.copy()
    y_scale = float(out_size[0]) / in_size[0]
    x_scale = float(out_size[1]) / in_size[1]

    bbox[:, 0] = x_scale * bbox[:, 0]
    bbox[:, 2] = x_scale * bbox[:, 2]
    bbox[:, 1] = y_scale * bbox[:, 1]
    bbox[:, 3] = y_scale * bbox[:, 3]
    return bbox

def resize_bboxes2(bbox,in_size,out_size):
    bbox = bbox.copy()
    y_scale = float(out_size[0]) / in_size[0]
    x_scale = float(out_size[1]) / in_size[1]

    bbox[:,:, 0] = x_scale * bbox[:,:, 0]
    bbox[:,:, 2] = x_scale * bbox[:,:, 2]
    bbox[:,:, 1] = y_scale * bbox[:,:, 1]
    bbox[:,:, 3] = y_scale * bbox[:,:, 3]
    return bbox


def resize_interact_bbox(bbox, in_size, out_size):
    bbox = bbox.copy()
    y_scale = float(out_size[0]) / in_size[0]
    x_scale = float(out_size[1]) / in_size[1]
    bbox[:, 0] = y_scale * bbox[:, 0]
    bbox[:, 2] = y_scale * bbox[:, 2]
    bbox[:, 1] = x_scale * bbox[:, 1]
    bbox[:, 3] = x_scale * bbox[:, 3]
    bbox[:, 4] = x_scale * bbox[:, 4]


    return bbox

def flip_bbox(bbox, size, y_flip=False, x_flip=False):
    """Flip bounding boxes accordingly.

    The bounding boxes are expected to be packed into a two dimensional
    tensor of shape :math:`(R, 4)`, where :math:`R` is the number of
    bounding boxes in the image. The second axis represents attributes of
    the bounding box. They are :math:`(y_{min}, x_{min}, y_{max}, x_{max})`,
    where the four attributes are coordinates of the top left and the
    bottom right vertices.

    Args:
        bbox (~numpy.ndarray): An array whose shape is :math:`(R, 4)`.
            :math:`R` is the number of bounding boxes.
        size (tuple): A tuple of length 2. The height and the width
            of the image before resized.
        y_flip (bool): Flip bounding box according to a vertical flip of
            an image.
        x_flip (bool): Flip bounding box according to a horizontal flip of
            an image.

    Returns:
        ~numpy.ndarray:
        Bounding boxes flipped according to the given flips.

    """
    H, W = size
    bbox = bbox.copy()
    if y_flip:
        y_max = H - bbox[:, 0]
        y_min = H - bbox[:, 2]
        bbox[:, 0] = y_min
        bbox[:, 2] = y_max
    if x_flip:
        x_max = W - bbox[:, 0]
        x_min = W - bbox[:, 2]
        bbox[:, 0] = x_min
        bbox[:, 2] = x_max
    return bbox

def flip_interact_bbox(bbox, size, y_flip=False, x_flip=False):
    H, W = size
    bbox = bbox.copy()
    if y_flip:
        y_max = H - bbox[:, 0]
        y_min = H - bbox[:, 2]
        bbox[:, 0] = y_min
        bbox[:, 2] = y_max
    if x_flip:
        x_2 = W - bbox[:, 1]
        x_1 = bbox[:,3]-bbox[:,1]-bbox[:,4]
        x_0 = W - bbox[:, 3]
        bbox[:, 1] = x_0
        bbox[:, 4] = x_1
        bbox[:, 3] = x_2
    return bbox

def crop_bbox(
        bbox, y_slice=None, x_slice=None,
        allow_outside_center=True, return_param=False):
    """Translate bounding boxes to fit within the cropped area of an image.

    This method is mainly used together with image cropping.
    This method translates the coordinates of bounding boxes like
    :func:`data.util.translate_bbox`. In addition,
    this function truncates the bounding boxes to fit within the cropped area.
    If a bounding box does not overlap with the cropped area,
    this bounding box will be removed.

    The bounding boxes are expected to be packed into a two dimensional
    tensor of shape :math:`(R, 4)`, where :math:`R` is the number of
    bounding boxes in the image. The second axis represents attributes of
    the bounding box. They are :math:`(y_{min}, x_{min}, y_{max}, x_{max})`,
    where the four attributes are coordinates of the top left and the
    bottom right vertices.

    Args:
        bbox (~numpy.ndarray): Bounding boxes to be transformed. The shape is
            :math:`(R, 4)`. :math:`R` is the number of bounding boxes.
        y_slice (slice): The slice of y axis.
        x_slice (slice): The slice of x axis.
        allow_outside_center (bool): If this argument is :obj:`False`,
            bounding boxes whose centers are outside of the cropped area
            are removed. The default value is :obj:`True`.
        return_param (bool): If :obj:`True`, this function returns
            indices of kept bounding boxes.

    Returns:
        ~numpy.ndarray or (~numpy.ndarray, dict):

        If :obj:`return_param = False`, returns an array :obj:`bbox`.

        If :obj:`return_param = True`,
        returns a tuple whose elements are :obj:`bbox, param`.
        :obj:`param` is a dictionary of intermediate parameters whose
        contents are listed below with key, value-type and the description
        of the value.

        * **index** (*numpy.ndarray*): An array holding indices of used \
            bounding boxes.

    """

    t, b = _slice_to_bounds(y_slice)
    l, r = _slice_to_bounds(x_slice)
    crop_bb = np.array((t, l, b, r))

    if allow_outside_center:
        mask = np.ones(bbox.shape[0], dtype=bool)
    else:
        center = (bbox[:, :2] + bbox[:, 2:]) / 2.0
        mask = np.logical_and(crop_bb[:2] <= center, center < crop_bb[2:]) \
            .all(axis=1)

    bbox = bbox.copy()
    bbox[:, :2] = np.maximum(bbox[:, :2], crop_bb[:2])
    bbox[:, 2:] = np.minimum(bbox[:, 2:], crop_bb[2:])
    bbox[:, :2] -= crop_bb[:2]
    bbox[:, 2:] -= crop_bb[:2]

    mask = np.logical_and(mask, (bbox[:, :2] < bbox[:, 2:]).all(axis=1))
    bbox = bbox[mask]

    if return_param:
        return bbox, {'index': np.flatnonzero(mask)}
    else:
        return bbox

def _slice_to_bounds(slice_):
    if slice_ is None:
        return 0, np.inf

    if slice_.start is None:
        l = 0
    else:
        l = slice_.start

    if slice_.stop is None:
        u = np.inf
    else:
        u = slice_.stop

    return l, u


def translate_bbox(bbox, y_offset=0, x_offset=0):
    """Translate bounding boxes.

    This method is mainly used together with image transforms, such as padding
    and cropping, which translates the left top point of the image from
    coordinate :math:`(0, 0)` to coordinate
    :math:`(y, x) = (y_{offset}, x_{offset})`.

    The bounding boxes are expected to be packed into a two dimensional
    tensor of shape :math:`(R, 4)`, where :math:`R` is the number of
    bounding boxes in the image. The second axis represents attributes of
    the bounding box. They are :math:`(y_{min}, x_{min}, y_{max}, x_{max})`,
    where the four attributes are coordinates of the top left and the
    bottom right vertices.

    Args:
        bbox (~numpy.ndarray): Bounding boxes to be transformed. The shape is
            :math:`(R, 4)`. :math:`R` is the number of bounding boxes.
        y_offset (int or float): The offset along y axis.
        x_offset (int or float): The offset along x axis.

    Returns:
        ~numpy.ndarray:
        Bounding boxes translated according to the given offsets.

    """

    out_bbox = bbox.copy()
    out_bbox[:, :2] += (y_offset, x_offset)
    out_bbox[:, 2:] += (y_offset, x_offset)

    return out_bbox


def random_flip(img, y_random=False, x_random=False,
                return_param=False, copy=False):
    """Randomly flip an image in vertical or horizontal direction.

    Args:
        img (~numpy.ndarray): An array that gets flipped. This is in
            CHW format.
        y_random (bool): Randomly flip in vertical direction.
        x_random (bool): Randomly flip in horizontal direction.
        return_param (bool): Returns information of flip.
        copy (bool): If False, a view of :obj:`img` will be returned.

    Returns:
        ~numpy.ndarray or (~numpy.ndarray, dict):

        If :obj:`return_param = False`,
        returns an array :obj:`out_img` that is the result of flipping.

        If :obj:`return_param = True`,
        returns a tuple whose elements are :obj:`out_img, param`.
        :obj:`param` is a dictionary of intermediate parameters whose
        contents are listed below with key, value-type and the description
        of the value.

        * **y_flip** (*bool*): Whether the image was flipped in the\
            vertical direction or not.
        * **x_flip** (*bool*): Whether the image was flipped in the\
            horizontal direction or not.

    """
    y_flip, x_flip = False, False
    if y_random:
        y_flip = random.choice([True, False])
    if x_random:
        x_flip = random.choice([True, False])

    if y_flip:
        img = img[:, ::-1, :]
    if x_flip:
        img = img[:, :, ::-1]

    if copy:
        img = img.copy()

    if return_param:
        return img, {'y_flip': y_flip, 'x_flip': x_flip}
    else:
        return img

def draw_center_crop(coor):
    coor[:,0] = coor[:,0] - 60
    coor[:,1] = coor[:,1] - 40
    coor[:,2] = coor[:,2] - 60
    coor[:,3] = coor[:,3] - 40
    delatX=coor[:,4]

    for k, i in enumerate(coor[:, 0]):
        if i < 0:
            x=abs(i)
            if x>delatX:
                coor[k,4]=0
            else:
                coor[k,4]=delatX-x
            coor[k, 0] = 0

    for k, i in enumerate(coor[:, 1]):
        if i < 0:
            coor[k, 1] = 0
    for k,i in enumerate(coor[:,2]):
        if i >780:
            coor[k,2]=780
    for k,i in enumerate(coor[:,3]):
        if i>520:
            coor[k,3]=520
    return coor

def center_resize_img(image,coor):
    image=image[:,60:540,90:810]   #image=(3,480,720)
    image=sktsf.resize(image,(3,600,900), mode='reflect',anti_aliasing=False)
    coor[:, 0] = (coor[:, 0] - 90) *1.25
    coor[:, 1] = (coor[:, 1] - 60) *1.25
    coor[:, 2] = (coor[:, 2] - 90) *1.25
    coor[:, 3] = (coor[:, 3] - 60) *1.25

    return image,coor

def random_crop_resize(img,coor,crop_random=False):
    center_crop=False
    if crop_random:
        center_crop = random.choice([True, False])
    if center_crop:
        img,coor=center_resize_img(img,coor)
        coor=np.array(coor)
    return img,coor

def load_json(p):
	with open(p,'r') as f:
		load_dict=json.load(f)
	return load_dict

def cal_box_center(box1,box2):
    box1_x1,box1_y1=box1[0],box1[1]
    box1_x2,box1_y2=box1[2],box1[3]
    box2_x1, box2_y1 =box2[0],box2[1]
    box2_x2, box2_y2 =box2[2],box2[3]
    box1_c_x=((box1_x2-box1_x1)/2.0)+box1_x1
    box1_c_y=((box1_y2-box1_y1)/2.0)+box1_y1
    box2_c_x = ((box2_x2 - box2_x1) / 2.0) + box2_x1
    box2_c_y = ((box2_y2 - box2_y1) / 2.0) + box2_y1
    dis = math.sqrt(math.pow((box2_c_x-box1_c_x),2)+math.pow((box2_c_y-box1_c_y),2))
    return dis

def get_box(box1,box2):
    x_min = min(box1[0],box2[0])
    y_min = min(box1[1],box2[1])
    x_max = max(box1[2],box2[2])
    y_max = max(box1[3],box2[3])
    return [int(x_min),int(y_min),int(x_max),int(y_max)]

def put_mask(image,bbox1):
    zeros1 = np.zeros((image.shape), dtype=np.uint8)
    zeros_mask1 = cv2.rectangle(zeros1, (bbox1[0], bbox1[1]), (bbox1[2], bbox1[3]),
                    color=(255,255,255), thickness=-1 ) #thickness=-1 表示矩形框内颜色填充
    zeros_mask = np.array((zeros_mask1))
    try:
    	# alpha 为第一张图片的透明度
        alpha = 1
        gamma = 0
        beta =1
        # cv2.addWeighted 将原始图片与 mask 融合
        mask_img = cv2.addWeighted(image, alpha, zeros_mask, beta, gamma)
        return mask_img
    except Exception as e:
        print(e)
        return None

def print_image(action_txt,path_image,out_path,group_p,label,bbox):
    font = cv2.FONT_HERSHEY_SIMPLEX
    image = cv2.imread(path_image)
    for id, (g,l) in enumerate(zip(group_p,label)):
        if l != 9:
            box1 = bbox[g[0]]
            box2 = bbox[g[1]]
            box = get_box(box1,box2)
            cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (255, 255, 255), 6)
            image = put_mask(image,[box[0],box[1]-55,box[0]+95,box[1]])
            b = [int(box[0]),int(box[1]-100),int(box[0]+180),int(box[1])]
            image = put_mask(image, b)
            image= cv2.putText(image,action_txt[int(l)] , (box[0]+5, box[1]-5), font, 4, (0, 0, 0), 8)
    cv2.imwrite(out_path,image)