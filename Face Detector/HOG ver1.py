import cv2
import numpy as np
import matplotlib.pyplot as plt


# define the 3x3 horizontal and vertical filters
def get_differential_filter():
    filter_x = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
    filter_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])

    return filter_x, filter_y


# apply given filter to given image
def filter_image(im, filter):
    filterWidth, filterHeight = filter.shape
    # define the size of padding for generic filter (assuming that filter is a square)
    paddedSize = int((filterWidth / 2) + filterWidth % 2 - 1)

    # bless numpy
    paddedIm = np.pad(im, paddedSize)
    paddedWidth, paddedHeight = paddedIm.shape

    im_filtered = np.zeros((paddedWidth, paddedHeight))
    # perform filtering for each cell
    for i in range(paddedSize, paddedWidth - paddedSize):
        for j in range(paddedSize, paddedHeight - paddedSize):
            im_filtered[i, j] = np.sum(
                np.sum(filter * paddedIm[i - paddedSize: i + paddedSize + 1, j - paddedSize: j + paddedSize + 1])
            )

    return im_filtered


def get_gradient(im_dx, im_dy):
    grad_mag = np.sqrt(im_dx * im_dx + im_dy * im_dy)
    grad_angle = np.arctan2(im_dy, im_dx)
    return grad_mag, grad_angle


def get_angle_index(angle):
    # convert to angles
    angle_deg = (angle * 180) / np.pi

    if 0 <= angle_deg < 15 or 165 <= angle_deg < 180:
        return 0

    elif 15 <= angle_deg < 45:
        return 1
    
    elif 45 <= angle_deg < 75:
        return 2

    elif 75 <= angle_deg < 105:
        return 3

    elif 105 <= angle_deg < 135:
        return 4

    elif 135 <= angle_deg < 165:
        return 5


def build_histogram(grad_mag, grad_angle, cell_size=8):
    sizeX, sizeY = grad_mag.shape

    width = int(sizeX / cell_size)
    height = int(sizeY / cell_size)

    ori_histo = np.zeros((width, height, 6))
    # for each, sort based on angle, by calling a separate function
    for i in range(width):
        for j in range(height):
            for m in range(cell_size):
                for n in range(cell_size):
                    bin_index = get_angle_index(grad_angle[i * cell_size + m][j * cell_size + n])
                    magnitude = grad_mag[i * cell_size + m][j * cell_size + n]
                    ori_histo[i][j][bin_index] += magnitude

    return ori_histo


def get_block_descriptor(ori_histo, block_size=2):
    e = 0.001
    lenX, lenY, lenZ = ori_histo.shape
    # initialize array of zeros
    ori_histo_normalized = np.zeros((lenX - block_size + 1, lenY - block_size + 1, lenZ * block_size * block_size))
    for i in range(lenX - block_size + 1):
        for j in range(lenY - block_size + 1):
            # get the blocks
            temp = []
            for m in range(block_size):
                for n in range(block_size):
                    for z in range(lenZ):
                        temp.append(ori_histo[i + m][j + n][z])

            temp = np.array(temp)
            denom = np.sqrt(np.sum(temp * temp) + e * e)
            # normalize
            norm = temp / denom

            for z in range(lenZ * block_size * block_size):
                ori_histo_normalized[i][j][z] = norm[z]

    return ori_histo_normalized


def extract_hog(im):
    # convert grey-scale image to double format and normalize
    im = im.astype('float') / 255.0
    # get x and y filter
    filter_x, filter_y = get_differential_filter()
    # get differentials
    im_dx, im_dy = filter_image(im, filter_x), filter_image(im, filter_y)
    # get magnitude and angle
    grad_mag, grad_angle = get_gradient(im_dx, im_dy)
    # build histogram based on magnitude, angle and cell size
    ori_histo = build_histogram(grad_mag, grad_angle, 8)
    # normalize the histogram
    ori_histo_normalized = get_block_descriptor(ori_histo, 2)
    # concatenate all block descriptors
    hog = ori_histo_normalized.reshape((-1))

    # show original image
    # plt.imshow(im, cmap='gray')
    # plt.show()
    # show dx and dy
    # plt.imshow(im_dx, cmap='hot', interpolation='nearest')
    # plt.show()
    # plt.imshow(im_dy, cmap='hot', interpolation='nearest')
    # plt.show()
    # show grad magnitude
    # plt.imshow(grad_mag, cmap='hot', interpolation='nearest')
    # plt.show()
    # show grad angle
    # plt.imshow(grad_angle, cmap='hot', interpolation='nearest')
    # plt.show()

    # visualize to verify
    # visualize_hog(im, hog, 8, 2)

    return hog


# visualize histogram of each block
def visualize_hog(im, hog, cell_size, block_size):
    num_bins = 6
    max_len = 7  # control sum of segment lengths for visualized histogram bin of each block
    im_h, im_w = im.shape
    num_cell_h, num_cell_w = int(im_h / cell_size), int(im_w / cell_size)
    num_blocks_h, num_blocks_w = num_cell_h - block_size + 1, num_cell_w - block_size + 1
    histo_normalized = hog.reshape((num_blocks_h, num_blocks_w, block_size ** 2, num_bins))
    histo_normalized_vis = np.sum(histo_normalized ** 2, axis=2) * max_len  # num_blocks_h x num_blocks_w x num_bins
    angles = np.arange(0, np.pi, np.pi / num_bins)
    mesh_x, mesh_y = np.meshgrid(np.r_[cell_size: cell_size * num_cell_w: cell_size],
                                 np.r_[cell_size: cell_size * num_cell_h: cell_size])
    mesh_u = histo_normalized_vis * np.sin(angles).reshape((1, 1, num_bins))  # expand to same dims as histo_normalized
    mesh_v = histo_normalized_vis * -np.cos(angles).reshape((1, 1, num_bins))  # expand to same dims as histo_normalized
    plt.imshow(im, cmap='gray', vmin=0, vmax=1)
    for i in range(num_bins):
        plt.quiver(mesh_x - 0.5 * mesh_u[:, :, i], mesh_y - 0.5 * mesh_v[:, :, i], mesh_u[:, :, i], mesh_v[:, :, i],
                   color='white', headaxislength=0, headlength=0, scale_units='xy', scale=1, width=0.002, angles='xy')
    plt.show()


def non_max_suppression(boxes, overlapThresh):

    keep = []
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 0] + 50
    y2 = boxes[:, 1] + 50

    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    index = np.argsort(y2)
    # check overlap and keep the data if not overlapped
    while len(index) > 0:

        last = len(index) - 1
        i = index[last]
        keep.append(i)

        xx1 = np.maximum(x1[i], x1[index[:last]])
        yy1 = np.maximum(y1[i], y1[index[:last]])
        xx2 = np.minimum(x2[i], x2[index[:last]])
        yy2 = np.minimum(y2[i], y2[index[:last]])

        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        boxOverlap = (w * h) / area[index[:last]]

        index = np.delete(index, np.concatenate(([last], np.where(boxOverlap > overlapThresh)[0])))

    return boxes[keep]


def normalized_cross_correlation(arg1, arg2):
    mean_data1 = np.mean(arg1)
    std_data1 = np.std(arg1, ddof=1)
    arg1_t = (arg1 - mean_data1) / std_data1

    mean_data2 = np.mean(arg2)
    std_data2 = np.std(arg2, ddof=1)
    arg2_t = (arg2 - mean_data2) / std_data2

    return (1.0 / (arg1.size - 1)) * np.sum(arg1_t * arg2_t)


def face_recognition(I_target, I_template):
    template_hog = extract_hog(I_template)

    bounding_boxes = np.zeros((I_target.shape[0] * I_target.shape[1], 3))
    k = 0
    # how many steps from previous picture, larger number - faster performance (up to 50)
    iterStep = 5
    for i in range(0, I_target.shape[1] - I_template.shape[1], iterStep):
        for j in range(0, I_target.shape[0] - I_template.shape[0], iterStep):
            target_hog = extract_hog(I_target[j: I_template.shape[1] + j + 1, i: I_template.shape[0] + i + 1])

            score = normalized_cross_correlation(template_hog, target_hog)

            # thresholding to avoid processing extra data and have local maximums
            # needs to be adjusted based on stepSize
            if score >= 0.44:
                bounding_boxes[k] = np.array([i, j, score])
                k = k + 1

    # trim the data
    bounding_boxes = bounding_boxes[0: k]
    bounding_boxes = non_max_suppression(bounding_boxes, 0.5)

    return bounding_boxes


def visualize_face_detection(I_target, bounding_boxes, box_size):
    hh, ww, cc = I_target.shape

    fimg = I_target.copy()
    for ii in range(bounding_boxes.shape[0]):

        x1 = bounding_boxes[ii, 0]
        x2 = bounding_boxes[ii, 0] + box_size
        y1 = bounding_boxes[ii, 1]
        y2 = bounding_boxes[ii, 1] + box_size

        if x1 < 0:
            x1 = 0
        if x1 > ww - 1:
            x1 = ww - 1
        if x2 < 0:
            x2 = 0
        if x2 > ww - 1:
            x2 = ww - 1
        if y1 < 0:
            y1 = 0
        if y1 > hh - 1:
            y1 = hh - 1
        if y2 < 0:
            y2 = 0
        if y2 > hh - 1:
            y2 = hh - 1
        fimg = cv2.rectangle(fimg, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 1)
        cv2.putText(fimg, "%.2f" % bounding_boxes[ii, 2], (int(x1) + 1, int(y1) + 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 255, 0), 2, cv2.LINE_AA)

    plt.figure(3)
    plt.imshow(fimg, vmin=0, vmax=1)
    plt.show()


if __name__ == '__main__':
    im = cv2.imread('einstein.tif', 0)
    hog = extract_hog(im)

    I_target = cv2.imread('target.png', 0)
    # MxN image

    I_template = cv2.imread('template.png', 0)
    # mxn  face template

    bounding_boxes = face_recognition(I_target, I_template)

    I_target_c = cv2.imread('target.png')
    # MxN image (just for visualization)
    visualize_face_detection(I_target_c, bounding_boxes, I_template.shape[0])
    # this is visualization code.
