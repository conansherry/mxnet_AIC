import cv2
import math
import numpy as np
import mxnet as mx
from scipy.ndimage.filters import gaussian_filter
from scipy.spatial.distance import squareform, pdist
import matplotlib.pyplot as plt
import scipy

def gauss(x, a, b, c, d=0):
    return a * np.exp(-(x - b) ** 2 / (2 * c ** 2)) + d

def color_heatmap(x):
    color = np.zeros((x.shape[0], x.shape[1], 3))
    color[:, :, 0] = gauss(x, .5, .6, .2) + gauss(x, 1, .8, .3)
    color[:, :, 1] = gauss(x, 1, .5, .3)
    color[:, :, 2] = gauss(x, 1, .2, .3)
    color[color > 1] = 1
    color = (color * 255).astype(np.uint8)
    return color

def color_vectormap(x, y):
    mag, ang = cv2.cartToPolar(x, y)
    color = color_heatmap(mag)
    return color

def cvim_with_heatmap(inp, out, num_rows=5, parts_to_show=None):
    inp = inp.transpose((2, 0, 1)).astype(np.float32) / 255
    out = out.transpose((2, 0, 1))
    return sample_with_heatmap(inp, out, num_rows=num_rows, parts_to_show=parts_to_show)


def cvim_with_vectormap(inp, out, num_rows=5, parts_to_show=None):
    inp = inp.transpose((2, 0, 1)).astype(np.float32) / 255
    out = out.transpose((2, 0, 1))
    return sample_with_vectormap(inp, out, num_rows=num_rows, parts_to_show=parts_to_show)


def sample_with_vectormap(inp, out, num_rows=5, parts_to_show=None):
    inp = inp * 255

    img = np.zeros((inp.shape[1], inp.shape[2], inp.shape[0]))
    for i in range(3):
        img[:, :, i] = inp[2-i, :, :]

    if parts_to_show is None:
        parts_to_show = np.arange(out.shape[0]/2)

    # Generate a single image to display input/output pair
    num_cols = int(np.ceil(float(len(parts_to_show)) / num_rows))
    size = img.shape[0] // num_rows

    full_img = np.zeros((img.shape[0], size * num_cols + img.shape[1], 3), np.uint8)
    full_img[:img.shape[0], :img.shape[1]] = img

    inp_small = scipy.misc.imresize(img, [size, size])

    # Set up heatmap display for each part
    for i, part in enumerate(parts_to_show):
        part_idx1 = 2*part
        part_idx2 = 2*part+1
        out_resizedx = scipy.misc.imresize(out[part_idx1], [size, size])
        out_resizedy = scipy.misc.imresize(out[part_idx2], [size, size])
        out_resizedx = np.array(out_resizedx).astype(float) / 255
        out_resizedy = np.array(out_resizedy).astype(float) / 255
        out_img = inp_small.copy() * .3
        color_hm = color_vectormap(out_resizedx, out_resizedy)
        out_img += color_hm * .7

        col_offset = (i % num_cols) * size + img.shape[1]
        row_offset = (i // num_cols) * size
        full_img[row_offset:row_offset + size, col_offset:col_offset + size] = out_img

    return full_img


def sample_with_heatmap(inp, out, num_rows=5, parts_to_show=None):
    inp = inp * 255

    img = np.zeros((inp.shape[1], inp.shape[2], inp.shape[0]))
    for i in range(3):
        img[:, :, i] = inp[2-i, :, :]

    if parts_to_show is None:
        parts_to_show = np.arange(out.shape[0])

    # Generate a single image to display input/output pair
    num_cols = int(np.ceil(float(len(parts_to_show)) / num_rows))
    size = img.shape[0] // num_rows

    full_img = np.zeros((img.shape[0], size * (num_cols + num_rows), 3), np.uint8)
    full_img[:img.shape[0], :img.shape[1]] = img

    inp_small = scipy.misc.imresize(img, [size, size])

    # Set up heatmap display for each part
    for i, part in enumerate(parts_to_show):
        part_idx = part
        out_resized = scipy.misc.imresize(out[part_idx], [size, size])
        out_resized = out_resized.astype(float) / 255
        out_img = inp_small.copy() * .3
        color_hm = color_heatmap(out_resized)
        out_img += color_hm * .7

        col_offset = (i % num_cols + num_rows) * size
        row_offset = (i // num_cols) * size
        full_img[row_offset:row_offset + size, col_offset:col_offset + size] = out_img

    return full_img

def padRightDownCorner(img, stride, padValue=127):
    assert(img.shape[0] % stride == 0)
    w = img.shape[1]
    pad = 0 if (w % stride == 0) else stride - (w % stride)  # right
    pad_right = np.tile(img[:, -2:-1, :]*0 + padValue, (1, pad, 1))
    img_padded = np.concatenate((img, pad_right), axis=1)

    return img_padded, pad

def multiscale_cnn_forward(oriImg, net, param):
    h = oriImg.shape[0]
    w = oriImg.shape[1]
    boxsize = param.boxsize
    starting_range = param.starting_range
    ending_range = param.ending_range
    octave = param.octave
    stride = param.stride
    padValue = param.padValue

    starting_scale = float(boxsize) / float(h) * starting_range
    ending_scale = float(boxsize) / float(h) * ending_range
    multiplier = np.linspace(starting_scale, ending_scale, octave)

    multiplier = [round(h * s / stride) * stride / h for s in multiplier]

    heatmap_avg = 0  # save memory
    paf_avg = 0  # save memory

    for m in range(len(multiplier)):
        scale = multiplier[m]
        sh = int(round(h * scale))
        sw = int(round(w * scale))

        imageToTest = cv2.resize(oriImg, (sw, sh), interpolation=cv2.INTER_CUBIC)
        imageToTest_padded, pad = padRightDownCorner(imageToTest, stride, padValue)

        # print imageToTest_padded.shape
        # cv2.imshow("pad", imageToTest_padded)
        # cv2.waitKey(0)

        imageToTest_padded = np.expand_dims(imageToTest_padded.transpose((2, 0, 1)), 0).astype(np.float32)
        # imageToTest_padded = (imageToTest_padded.astype(np.uint8) - 127) / 255.
        # imageToTest_padded = (imageToTest_padded - 127) / 255.
        imageToTest_padded = (imageToTest_padded - 128) / 256.

        net.blobs['data'].reshape(*imageToTest_padded.shape)
        net.reshape()
        forward_kwargs = {'data': imageToTest_padded}
        blobs_out = net.forward(**forward_kwargs)

        output1 = blobs_out['Mconv7_stage6_L1'][0]
        resize_output1 = np.zeros((output1.shape[0], h, w))
        output2 = blobs_out['Mconv7_stage6_L2'][0]
        # output2 = net.blobs['Mconv7_stage5_L2'].data[0]
        resize_output2 = np.zeros((output2.shape[0], h, w))

        for i in range(output1.shape[0]):
            resize_output1[i] = cv2.resize(output1[i], (w, h))
        for i in range(output2.shape[0]):
            resize_output2[i] = cv2.resize(output2[i], (w, h))

        heatmap_avg += resize_output2[:-1]
        paf_avg += resize_output1

    heatmap_avg = heatmap_avg / float(octave)
    paf_avg = paf_avg / float(octave)

    visual = True
    if visual:
        div_num = 255.
        mean_value = 127
        npaf = 38
        nparts = 18
        stride = 8
        label = np.concatenate((paf_avg, heatmap_avg), axis=0)
        img = np.copy(oriImg)

        perimg_len = 200

        show_height = 6
        show_width = 7
        show_pafs = np.zeros((show_height * perimg_len, show_width * perimg_len, 3), dtype=np.uint8)
        for j in range(npaf):
            pafs = (np.abs(label[j, :, :]) * 255).astype(np.uint8)
            pafs = cv2.applyColorMap(pafs, cv2.COLORMAP_JET)
            img_pafs = (0.6 * img + 0.4 * pafs).astype(np.uint8)
            index_row = j / show_width
            index_col = j % show_width
            show_pafs[index_row * perimg_len:(index_row + 1) * perimg_len,
            index_col * perimg_len:(index_col + 1) * perimg_len, :] = cv2.resize(img_pafs, (perimg_len, perimg_len))
        cv2.imshow('show_pafs', show_pafs)

        show_len = 5
        show_parts = np.zeros((show_len * perimg_len, show_len * perimg_len, 3), dtype=np.uint8)
        for j in range(nparts):
            parts = (label[npaf + j, :, :] * 255).astype(np.uint8)
            parts = cv2.applyColorMap(parts, cv2.COLORMAP_JET)
            img_parts = (0.6 * img + 0.4 * parts).astype(np.uint8)
            index_row = j / show_len
            index_col = j % show_len
            show_parts[index_row * perimg_len:(index_row + 1) * perimg_len,
            index_col * perimg_len:(index_col + 1) * perimg_len, :] = cv2.resize(img_parts,
                                                                                 (perimg_len, perimg_len))
        cv2.imshow('show_parts', show_parts)

        cv2.imshow('img', img)

    heatmap_avg = heatmap_avg.transpose((1, 2, 0))
    paf_avg = paf_avg.transpose((1, 2, 0))

    if visual:
        heatmap_show = (np.max(heatmap_avg, 2) * 255).astype(np.uint8)
        heatmap_show = cv2.applyColorMap(heatmap_show, cv2.COLORMAP_JET)
        img_parts = np.copy(oriImg)
        img_parts = (0.6 * img_parts + 0.4 * heatmap_show).astype(np.uint8)
        cv2.imshow('heatmap_show', img_parts)
        cv2.waitKey()

    return heatmap_avg, paf_avg

# find connection in the specified sequence, center 29 is in the position 15
limbSeq = [[14, 1], [14, 4], [1, 2], [2, 3], [4, 5], [5, 6], [14, 7], [7, 8], \
           [8, 9], [14, 10], [10, 11], [11, 12], [14, 13]]

# the middle joints heatmap correpondence
mapIdx = [[27, 28], [33, 34], [29, 30], [31, 32], [35, 36], [37, 38], [15, 16], [17, 18], \
          [19, 20], [21, 22], [23, 24], [25, 26], [39, 40]]

# visualize
colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0], \
          [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255], \
          [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]

def findPeaks(map, thre):
    map_left = np.zeros(map.shape)
    map_left[1:, :] = map[:-1, :]
    map_right = np.zeros(map.shape)
    map_right[:-1, :] = map[1:, :]
    map_up = np.zeros(map.shape)
    map_up[:, 1:] = map[:, :-1]
    map_down = np.zeros(map.shape)
    map_down[:, :-1] = map[:, 1:]

    peaks_binary = np.logical_and.reduce(
        (map >= map_left, map >= map_right, map >= map_up, map >= map_down, map > thre))

    (Y, X) = np.nonzero(peaks_binary)
    score = [map[y, x] for (x, y) in zip(X, Y)]

    if len(score) <= 1:
        peaks_with_score = [(x, y, map[y, x]) for (x, y) in zip(X, Y)]
    else:  # NMS
        xyscore = np.vstack((X, Y, score)).T
        xyscore = sorted(xyscore, key=lambda x: x[2], reverse=True)
        distance_table = squareform(pdist(np.array(xyscore)[:, :2], 'euclidean'), force='tomatrix') < 6
        kill_by_id = np.sum(np.cumsum(distance_table != 0, 1) == 0, axis=1)
        vaildIdx = np.where(kill_by_id == range(len(xyscore)))[0]
        peaks_with_score = [(xyscore[i][0].astype(int), xyscore[i][1].astype(int), xyscore[i][2]) for i in vaildIdx]
    return peaks_with_score

def connect_aic_LineVec(oriImg, heatmap_avg, paf_avg, param):
    all_peaks = []
    peak_counter = 0

    for part in range(14):
        peaks_with_score = findPeaks(heatmap_avg[:, :, part], param.thre1)

        idc = range(peak_counter, peak_counter + len(peaks_with_score))
        peaks_with_score_and_id = [peaks_with_score[i] + (idc[i],) for i in range(len(idc))]

        all_peaks.append(peaks_with_score_and_id)
        peak_counter += len(peaks_with_score)

    connection_all = []
    special_k = []
    mid_num = 10

    for k in range(len(mapIdx)):
        score_mid = paf_avg[:, :, [x - 15 for x in mapIdx[k]]]
        candA = all_peaks[limbSeq[k][0] - 1]
        candB = all_peaks[limbSeq[k][1] - 1]
        nA = len(candA)
        nB = len(candB)
        indexA, indexB = limbSeq[k]

        if nA != 0 and nB != 0:
            connection_candidate = []
            for i in range(nA):
                for j in range(nB):
                    vec = np.subtract(candB[j][:2], candA[i][:2])
                    norm = math.sqrt(vec[0] * vec[0] + vec[1] * vec[1])
                    vec = np.divide(vec, norm)

                    startend = zip(np.linspace(candA[i][0], candB[j][0], num=mid_num), \
                                   np.linspace(candA[i][1], candB[j][1], num=mid_num))

                    vec_x = np.array([score_mid[int(round(startend[I][1])), int(round(startend[I][0])), 0] \
                                      for I in range(len(startend))])
                    vec_y = np.array([score_mid[int(round(startend[I][1])), int(round(startend[I][0])), 1] \
                                      for I in range(len(startend))])

                    score_midpts = np.multiply(vec_x, vec[0]) + np.multiply(vec_y, vec[1])
                    score_with_dist_prior = sum(score_midpts) / len(score_midpts) + min(
                        float(oriImg.shape[0]) / (norm + 1) - 1, 0)
                    # criterion1 = len(np.nonzero(score_midpts > param.thre2)[0]) > 0.8 * len(score_midpts)
                    criterion1 = (sum(score_midpts) / mid_num) > 0.1
                    criterion2 = score_with_dist_prior > 0

                    if False:
                        score_mid_norm = np.linalg.norm(score_mid, axis=2)
                        score_crop = np.zeros(score_mid_norm.shape)
                        crop_x1 = min(int(candA[i][0]), int(candB[j][0]))
                        crop_x2 = max(int(candA[i][0]), int(candB[j][0]))
                        crop_y1 = min(int(candA[i][1]), int(candB[j][1]))
                        crop_y2 = max(int(candA[i][1]), int(candB[j][1]))
                        score_crop[crop_y1:crop_y2, crop_x1:crop_x2] = score_mid_norm[crop_y1:crop_y2, crop_x1:crop_x2]
                        score_crop = (score_crop * 255).astype(np.uint8)
                        score_crop = cv2.applyColorMap(score_crop, cv2.COLORMAP_JET)
                        show_img = np.copy(oriImg)
                        cv2.putText(show_img, str(limbSeq[k][0]), (int(candA[i][0]), int(candA[i][1])), cv2.FONT_HERSHEY_COMPLEX, 1.0, (0, 255, 0))
                        cv2.putText(show_img, 'score_with_dist_prior ' + str(score_with_dist_prior), (int(candA[i][0]) - 10, int(candA[i][1]) - 10), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0))
                        cv2.putText(show_img, str(limbSeq[k][1]), (int(candB[j][0]), int(candB[j][1])), cv2.FONT_HERSHEY_COMPLEX, 1.0, (0, 255, 0))

                        if criterion1 and criterion2:
                            cv2.arrowedLine(show_img, (int(candA[i][0]), int(candA[i][1])),
                                            (int(candB[j][0]), int(candB[j][1])), (255, 255, 0), 3)
                        else:
                            cv2.arrowedLine(show_img, (int(candA[i][0]), int(candA[i][1])),
                                            (int(candB[j][0]), int(candB[j][1])), (0, 0, 0), 3)

                        for ci in range(0, mid_num):
                            real_x = int(round(startend[ci][0]))
                            real_y = int(round(startend[ci][1]))
                            real_x_next = real_x + int(vec_x[ci] * 20)
                            real_y_next = real_y + int(vec_y[ci] * 20)
                            cv2.arrowedLine(show_img, (real_x, real_y), (real_x_next, real_y_next), (0, 0, 255), 1)

                        show_img = (0.6 * show_img + 0.4 * score_crop).astype(np.uint8)
                        cv2.imshow('show_img', show_img)
                        cv2.waitKey()

                    if criterion1 and criterion2:
                        connection_candidate.append(
                            [i, j, score_with_dist_prior, score_with_dist_prior + candA[i][2] + candB[j][2]])

            connection_candidate = sorted(connection_candidate, key=lambda x: x[2], reverse=True)
            connection = np.zeros((0, 5))
            for c in range(len(connection_candidate)):
                i, j, s = connection_candidate[c][0:3]
                if (i not in connection[:, 3] and j not in connection[:, 4]):
                    connection = np.vstack([connection, [candA[i][3], candB[j][3], s, i, j]])
                    if (len(connection) >= min(nA, nB)):
                        break
            connection_all.append(connection)
        else:
            special_k.append(k)
            connection_all.append([])

    # last number in each row is the total parts number of that person
    # the second last number in each row is the score of the overall configuration
    subset = -1 * np.ones((0, 16))
    candidate = np.array([item for sublist in all_peaks for item in sublist])

    for k in range(len(mapIdx)):
        if k not in special_k:
            partAs = connection_all[k][:, 0]
            partBs = connection_all[k][:, 1]
            indexA, indexB = np.array(limbSeq[k]) - 1

            for i in range(len(connection_all[k])):  # = 1:size(temp,1)
                found = 0
                subset_idx = [-1, -1]
                for j in range(len(subset)):  # 1:size(subset,1):
                    if subset[j][indexA] == partAs[i] or subset[j][indexB] == partBs[i]:
                        subset_idx[found] = j
                        found += 1

                if found == 1:
                    j = subset_idx[0]
                    if (subset[j][indexB] != partBs[i]):
                        subset[j][indexB] = partBs[i]
                        subset[j][-1] += 1
                        subset[j][-2] += candidate[partBs[i].astype(int), 2] + connection_all[k][i][2]
                elif found == 2:  # if found 2 and disjoint, merge them
                    j1, j2 = subset_idx
                    # print "found = 2"
                    membership = ((subset[j1] >= 0).astype(int) + (subset[j2] >= 0).astype(int))[:-2]
                    if len(np.nonzero(membership == 2)[0]) == 0:  # merge
                        subset[j1][:-2] += (subset[j2][:-2] + 1)
                        subset[j1][-2:] += subset[j2][-2:]
                        subset[j1][-2] += connection_all[k][i][2]
                        subset = np.delete(subset, j2, 0)
                    else:  # as like found == 1
                        subset[j1][indexB] = partBs[i]
                        subset[j1][-1] += 1
                        subset[j1][-2] += candidate[partBs[i].astype(int), 2] + connection_all[k][i][2]

                # if find no partA in the subset, create a new subset
                elif not found:
                    row = -1 * np.ones(16)
                    row[indexA] = partAs[i]
                    row[indexB] = partBs[i]
                    row[-1] = 2
                    row[-2] = sum(candidate[connection_all[k][i, :2].astype(int), 2]) + connection_all[k][i][2]
                    subset = np.vstack([subset, row])

    # delete some rows of subset which has few parts occur
    deleteIdx = []
    for i in range(len(subset)):
        if subset[i][-1] < 2 or subset[i][-2] / subset[i][-1] < 0.2:
            deleteIdx.append(i)
    subset = np.delete(subset, deleteIdx, axis=0)

    if param.visual:
        canvas = oriImg.copy()  # B,G,R order
        for i in range(14):
            for j in range(len(all_peaks[i])):
                cv2.circle(canvas, all_peaks[i][j][0:2], 4, colors[i], thickness=-1)

        stickwidth = 4
        for i in range(13):
            for n in range(len(subset)):
                index = subset[n][np.array(limbSeq[i]) - 1]
                if -1 in index:
                    continue
                cur_canvas = canvas.copy()
                Y = candidate[index.astype(int), 0]
                X = candidate[index.astype(int), 1]
                mX = np.mean(X)
                mY = np.mean(Y)
                length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
                angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
                polygon = cv2.ellipse2Poly((int(mY), int(mX)), (int(length / 2), stickwidth), int(angle), 0, 360, 1)
                cv2.fillConvexPoly(cur_canvas, polygon, colors[i])
                canvas = cv2.addWeighted(canvas, 0.4, cur_canvas, 0.6, 0)
        # cv2.imwrite('result.png', canvas)
        cv2.imshow('result', canvas)
        cv2.waitKey(0)

    return candidate, subset