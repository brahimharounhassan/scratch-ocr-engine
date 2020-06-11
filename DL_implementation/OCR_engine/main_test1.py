import cv2
import time
import math
import os
import numpy as np
import tensorflow as tf

import locality_aware_nms as nms_locality
import lanms
from bktree import BKTree, levenshtein, list_words, dict_words

from imutils.object_detection import non_max_suppression


tf.app.flags.DEFINE_string('test_data_path', '/home/unityone/Téléchargements/FOTS_TF-dev/FOTS_TF-dev/test_images', '')
tf.app.flags.DEFINE_string('gpu_list', '0', '')
tf.app.flags.DEFINE_string('checkpoint_path', 'checkpoints/SynthText_6_epochs', '')
tf.app.flags.DEFINE_string('output_dir', 'outputs/', '')
# tf.app.flags.DEFINE_bool('no_write_images', False, 'do not write images')
tf.app.flags.DEFINE_bool('use_vocab', False, 'strong')

# tf.app.flags.DEFINE_bool('use_vocab', True, 'strong, normal or weak')

from module import Backbone_branch, Recognition_branch, RoI_rotate
from data_provider.data_utils import restore_rectangle, ground_truth_to_word
FLAGS = tf.app.flags.FLAGS
detect_part = Backbone_branch.Backbone(is_training=False)
roi_rotate_part = RoI_rotate.RoIRotate()
# recognize_part = Recognition_branch.Recognition(keepProb=1.0,is_training=False)
recognize_part = Recognition_branch.Recognition(is_training=False)
font = cv2.FONT_HERSHEY_SIMPLEX

def get_images():
    '''
    find image files in test data path
    :return: list of files found
    '''
    files = []
    exts = ['jpg', 'png', 'jpeg', 'JPG']
    for parent, dirnames, filenames in os.walk(FLAGS.test_data_path):
        for filename in filenames:
            for ext in exts:
                if filename.endswith(ext):
                    files.append(os.path.join(parent, filename))
                    break
    print('Find {} images'.format(len(files)))
    return files


def resize_image(im, max_side_len=2400):
    '''
    resize image to a size multiple of 32 which is required by the network
    :param im: the resized image
    :param max_side_len: limit of max image size to avoid out of memory in gpu
    :return: the resized image and the resize ratio
    '''
    h, w, _ = im.shape
    # h, w = im.shape
    

    resize_w = w
    resize_h = h

    # limit the max side
    if max(resize_h, resize_w) > max_side_len:
        ratio = float(max_side_len) / resize_h if resize_h > resize_w else float(max_side_len) / resize_w
    else:
        ratio = 1.
    resize_h = int(resize_h * ratio)
    resize_w = int(resize_w * ratio)

    resize_h = resize_h if resize_h % 32 == 0 else (resize_h // 32 - 1) * 32
    resize_w = resize_w if resize_w % 32 == 0 else (resize_w // 32 - 1) * 32
    resize_h = max(32, resize_h)
    resize_w = max(32, resize_w)
    im = cv2.resize(im, (int(resize_w), int(resize_h)))

    ratio_h = resize_h / float(h)
    ratio_w = resize_w / float(w)

    return im, (ratio_h, ratio_w)


def detect(score_map, geo_map, timer, score_map_thresh=0.8, box_thresh=0.1, nms_thres=0.2):
    '''
    restore text boxes from score map and geo map
    :param score_map:
    :param geo_map:
    :param timer:
    :param score_map_thresh: threshhold for score map
    :param box_thresh: threshhold for boxes
    :param nms_thres: threshold for nms
    :return:
    '''
    if len(score_map.shape) == 4:
        score_map = score_map[0, :, :, 0]
        geo_map = geo_map[0, :, :, ]
    # filter the score map
    xy_text = np.argwhere(score_map > score_map_thresh)
    # sort the text boxes via the y axis
    xy_text = xy_text[np.argsort(xy_text[:, 0])]
    # restore
    start = time.time()
    text_box_restored = restore_rectangle(xy_text[:, ::-1]*4, geo_map[xy_text[:, 0], xy_text[:, 1], :]) # N*4*2
    print('{} text boxes before nms'.format(text_box_restored.shape[0]))
    boxes = np.zeros((text_box_restored.shape[0], 9), dtype=np.float32)
    boxes[:, :8] = text_box_restored.reshape((-1, 8))
    boxes[:, 8] = score_map[xy_text[:, 0], xy_text[:, 1]]
    timer['restore'] = time.time() - start
    # nms part
    start = time.time()
    # boxes = nms_locality.nms_locality(boxes.astype(np.float64), nms_thres)
    boxes = lanms.merge_quadrangle_n9(boxes.astype('float32'), nms_thres)
    timer['nms'] = time.time() - start

    if boxes.shape[0] == 0:
        return None, timer

    # here we filter some low score boxes by the average score map, this is different from the orginal paper
    for i, box in enumerate(boxes):
        mask = np.zeros_like(score_map, dtype=np.uint8)
        cv2.fillPoly(mask, box[:8].reshape((-1, 4, 2)).astype(np.int32) // 4, 1)
        boxes[i, 8] = cv2.mean(score_map, mask)[0]
    boxes = boxes[boxes[:, 8] > box_thresh]

    return boxes, timer

def get_project_matrix_and_width(text_polyses, target_height=8.0):
    project_matrixes = []
    box_widths = []
    filter_box_masks = []
    # max_width = 0
    # max_width = 0

    for i in range(text_polyses.shape[0]):
        x1, y1, x2, y2, x3, y3, x4, y4 = text_polyses[i] / 4

        rotated_rect = cv2.minAreaRect(np.array([[x1, y1], [x2, y2], [x3, y3], [x4, y4]]))
        box_w, box_h = rotated_rect[1][0], rotated_rect[1][1]

        if box_w <= box_h:
            box_w, box_h = box_h, box_w

        mapped_x1, mapped_y1 = (0, 0)
        mapped_x4, mapped_y4 = (0, 8)

        width_box = math.ceil(8 * box_w / box_h)
        width_box = int(min(width_box, 128)) # not to exceed feature map's width
        # width_box = int(min(width_box, 512)) # not to exceed feature map's width
        """
        if width_box > max_width: 
            max_width = width_box 
        """
        mapped_x2, mapped_y2 = (width_box, 0)
        # mapped_x3, mapped_y3 = (width_box, 8)

        src_pts = np.float32([(x1, y1), (x2, y2), (x4, y4)])
        dst_pts = np.float32([(mapped_x1, mapped_y1), (mapped_x2, mapped_y2), (mapped_x4, mapped_y4)])
        affine_matrix = cv2.getAffineTransform(dst_pts.astype(np.float32), src_pts.astype(np.float32))
        affine_matrix = affine_matrix.flatten()

        # project_matrix = cv2.getPerspectiveTransform(dst_pts.astype(np.float32), src_pts.astype(np.float32))
        # project_matrix = project_matrix.flatten()[:8]

        project_matrixes.append(affine_matrix)
        box_widths.append(width_box)

    project_matrixes = np.array(project_matrixes)
    box_widths = np.array(box_widths)

    return project_matrixes, box_widths

def sort_poly(p):
    min_axis = np.argmin(np.sum(p, axis=1))
    p = p[[min_axis, (min_axis+1)%4, (min_axis+2)%4, (min_axis+3)%4]]
    if abs(p[0, 0] - p[1, 0]) > abs(p[0, 1] - p[1, 1]):
        return p
    else:
        return p[[0, 3, 2, 1]]
def bktree_search(bktree, pred_word, dist=5):
    return bktree.query(pred_word, dist)

####################################

def remove_image_background(image):
    """
    This function removes the background of an image
    :param image: Original Image
    :return: Image without background
    """
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # edged = canny_edges(gray_img)
    ret, thresh = cv2.threshold(gray_img, 127, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    img_contours = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2]
    img_contours = sorted(img_contours, key=cv2.contourArea)
    for i in img_contours:
        if cv2.contourArea(i) > 15000:
            break
    mask = np.zeros(image.shape[:2], np.uint8)
    cv2.drawContours(mask, [i],-1, 255, -1)
    return cv2.bitwise_and(image, image, mask=mask)


def image_process(image):
    # gray = cv2.fastNlMeansDenoisingColored(image, None, 5,5, 7, 21)
    # gray = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    # kernel = np.ones((3,3),np.uint8)
    # gray = cv2.erode(gray,np.ones((3,3),np.uint8),iterations = 1)
    gray = cv2.morphologyEx(image,cv2.MORPH_TOPHAT,np.ones((31,31),np.uint8))
    
    # gray = cv2.morphologyEx(gray,cv2.MORPH_OPEN,kernel)
    # gray = cv2.bilateralFilter(gray,11,17,17)
	#closed = cv2.GaussianBlur(gray,(3,3),0)
    # thresh = cv2.threshold(gray,128,255,cv2.THRESH_TOZERO + cv2.THRESH_OTSU)[1]
    # filtred = cv2.GaussianBlur(gray,(3,3),0)
	# closed = cv2.blur(closed,(3,3),0)
    # filtred = cv2.bilateralFilter(gray,11,17,17)
    return gray


def myEASTtest():
    default_size = 320
    min_confidence = 0.5
    
    original_image = im.copy()
    (H, W) = im.shape[:2]

    # set the new width and height and then determine the ratio in change
    # for both the width and height
    (newW, newH) = (default_size, default_size)

    rW = W / float(newW)
    rH = H / float(newH)
    print('rW',rW)
    print('rH',rH)

    # resize the image and grab the new image dimensions
    # im_resized2 = cv2.resize(image, (newW, newH))
    (H, W) = im_resized.shape[:2]
    print('W',W)
    print('H',H)
    for tab in boxes:
        boxe = sorted([int(r) for r in tab])
        boxe = list(dict.fromkeys(boxe))
        new_boxe = list()
        old = -1
        for b in boxe :
            if b not in new_boxe and (old + 1) != b:
                new_boxe.append(b)
                old = b
        print("-------------- geometry ",new_boxe)
        # (startX, startY, endX, endY) = np.array(new_boxe)
        # startX = int(startX * rW)
        # startY = int(startY * rH)
        # endX = int(endX * rW)
        # endY = int(endY * rH)
        # print(startX,'-',endX,'**',startY,'-',endY)
        # draw the bounding box on the image
        
        
        #####################################################
        # h,w,_ = im_resized.shape
                
                
####################################

def main(argv=None):
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu_list
    try:
        os.makedirs(FLAGS.output_dir)
    except OSError as e:
        if e.errno != 17:
            raise
        
    if FLAGS.use_vocab and os.path.exists("./vocab.txt"):
        bk_tree = BKTree(levenshtein, list_words('./vocab.txt'))
        # bk_tree = bktree.Tree()
               
    with tf.get_default_graph().as_default():
        input_images = tf.placeholder(tf.float32, shape=[None, None, None, 3], name='input_images')
        input_feature_map = tf.placeholder(tf.float32, shape=[None, None, None, 32], name='input_feature_map')
        input_transform_matrix = tf.placeholder(tf.float32, shape=[None, 6], name='input_transform_matrix')
        input_box_mask = []
        input_box_mask.append(tf.placeholder(tf.int32, shape=[None], name='input_box_masks_0'))
        input_box_widths = tf.placeholder(tf.int32, shape=[None], name='input_box_widths')

        input_seq_len = input_box_widths[tf.argmax(input_box_widths, 0)] * tf.ones_like(input_box_widths)
        global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)

        shared_feature, f_score, f_geometry = detect_part.model(input_images)
        pad_rois = roi_rotate_part.roi_rotate_tensor_pad(input_feature_map, input_transform_matrix, input_box_mask, input_box_widths)
        recognition_logits = recognize_part.build_graph(pad_rois, input_box_widths)
        _, dense_decode = recognize_part.decode(recognition_logits, input_box_widths)

        variable_averages = tf.train.ExponentialMovingAverage(0.997, global_step)
        saver = tf.train.Saver(variable_averages.variables_to_restore())

        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
            ckpt_state = tf.train.get_checkpoint_state(FLAGS.checkpoint_path)
            model_path = os.path.join(FLAGS.checkpoint_path, os.path.basename(ckpt_state.model_checkpoint_path))
            print('Restore from {}'.format(model_path))
            saver.restore(sess, model_path)

            im_fn_list = get_images()
            for im_fn in im_fn_list:
                im = cv2.imread(im_fn)[:, :, ::-1]
                # im = cv2.imread("test_images/95_184_176_66.png")[:, :, ::-1]
                original_image = im.copy()
                start_time = time.time()
                im_resized, (ratio_h, ratio_w) = resize_image(im)                
                # im_resized_d, (ratio_h_d, ratio_w_d) = resize_image_detection(im)

                timer = {'detect': 0, 'restore': 0, 'nms': 0, 'recog': 0}
                start = time.time()
                shared_feature_map, score, geometry = sess.run([shared_feature, f_score, f_geometry], feed_dict={input_images: [im_resized]})
                
                boxes, timer = detect(score_map=score, geo_map=geometry, timer=timer)
##########################################################
                # print(score )
                min_confidence = 0.5
                scores = score
                
                (numRows, numCols) = score.shape[1:3]
                # print(score)
                new_score = list()
                # for y in range(0, 60):
                    # tab = np.array([s[y] for s in score[0,0]])
                data  = np.array([s for s in score[0]])
                first_tab = data[0]
                second_tab = data[0,0]
                # print('first tab',len(first_tab))
                # i_ = 0
                # for y in first_tab:
                #     print(y,"---------",i_)
                #     i_ += 1
                # # print('second tab',i_)
                rects = []
                confidences = []

                # loop over the number of rows
                # for y in range(0, numCols):
                #     # extract the scores (probabilities), followed by the geometrical
                #     # data used to derive potential bounding box coordinates that
                #     # surround text
                #     scoresData = np.array([s[y] for s in score[0,0]])
                #     # scoresData = scores[0, 0, y]                    
                #     xData0 = geometry[0, 0, y]
                #     xData1 = geometry[0, 1, y]
                #     xData2 = geometry[0, 2, y]
                #     xData3 = geometry[0, 3, y]
                #     anglesData = geometry[0, 4, y]

                #     # loop over the number of columns
                #     for x in range(0, numRows):
                #         # if our score does not have sufficient probability, ignore it
                #         if scoresData[x] < min_confidence:
                #             continue

                #         # compute the offset factor as our resulting feature maps will
                #         # be 4x smaller than the input image
                #         (offsetX, offsetY) = (x * 4.0, y * 4.0)

                #         # extract the rotation angle for the prediction and then
                #         # compute the sin and cosine
                #         angle = anglesData[x]
                #         cos = np.cos(angle)
                #         sin = np.sin(angle)

                #         # use the geometry volume to derive the width and height of
                #         # the bounding box
                #         h = xData0[x] + xData2[x]
                #         w = xData1[x] + xData3[x]

                #         # compute both the starting and ending (x, y)-coordinates for
                #         # the text prediction bounding box
                #         endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
                #         endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
                #         startX = int(endX - w)
                #         startY = int(endY - h)

                #         # add the bounding box coordinates and probability score to
                #         # our respective lists
                #         rects.append((startX, startY, endX, endY))
                #         confidences.append(scoresData[x])

                # # apply non-maxima suppression to suppress weak, overlapping bounding
                # # boxes
                
                # boxes = non_max_suppression(np.array(rects), probs=confidences)

                # # loop over the bounding boxes
                # first_detected = dict()
                # for (startX, startY, endX, endY) in boxes:
                #     # scale the bounding box coordinates based on the respective
                #     # ratios
                #     startX = int(startX * rW)
                #     startY = int(startY * rH)
                #     endX = int(endX * rW)
                #     endY = int(endY * rH)
                #     # draw the bounding box on the image
                #     cv2.rectangle(original_image, (startX, startY), (endX, endY), (0, 255, 0), 2)
                # cv2.imshow("Text Detection", original_image)
                # cv2.waitKey(0)
##########################################################
                
                
                timer['detect'] = time.time() - start
                start = time.time() # reset for recognition

                if boxes is not None and boxes.shape[0] != 0:
                    res_file_path = os.path.join(FLAGS.output_dir, 'res_' + '{}.txt'.format(os.path.basename(im_fn).split('.')[0]))

                    input_roi_boxes = boxes[:, :8].reshape(-1, 8)
                    recog_decode_list = []
                    # Here avoid too many text area leading to OOM
                    for batch_index in range(input_roi_boxes.shape[0] // 32 + 1): # test roi batch size is 32
                        start_slice_index = batch_index * 32
                        end_slice_index = (batch_index + 1) * 32 if input_roi_boxes.shape[0] >= (batch_index + 1) * 32 else input_roi_boxes.shape[0]
                        tmp_roi_boxes = input_roi_boxes[start_slice_index:end_slice_index]

                        boxes_masks = [0] * tmp_roi_boxes.shape[0]
                        transform_matrixes, box_widths = get_project_matrix_and_width(tmp_roi_boxes)
                        # max_box_widths = max_width * np.ones(boxes_masks.shape[0]) # seq_len
                    
                        # Run end to end
                        recog_decode = sess.run(dense_decode, feed_dict={input_feature_map: shared_feature_map, input_transform_matrix: transform_matrixes, input_box_mask[0]: boxes_masks, input_box_widths: box_widths})
                        recog_decode_list.extend([r for r in recog_decode])

                    timer['recog'] = time.time() - start
                    # Preparing for draw boxes
                    boxes = boxes[:, :8].reshape((-1, 4, 2))
                    boxes[:, :, 0] /= ratio_w
                    boxes[:, :, 1] /= ratio_h

                    if len(recog_decode_list) != boxes.shape[0]:
                        print("detection and recognition result are not equal!")
                        exit(-1)

                    with open(res_file_path, 'w') as f:
                        for i, box in enumerate(boxes):
                            # to avoid submitting errors
                            box = sort_poly(box.astype(np.int32))
                            if np.linalg.norm(box[0] - box[1]) < 5 or np.linalg.norm(box[3]-box[0]) < 5:
                                continue
                            recognition_result = ground_truth_to_word(recog_decode_list[i])
                           
                            
                            if FLAGS.use_vocab:
                                fix_result = bktree_search(bk_tree, recognition_result.upper())
                                if len(fix_result) != 0:
                                    recognition_result = fix_result[0][1]
			                
                            """
                            f.write('{},{},{},{},{},{},{},{},{}\r\n'.format(
                                box[0, 0], box[0, 1], box[1, 0], box[1, 1], box[2, 0], box[2, 1], box[3, 0], box[3, 1], recognition_result
                            ))
                            """
                            
                            
                            # Draw bounding box
                            cv2.polylines(im[:, :, ::-1], [box.astype(np.int32).reshape((-1, 1, 2))], True, color=(255, 255, 0), thickness=1)
                            # Draw recognition results area
                            text_area = box.copy()
                            text_area[2, 1] = text_area[1, 1]
                            text_area[3, 1] = text_area[0, 1]
                            text_area[0, 1] = text_area[0, 1] - 15
                            text_area[1, 1] = text_area[1, 1] - 15
                            cv2.fillPoly(im[:, :, ::-1], [text_area.astype(np.int32).reshape((-1, 1, 2))], color=(255, 255, 0))
                            im_txt = cv2.putText(im[:, :, ::-1], recognition_result, (box[0, 0], box[0, 1]), font, 0.5, (0, 0, 255), 1)
                            # print(recognition_result)

                            # cv2.rectangle(np.ascontiguousarray(im_txt), (startX, startY), (endX, endY), (0, 255, 0), 2)
                    cv2.imshow('image finale',im_txt)
                    cv2.waitKey(0)
                else:
                    res_file = os.path.join(FLAGS.output_dir, 'res_' + '{}.txt'.format(os.path.basename(im_fn).split('.')[0]))
                    f = open(res_file, "w")
                    im_txt = None
                    f.close()

                print('{} : detect {:.0f}ms, restore {:.0f}ms, nms {:.0f}ms, recog {:.0f}ms'.format(
                    im_fn, timer['detect']*1000, timer['restore']*1000, timer['nms']*1000, timer['recog']*1000))

                duration = time.time() - start_time
                print('[timing] {}'.format(duration))



                # if not FLAGS.no_write_images:
                #     img_path = os.path.join(FLAGS.output_dir, os.path.basename(im_fn))
                #     cv2.imwrite(img_path, im[:, :, ::-1])
                #     if im_txt is not None:
                #         cv2.imwrite(img_path, im_txt)


if __name__ == '__main__':
    tf.app.run()