import cv2
import torch
import dlib
from PIL import Image
import numpy as np
import random


MASKS = ['center','full_face','d_eye','mouse']
RE_MASKS = ['left_eye','right_eye','left_face','right_face','top_face','bottom_face']
OCCLUSIONS = ['white','black','object','g_fill','id_fill']


def get_mask(img,typestr='center'):
    return

def get_occlusionOut(img,mask,otype='white'):
    if otype=='white':
        return img*mask+(1-mask)*255
    elif otype=='black':
        return img*mask



def get_randommask(img_file,detector,predictor):
    idx = random.randrange(len(RE_MASKS))
    return get_remask(img_file, detector, predictor, typestr=RE_MASKS[idx])    # random place of white,black,object
    # return random_mask

def get_remask(img_file,detector,predictor,typestr='left_eye'):
    if typestr=='left_eye':
        return get_eye_Mask(img_file,detector,predictor,type = 'left')
    elif typestr =='right_eye':
        return get_eye_Mask(img_file,detector,predictor,type = 'right')
    elif typestr == 'left_face':
        return occlusion_mask(img_file,detector,predictor,mtype = 'left')
    elif typestr=='right_face':
        return occlusion_mask(img_file,detector,predictor,mtype = 'right')
    elif typestr=='top_face':
        return occlusion_mask(img_file,detector,predictor,mtype = 'top')
    elif typestr=='bottom_face':
        return occlusion_mask(img_file,detector,predictor,mtype = 'bottom')
    else:
        print("wrong mask type")

    return

#size: 1,1,256,256
def center_mask(img):
    print(img.size())
    """Generates a center hole with 1/4*W and 1/4*H"""
    mask = torch.ones_like(img)
    size = img.size()
    x = int(size[2] / 4)
    y = int(size[3] / 4)
    range_x = int(size[2] * 3 / 4)
    range_y = int(size[3] * 3 / 4)
    mask[:,:, x:range_x, y:range_y] = 0

    return mask

def center_mask_size(size):
    #     (3,256,256)
    """Generates a center hole with 1/4*W and 1/4*H"""
    mask = torch.ones(size)
    x = int(size[1] / 4)
    y = int(size[2] / 4)
    range_x = int(size[1] * 3 / 4)
    range_y = int(size[2] * 3 / 4)
    if len(size) == 3:
        mask[:, x:range_x, y:range_y] = 0
    else:
        mask[:, :, x:range_x, y:range_y] = 0

    return mask

#mask = face_mask(args.target_image,detector,predictor).unsqueeze(0).to(device)
def get_eye_Mask(img_file,detector,predictor,type = 'two'):
    '''

    :param img_file: input image path, size(3,256,256)
    :param detector:
    :param predictor:
    :param type: ['left','right','two']
    :return: mask:size(3,256,256)
    '''
    img = dlib.load_rgb_image(img_file)
    dets = detector(img, 1)
    if len(dets) < 0:
        print("no face landmark detected")
        return
    else:
        try:
            shape = predictor(img, dets[0])
        except:
            return    
        landmarks = shape.parts()
        image_size = img.shape
        mask = np.full(image_size[0:2] + (1,), 1, dtype=np.float32)
    
        left_eye_x = int((landmarks[39].x+landmarks[36].x)/2)
        left_eye_y = int((landmarks[37].y+landmarks[38].y+landmarks[41].y+landmarks[42].y)/4)
        left_eye_w = int(landmarks[39].x-landmarks[36].x)
        left_eye_h = int(landmarks[40].y-landmarks[38].y)
        right_eye_x = int((landmarks[42].x + landmarks[45].x) / 2)
        right_eye_y = int((landmarks[43].y + landmarks[44].y + landmarks[47].y + landmarks[46].y) / 4)
        right_eye_w = int(landmarks[45].x - landmarks[42].x)
        right_eye_h = int(landmarks[46].y - landmarks[44].y)
        current_eye_pos = [left_eye_x,left_eye_y,left_eye_h,left_eye_w,right_eye_x,right_eye_y,right_eye_h,right_eye_w]
        if type=='left' or type=='two':
            #left eye x
            scale = current_eye_pos[0] - 30 #current_eye_pos[3] / 2
            down_scale = current_eye_pos[0] + 30 #current_eye_pos[3] / 2
            l1_1 =int(scale)
            u1_1 =int(down_scale)
                #y
            scale = current_eye_pos[1] - 25 #current_eye_pos[2] / 2 old :15
            down_scale = current_eye_pos[1] + 20 #current_eye_pos[2] / 2  old :15
            l1_2 = int(scale)
            u1_2 = int(down_scale)

            mask[l1_2:u1_2,l1_1:u1_1,:] = 0

        if type == 'right' or type == 'two':
            #right eye, x
            scale = current_eye_pos[4] - 30 #current_eye_pos[7] / 2
            down_scale = current_eye_pos[4] + 30 #current_eye_pos[7] / 2
            l2_1 = int(scale)
            u2_1 = int(down_scale)
            #y
            scale = current_eye_pos[5] - 25 #current_eye_pos[6] / 2  old :15
            down_scale = current_eye_pos[5] + 20 #current_eye_pos[6] / 2  old :15
            l2_2 = int(scale)
            u2_2 = int(down_scale)
            mask[l2_2:u2_2,l2_1:u2_1, :] = 0
        mask = torch.from_numpy(mask.transpose((2, 0, 1)))

        return mask

def get_mouth_Mask(img_file,detector,predictor):
    
    img = dlib.load_rgb_image(img_file)
    dets = detector(img, 1)
    if len(dets) < 0:
        print("no face landmark detected")
        return
    else:
        try:
            shape = predictor(img, dets[0])
        except:
            return    
        landmarks = shape.parts()
        image_size = img.shape
        mask = np.full(image_size[0:2] + (1,), 1, dtype=np.float32)
    
        left_mouth_x = int((landmarks[48].x+landmarks[54].x)/2)
        left_mouth_y = int((landmarks[50].y+landmarks[52].y+landmarks[57].y)/3)
        left_mouth_w = int(landmarks[54].x-landmarks[48].x)
        left_mouth_h = int(landmarks[57].y-left_mouth_y)*2
        
        current_pos = [left_mouth_x,left_mouth_y,left_mouth_h,left_mouth_w]
        #left eye x
        scale = current_pos[0] - (left_mouth_w/2)-0 #current_eye_pos[3] / 2
        down_scale = current_pos[0] + (left_mouth_w/2)+0 #current_eye_pos[3] / 2
        l1_1 =int(scale)
        u1_1 =int(down_scale)
            #y
        scale = current_pos[1] - (left_mouth_h/2)-5 #current_eye_pos[2] / 2
        down_scale = current_pos[1] + (left_mouth_h/2)+5 #current_eye_pos[2] / 2
        l1_2 = int(scale)
        u1_2 = int(down_scale)

        mask[l1_2:u1_2,l1_1:u1_1,:] = 0
       
        mask = torch.from_numpy(mask.transpose((2, 0, 1)))

        return mask
    
    
#mask = face_mask(args.target_image,detector,predictor).unsqueeze(0).to(device)
def face_mask(test_img_path,detector,predictor):
    img = cv2.imread(test_img_path)                        
    # Take grayscale                                       
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)       
    # Face count rects (rectangles)                        
    rects = detector(img_gray, 0) 
    
    if len(rects) > 0:
        # landmarks = predictor(img, rects[0])                 
        landmarks = predictor(img, rects[0]).parts()           
        landmarks = np.matrix([[p.x,p.y] for p in landmarks])  
        # 画出关键点points = []                                     
        pts = np.array(landmarks, np.uint8)# for index, pt in e
        mask = get_image_hull_mask(img.shape,pts)            
        img = torch.from_numpy(mask.transpose((2, 0, 1)))
        return img
    
    return center_mask_size((3,256,256))

#获取face mask
def get_image_hull_mask(image_shape, image_landmarks, ie_polys=None):
    # get the mask of the image
    # print(image_landmarks.shape)
    if image_landmarks.shape[0] != 68:
        raise Exception(
            'get_image_hull_mask works only with 68 landmarks')
    int_lmrks = np.array(image_landmarks, dtype=np.int)
    hull_mask = np.full(image_shape[0:2] + (1,), 1, dtype=np.float32)
    for i in range(0,6):
        int_lmrks[i][0]+=10
    for i in range(5,12):
        int_lmrks[i][1]-=10
    for i in range(11,17):
        int_lmrks[i][0]-=10
    for i in range(17,21):
        int_lmrks[i][1]-=5
    for i in range(22,26):
        int_lmrks[i][1]-=5
    cv2.fillConvexPoly(hull_mask, cv2.convexHull(
        np.concatenate((int_lmrks[0:9],
                        int_lmrks[17:18]))), (0,))

    cv2.fillConvexPoly(hull_mask, cv2.convexHull(
        np.concatenate((int_lmrks[8:17],
                        int_lmrks[26:27]))), (0,))

    cv2.fillConvexPoly(hull_mask, cv2.convexHull(
        np.concatenate((int_lmrks[17:20],
                        int_lmrks[8:9]))), (0,))

    cv2.fillConvexPoly(hull_mask, cv2.convexHull(
        np.concatenate((int_lmrks[24:27],
                        int_lmrks[8:9]))), (0,))

    cv2.fillConvexPoly(hull_mask, cv2.convexHull(
        np.concatenate((int_lmrks[19:25],
                        int_lmrks[8:9],
                        ))), (0,))

    cv2.fillConvexPoly(hull_mask, cv2.convexHull(
        np.concatenate((int_lmrks[17:22],
                        int_lmrks[27:28],
                        int_lmrks[31:36],
                        int_lmrks[8:9]
                        ))), (0,))

    cv2.fillConvexPoly(hull_mask, cv2.convexHull(
        np.concatenate((int_lmrks[22:27],
                        int_lmrks[27:28],
                        int_lmrks[31:36],
                        int_lmrks[8:9]
                        ))), (0,))

    # nose
    cv2.fillConvexPoly(
        hull_mask, cv2.convexHull(int_lmrks[27:36]), (0,))

    if ie_polys is not None:
        ie_polys.overlay_mask(hull_mask)
    # print()
    return hull_mask



#mask = occlusion_mask(args.target_image,detector,predictor).unsqueeze(0).to(device)
# for face occlusion mask generation, with eyes region retained.
def occlusion_mask(test_img_path,detector,predictor,mtype = 'bottom'):
    img = cv2.imread(test_img_path)                        
    # Take grayscale                                       
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)       
    # Face count rects (rectangles)                        
    rects = detector(img_gray, 0) 
    
    if len(rects) > 0:
        # landmarks = predictor(img, rects[0])                 
        landmarks = predictor(img, rects[0]).parts()           
        landmarks = np.matrix([[p.x,p.y] for p in landmarks])  
        # 画出关键点points = []                                     
        pts = np.array(landmarks, np.uint8)# for index, pt in e
        mask = get_image_occlusion_mask(img.shape,pts,type=mtype)
        img = torch.from_numpy(mask.transpose((2, 0, 1)))
        return img
    else:
        print("landmark detection failed!")
        
        return None

#获取occlusion mask
def get_image_occlusion_mask(image_shape, image_landmarks,type):
    # get the mask of the image
    # print(image_landmarks.shape)
    if image_landmarks.shape[0] != 68:
        raise Exception(
            'get_image_hull_mask works only with 68 landmarks')
    int_lmrks = np.array(image_landmarks, dtype=np.int)
    hull_mask = np.full(image_shape[0:2] + (1,), 1, dtype=np.float32)
    for i in range(17,21):
        int_lmrks[i][1]-=5
    for i in range(22,26):
        int_lmrks[i][1]-=5
    if type=='top':
        cv2.fillConvexPoly(
            hull_mask, cv2.convexHull(
                np.concatenate((int_lmrks[0:3],int_lmrks[30:31],int_lmrks[14:17],int_lmrks[17:20] ))), (0,))
        cv2.fillConvexPoly(
            hull_mask,cv2.convexHull(
                np.concatenate((int_lmrks[0:1],int_lmrks[17:20],int_lmrks[24:27],int_lmrks[16:17],int_lmrks[29:30]))), (0,))
    elif type=='bottom':
        cv2.fillConvexPoly(
            hull_mask, cv2.convexHull(int_lmrks[1:16]), (0,))
    elif type=='left':
        cv2.fillConvexPoly(
            hull_mask, cv2.convexHull(
                np.concatenate((int_lmrks[0:9], int_lmrks[27:28]))), (0,))
        cv2.fillConvexPoly(
            hull_mask, cv2.convexHull(
                np.concatenate((int_lmrks[0:1], int_lmrks[17:20],[[int_lmrks[27][0],int_lmrks[19][1]]],int_lmrks[8:9]))), (0,))
    elif type == 'right':
        cv2.fillConvexPoly(
            hull_mask, cv2.convexHull(
                np.concatenate((int_lmrks[8:17], int_lmrks[27:28]))), (0,))
        cv2.fillConvexPoly(
            hull_mask, cv2.convexHull(
                np.concatenate(
                    ([[int_lmrks[27][0], int_lmrks[19][1]]],int_lmrks[24:27], int_lmrks[16:17], int_lmrks[8:9]))), (0,))
    else:
        print("wrong type input")
    return hull_mask


def initPredictor(modelpath):
    
    predictor_model = modelpath          
    detector = dlib.get_frontal_face_detector()# dlib face detector    
    predictor = dlib.shape_predictor(predictor_model) 
    return detector,predictor




