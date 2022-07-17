from PIL import Image,ImageOps
import numpy as np
from torch.nn import DataParallel
import cv2
from torchvision.transforms import functional as F
import torchvision.transforms as transforms
import torch
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import os

from model.CBAM import CBAMResNet
cudnn.benchmark = True
from omegaconf import OmegaConf



def load_imge_for_scoremodel(img_path):
    trans = gettransform()
    img = trans(Image.open(img_path)).unsqueeze(0)
    return img

def load_image(img_path,cropmodel,crop_size,iscrop):
    '''

    :param img_path:
    :return: shape: 1,128,128
    '''
    image = cv2.imread(str(img_path), 0)
    if image is None:
        return None
    if iscrop:
        image = image.resize(crop_size)

    image = np.dstack((image, np.fliplr(image)))
    return image
# numpy
def cosin_metric_np(x1, x2):
    return np.dot(x1, x2) / (np.linalg.norm(x1) * np.linalg.norm(x2))

def gettransform():
    transform = transforms.Compose([
            transforms.Resize(112, 112),
            transforms.ToTensor(),  # range [0, 255] -> [0.0,1.0]
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))  # range [0.0, 1.0] -> [-1.0,1.0]
        ])
    return transform



def extractDeepFeature(img_path, model,trans,device):
    # image = cv2.imread(str(img_path))
    image = Image.open(img_path).convert('RGB')
    imageflp = trans(ImageOps.flip(image)).unsqueeze(0).to(device)
    image = trans(image).unsqueeze(0).to(device)
    output = model(image).data.cpu().numpy()
    outputflp = model(imageflp).data.cpu().numpy()
    feature = np.concatenate((output, outputflp), 1)
    return feature[0]


def KFold(n=6000, n_folds=10):
    folds = []
    base = list(range(n))
    for i in range(n_folds):
        test = base[int(i * n / n_folds):int((i + 1) * n / n_folds)]
        train = list(set(base) - set(test))
        folds.append([train, test])
    return folds

def eval_acc_index(threshold, diff,index,flagindex):
    y_true = []
    y_predict = []
    for d in diff:
        same = 1 if float(d[index]) > threshold else 0
        y_predict.append(same)
        y_true.append(int(d[flagindex]))
    y_true = np.array(y_true)
    y_predict = np.array(y_predict)
    accuracy = 1.0 * np.count_nonzero(y_true == y_predict) / len(y_true)
    return accuracy

def find_best_threshold_index(thresholds, predicts,index,flagindex):
    best_threshold = best_acc = 0
    for threshold in thresholds:
        accuracy = eval_acc_index(threshold, predicts,index,flagindex)
        
        if accuracy >= best_acc:
            best_acc = accuracy
            best_threshold = threshold
    return best_threshold


def computeaccuracy(result_path):
    
    accuracy = []
    thd = []
    predicts = np.loadtxt(result_path,delimiter='%s',dtype=np.str)
    predicts = np.array(list(map(lambda line: line.strip('\n').split(), predicts)))
    folds = KFold(n=len(predicts), n_folds=10)
    count = len(predicts[0])
    for i in range(count-1):
        thresholds = np.arange(-1.0, 1.0, 0.005)
        for idx, (train, test) in enumerate(folds):
            best_thresh =find_best_threshold_index(thresholds, predicts[train],i,-1)
            accuracy.append(eval_acc_index(best_thresh, predicts[test],i,-1))
            thd.append(best_thresh)
        print('LFWACC={:.4f} std={:.4f} thd={:.4f}'.format(np.mean(accuracy), np.std(accuracy), np.mean(thd)))
#     return np.mean(accuracy)


def evalssublist(model,root_dir,dir1,dir2list,pair_path,savename,device):
    trans = gettransform()
    predicts = []
    with open(pair_path) as f:
        pairs_lines = f.readlines()[1:]
        print("linenumber:", len(pairs_lines))

    with torch.no_grad():
        for i in range(len(pairs_lines)):
            p = pairs_lines[i].replace('\n', '').split('\t')

            if 3 == len(p):
                sameflag = 1
                name1 = p[0] + '/' + p[0] + '_' + '{:04}.jpg'.format(int(p[1]))
                name2 = p[0] + '/' + p[0] + '_' + '{:04}.jpg'.format(int(p[2]))
            elif 4 == len(p):
                sameflag = 0
                name1 = p[0] + '/' + p[0] + '_' + '{:04}.jpg'.format(int(p[1]))
                name2 = p[2] + '/' + p[2] + '_' + '{:04}.jpg'.format(int(p[3]))
            else:
                continue
                # raise ValueError("WRONG LINE IN 'pairs.txt! ")

            dirnames = dir2list
            evallist = []
            img1path = os.path.join(os.path.join(root_dir,dir1), name1)
            if os.path.exists(img1path):
                for i in range(0, len(dirnames)):
                    dirname = os.path.join(root_dir, dirnames[i])
                    img2path = os.path.join(dirname, name2)
                    if os.path.exists(img2path):
                        evallist.append(img2path)
                    else:
                        continue
            else:
                continue

            if len(evallist)!=len(dirnames):
                continue
            distances = []
            f1 = extractDeepFeature(img1path, model, trans, device)
            for i in range(0, len(dirnames)):
                print(evallist[i])
                fn = extractDeepFeature(evallist[i], model, trans, device)
                distances.append(cosin_metric_np(fn, f1))
            strformat=''
            for i in range(0,len(distances)):
                strformat += '\t'+ str(distances[i])
            strformat+='\t'+str(sameflag)
            predicts.append(strformat)
            # print(strformat)

    predicts = np.array(list(map(lambda line: line.strip('\n').split(), predicts)))
    save_path = os.path.join(root_dir, savename)
    np.savetxt(save_path, predicts, '%s')
    # return predicts

def computeSimilarity(result_path):
    samepredictlist = []
    crosspredictlist = []
    predicts = np.loadtxt(result_path, delimiter='%s', dtype=np.str)
    predicts = np.array(list(map(lambda line: line.strip('\n').split(), predicts)))
    for p in predicts:
        if int(p[-1])==1:
            samepredictlist.append(p[0:-1])
        elif int(p[-1])==0:
            crosspredictlist.append(p[0:-1])
    samearr=np.array(samepredictlist,dtype=np.float)
    samemean = np.mean(samearr,axis=0)
    crossarr = np.array(crosspredictlist,dtype=np.float)
    crossmean = np.mean(crossarr,axis=0)
    gap = samemean-crossmean
    for i in range(len(samemean)):
        print(i,'-th mean similarity: same/cross/gap={:.4f}/{:.4f}/{:.4f}'.format(samemean[i],crossmean[i],gap[i]))
    return samemean,crossmean,gap

def generateSimilaritytxt(rootdir,dir1,dir2list,pairpath,savename):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    recognizer = CBAMResNet(50, feature_dim=256, mode='ir') # resnet18
    # recognizer = DataParallel(recognizer)
    recognizer.load_state_dict(torch.load('./arcface/Resnet50_70000_net.pth', map_location=device))

    recognizer.to(device)
    recognizer.eval()

    evalssublist(recognizer, rootdir, dir1, dir2list, pairpath,savename, device)

if __name__ == '__main__':
    rootdir = 'data/LFWs/'
    dir1 = 'data/LFWs/lfw_112/'
    dir2_list = ['mouth/mouth_ca_fill', 'left_eye/left_eye_ca_fill',  'right_eye/right_eye_ca_fill', 'nose/nose_ca_fill',
                 'left_face/left_face_ca_fill',  'right_face/right_face_ca_fill',
                 'up_face/up_face_ca_fill', 'down_face/down_face_ca_fill', 'twoeyes/twoeyes_ca_fill']
    savename = 'ca_similarity.txt'
    # dir2_list = ['nose/nose_occ', 'nose/nose_white',  'nose/nose_black', 'nose/nose_gray', 'nose/nose_pic_fill', 'nose/nose_id_fill_1']
    # savename = 'nose_similarity.txt'
    # dir2_list = ['right_face/right_face_occ', 'right_face/right_face_white',  'right_face/right_face_black', 'right_face/right_face_gray', 'right_face/right_face_pic_fill', 'right_face/right_face_id_fill_1']
    # savename = 'right_face_similarity.txt'
    pairpath = 'data/LFWs/pairs.txt'
    generateSimilaritytxt(rootdir,dir1,dir2_list,pairpath,savename)
    computeSimilarity(os.path.join(rootdir,savename)) # compute the similarity scroe according to the
    # computeaccuracy(os.path.join(rootdir,savename))
