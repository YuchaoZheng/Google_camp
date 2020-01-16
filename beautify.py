import sys,os,traceback
import cv2
import dlib
import numpy as np
import argparse
import copy
import pdb
import math

epsilon = 1e-5


class Organ():
    def __init__(self,im_bgr,im_hsv,temp_bgr,temp_hsv,landmark,name,ksize=None):
        self.im_bgr,self.im_hsv,self.landmark,self.name=im_bgr,im_hsv,landmark,name
        self.get_rect()
        self.shape=(int(self.bottom-self.top),int(self.right-self.left))
        self.size=self.shape[0]*self.shape[1]*3
        self.move=int(np.sqrt(self.size/3)/20)
        self.ksize=self.get_ksize()
        self.patch_bgr,self.patch_hsv=self.get_patch(self.im_bgr),self.get_patch(self.im_hsv)
        self.set_temp(temp_bgr,temp_hsv)
        self.patch_mask=self.get_mask_re()
        pass
    
    def set_temp(self,temp_bgr,temp_hsv):
        self.im_bgr_temp,self.im_hsv_temp=temp_bgr,temp_hsv
        self.patch_bgr_temp,self.patch_hsv_temp=self.get_patch(self.im_bgr_temp),self.get_patch(self.im_hsv_temp)

    def confirm(self):
        self.im_bgr[:],self.im_hsv[:]=self.im_bgr_temp[:],self.im_hsv_temp[:]
    
    def update_temp(self):
        self.im_bgr_temp[:],self.im_hsv_temp[:]=self.im_bgr[:],self.im_hsv[:]
        
    def get_ksize(self,rate=15):
        size=max([int(np.sqrt(self.size/3)/rate),1])
        size=(size if size%2==1 else size+1)
        return (size,size)
        
    def get_rect(self):
        ys,xs=self.landmark[:,1],self.landmark[:,0]
        self.top,self.bottom,self.left,self.right=np.min(ys),np.max(ys),np.min(xs),np.max(xs)

    def get_patch(self,im):
        shape=im.shape
        return im[np.max([self.top-self.move,0]):np.min([self.bottom+self.move,shape[0]]),np.max([self.left-self.move,0]):np.min([self.right+self.move,shape[1]])]

    def _draw_convex_hull(self,im, points, color):
        points = cv2.convexHull(points)
        cv2.fillConvexPoly(im, points, color=color)
        
    def get_mask_re(self,ksize=None):
        if ksize==None:
            ksize=self.ksize
        landmark_re=self.landmark.copy()
        landmark_re[:,1]-=np.max([self.top-self.move,0])
        landmark_re[:,0]-=np.max([self.left-self.move,0])
        mask = np.zeros(self.patch_bgr.shape[:2], dtype=np.float64)
    
        self._draw_convex_hull(mask,
                         landmark_re,
                         color=1)
    
        mask = np.array([mask, mask, mask]).transpose((1, 2, 0))
    
        mask = (cv2.GaussianBlur(mask, ksize, 0) > 0) * 1.0
        return cv2.GaussianBlur(mask, ksize, 0)[:]
        
    def get_mask_abs(self,ksize=None):
        if ksize==None:
            ksize=self.ksize
        mask = np.zeros(self.im_bgr.shape, dtype=np.float64)
        patch=self.get_patch(mask)
        patch[:]=self.patch_mask[:]
        return mask
        
    def whitening(self,rate=0.15,confirm=True):
        if confirm:
            self.confirm()
            self.patch_hsv[:,:,-1]=np.minimum(self.patch_hsv[:,:,-1]+self.patch_hsv[:,:,-1]*self.patch_mask[:,:,-1]*rate,255).astype('uint8')
            self.im_bgr[:]=cv2.cvtColor(self.im_hsv, cv2.COLOR_HSV2BGR)[:]
            self.update_temp()
        else:
            self.patch_hsv_temp[:]=cv2.cvtColor(self.patch_bgr_temp, cv2.COLOR_BGR2HSV)[:]
            self.patch_hsv_temp[:,:,-1]=np.minimum(self.patch_hsv_temp[:,:,-1]+self.patch_hsv_temp[:,:,-1]*self.patch_mask[:,:,-1]*rate,255).astype('uint8')
            self.patch_bgr_temp[:]=cv2.cvtColor(self.patch_hsv_temp, cv2.COLOR_HSV2BGR)[:]
            
    def brightening(self,rate=0.3,confirm=True):
        patch_mask=self.get_mask_re((1,1))
        if confirm:
            self.confirm()
            patch_new=self.patch_hsv[:,:,1]*patch_mask[:,:,1]*rate
            patch_new=cv2.GaussianBlur(patch_new,(3,3),0)
            self.patch_hsv[:,:,1]=np.minimum(self.patch_hsv[:,:,1]+patch_new,255).astype('uint8')
            self.im_bgr[:]=cv2.cvtColor(self.im_hsv, cv2.COLOR_HSV2BGR)[:]
            self.update_temp()
        else:
            self.patch_hsv_temp[:]=cv2.cvtColor(self.patch_bgr_temp, cv2.COLOR_BGR2HSV)[:]
            patch_new=self.patch_hsv_temp[:,:,1]*patch_mask[:,:,1]*rate
            patch_new=cv2.GaussianBlur(patch_new,(3,3),0)
            self.patch_hsv_temp[:,:,1]=np.minimum(self.patch_hsv[:,:,1]+patch_new,255).astype('uint8')
            self.patch_bgr_temp[:]=cv2.cvtColor(self.patch_hsv_temp, cv2.COLOR_HSV2BGR)[:]
        
    def smooth(self,rate=0.6,ksize=None,confirm=True):
        if ksize==None:
            ksize=self.get_ksize(80)
        index=self.patch_mask>0
        if confirm:
            self.confirm()
            patch_new=cv2.GaussianBlur(cv2.bilateralFilter(self.patch_bgr,3,*ksize),ksize,0)
            self.patch_bgr[index]=np.minimum(rate*patch_new[index]+(1-rate)*self.patch_bgr[index],255).astype('uint8')
            self.im_hsv[:]=cv2.cvtColor(self.im_bgr, cv2.COLOR_BGR2HSV)[:]
            self.update_temp()
        else:
            patch_new=cv2.GaussianBlur(cv2.bilateralFilter(self.patch_bgr_temp,3,*ksize),ksize,0)
            self.patch_bgr_temp[index]=np.minimum(rate*patch_new[index]+(1-rate)*self.patch_bgr_temp[index],255).astype('uint8')
            self.patch_hsv_temp[:]=cv2.cvtColor(self.patch_bgr_temp, cv2.COLOR_BGR2HSV)[:]
        
    def sharpen(self,rate=0.3,confirm=True):
        patch_mask=self.get_mask_re((3,3))
        kernel = np.zeros( (9,9), np.float32)
        kernel[4,4] = 2.0   #Identity, times two! 
        #Create a box filter:
        boxFilter = np.ones( (9,9), np.float32) / 81.0
        
        #Subtract the two:
        kernel = kernel - boxFilter
        index=patch_mask>0
        if confirm:
            self.confirm()
            sharp=cv2.filter2D(self.patch_bgr,-1,kernel)
            self.patch_bgr[index]=np.minimum(((1-rate)*self.patch_bgr)[index]+sharp[index]*rate,255).astype('uint8')
            self.update_temp()
        else:
            sharp=cv2.filter2D(self.patch_bgr_temp,-1,kernel)
            self.patch_bgr_temp[:]=np.minimum(self.patch_bgr_temp+self.patch_mask*sharp*rate,255).astype('uint8')
            self.patch_hsv_temp[:]=cv2.cvtColor(self.patch_bgr_temp, cv2.COLOR_BGR2HSV)[:]


class Forehead(Organ):
    def __init__(self,im_bgr,im_hsv,temp_bgr,temp_hsv,landmark,mask_organs,name,ksize=None):
        self.mask_organs=mask_organs
        super(Forehead,self).__init__(im_bgr,im_hsv,temp_bgr,temp_hsv,landmark,name,ksize)
    
    def get_mask_re(self,ksize=None):
        if ksize==None:
            ksize=self.ksize
        landmark_re=self.landmark.copy()
        landmark_re[:,1]-=np.max([self.top-self.move,0])
        landmark_re[:,0]-=np.max([self.left-self.move,0])
        mask = np.zeros(self.patch_bgr.shape[:2], dtype=np.float64)
    
        self._draw_convex_hull(mask,
                         landmark_re,
                         color=1)
        
        mask = np.array([mask, mask, mask]).transpose((1, 2, 0))
    
        mask = (cv2.GaussianBlur(mask, ksize, 0) > 0) * 1.0
        patch_organs=self.get_patch(self.mask_organs)
        mask= cv2.GaussianBlur(mask, ksize, 0)[:]
        mask[patch_organs>0]=(1-patch_organs[patch_organs>0])
        return mask
        
def dist(x1, y1, x2, y2):
    return ((x1 - x2) ** 2) + ((y1 - y2) ** 2)


def bilinear_ins(src_img, x, y):
    width, height, _ = src_img.shape

    x = max(min(x, width - 2), 0)
    y = max(min(y, height - 2), 0)
    x1 = int(x)
    y1 = int(y)
    x2 = min(width - 1, x1 + 1)
    y2 = min(height - 1, y1 + 1)

    value1 = src_img[x1, y1].astype(np.float32) * (x2 - x) * (y2 - y)
    value2 = src_img[x1, y2].astype(np.float32) * (x2 - x) * (y - y1)
    value3 = src_img[x2, y1].astype(np.float32) * (x - x1) * (y2 - y)
    value4 = src_img[x2, y2].astype(np.float32) * (x - x1) * (y - y1)

    value = np.minimum(value1 + value2 + value3 + value4, np.array([255]))
    return value.astype(np.uint8)


class Face(Organ):
    def __init__(self,im_bgr,img_hsv,temp_bgr,temp_hsv,landmarks,index):
        self.index=index

        self.organs_name=['jaw','mouth','nose','left eye','right eye','left brow','right brow']
        
        self.organs_points=[list(range(0, 17)),list(range(48, 61)),list(range(27, 35)),list(range(42, 48)),list(range(36, 42)),list(range(22, 27)),list(range(17, 22))]

        self.organs={name:Organ(im_bgr,img_hsv,temp_bgr,temp_hsv,landmarks[points],name) for name,points in zip(self.organs_name,self.organs_points)}

        mask_nose=self.organs['nose'].get_mask_abs()
        mask_organs=(self.organs['mouth'].get_mask_abs()+mask_nose+self.organs['left eye'].get_mask_abs()+self.organs['right eye'].get_mask_abs()+self.organs['left brow'].get_mask_abs()+self.organs['right brow'].get_mask_abs())
        forehead_landmark=self.get_forehead_landmark(im_bgr,landmarks,mask_organs,mask_nose)
        self.organs['forehead']=Forehead(im_bgr,img_hsv,temp_bgr,temp_hsv,forehead_landmark,mask_organs,'forehead')
        mask_organs+=self.organs['forehead'].get_mask_abs()

        self.FACE_POINTS = np.concatenate([landmarks,forehead_landmark])
        super(Face,self).__init__(im_bgr,img_hsv,temp_bgr,temp_hsv,self.FACE_POINTS,'face')

        mask_face=self.get_mask_abs()-mask_organs
        self.patch_mask=self.get_patch(mask_face)
        pass
        
    
    def get_forehead_landmark(self,im_bgr,face_landmark,mask_organs,mask_nose):
        radius=(np.linalg.norm(face_landmark[0]-face_landmark[16])/2).astype('int32')
        center_abs=tuple(((face_landmark[0]+face_landmark[16])/2).astype('int32'))
        
        angle=np.degrees(np.arctan((lambda l:l[1]/l[0])(face_landmark[16]-face_landmark[0]))).astype('int32')
        mask=np.zeros(mask_organs.shape[:2], dtype=np.float64)
        cv2.ellipse(mask,center_abs,(radius,radius),angle,180,360,1,-1)

        mask[mask_organs[:,:,0]>0]=0
        
        index_bool=[]
        for ch in range(3):
            mean,std=np.mean(im_bgr[:,:,ch][mask_nose[:,:,ch]>0]),np.std(im_bgr[:,:,ch][mask_nose[:,:,ch]>0])
            up,down=mean+0.5*std,mean-0.5*std
            index_bool.append((im_bgr[:,:,ch]<down)|(im_bgr[:,:,ch]>up))
        index_zero=((mask>0)&index_bool[0]&index_bool[1]&index_bool[2])
        mask[index_zero]=0
        index_abs=np.array(np.where(mask>0)[::-1]).transpose()
        landmark=cv2.convexHull(index_abs).squeeze()
        return landmark
    
    def local_transform(self, from_y, from_x, to_y, to_x, r):
        width, height, _ = self.im_bgr.shape

        src_img = copy.deepcopy(self.im_bgr)
        sqar_r = r * r
        sqar_dis_trans = dist(from_x, from_y, to_x, to_y)

        for i in range(width):
            for j in range(height):
                if abs(i - from_x) >= r or abs(j - from_y) >= r:
                    continue
                dis_from_start = dist(i, j, from_x, from_y)
                if dis_from_start < sqar_r:
                    ratio = (sqar_r - dis_from_start) / (sqar_r - dis_from_start + sqar_dis_trans + epsilon)
                    ratio **= 2

                    trans_x = i - ratio * (to_x - from_x)
                    trans_y = j - ratio * (to_y - from_y)

                    self.im_bgr[i, j] = bilinear_ins(src_img, trans_x, trans_y)

    def face_thining(self, degree):
        #pdb.set_trace()
        #degree *= 1.5
        """
        width, height, _ = self.im_bgr.shape
        for ldmk in self.landmark:
            y, x = ldmk
            for i in range(-2, 3):
                for j in range(-2, 3):
                    if i + x >= 0 and i + x < width and j + y >= 0 and j + y <= height:
                        self.im_bgr[i+x, j+y, :] = np.array([0, 0, 0], dtype=np.uint8)
        """
        
        if degree <= 0.01:
            return

        left_ldmk1 = self.landmark[3]
        left_ldmk2 = self.landmark[5]

        right_ldmk1 = self.landmark[13]
        right_ldmk2 = self.landmark[15]

        to_point = self.landmark[30]

        trans_r_left = math.sqrt(dist(left_ldmk1[0], left_ldmk1[1], left_ldmk2[0], left_ldmk2[1]))
        trans_r_right = math.sqrt(dist(right_ldmk1[0], right_ldmk1[1], right_ldmk2[0], right_ldmk2[1]))

        self.local_transform(left_ldmk1[0], left_ldmk1[1], to_point[0], to_point[1], trans_r_left * degree)
        self.local_transform(right_ldmk1[0], right_ldmk1[1], to_point[0], to_point[1], trans_r_right * degree)


class Makeup():
    def __init__(self, raw_img, predictor_path="./data/shape_predictor_68_face_landmarks.dat"):
        self.predictor_path = predictor_path
        self.raw_img = copy.deepcopy(raw_img)

        assert raw_img.shape[2] == 1 or raw_img.shape[2] == 3
        
        #dlib detector and predictor
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(self.predictor_path)

        img_bgr = self.raw_img
        img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

        rect = self.detector(img_bgr, 1)

        assert len(rect) == 1
        rect = rect[0]

        self.raw_landmarks = np.array([[p.x, p.y] for p in self.predictor(img_bgr, rect).parts()])
        

    def get_faces(self, im_bgr, landmark):
        im_hsv = cv2.cvtColor(im_bgr, cv2.COLOR_BGR2HSV)
        
        temp_bgr = copy.deepcopy(im_bgr)
        temp_hsv = copy.deepcopy(im_hsv)

        return Face(im_bgr,im_hsv,temp_bgr,temp_hsv,landmark,0)

    def beautify(self, smooth_val=0.75, lip_brighten_val=0.3, whiten_val=0.2, sharpen_val=0.35, thin_val = 0.8):
        assert smooth_val >= 0 and smooth_val <= 1
        assert lip_brighten_val >= 0 and lip_brighten_val <= 1
        assert whiten_val >= 0 and whiten_val <= 1
        assert sharpen_val >= 0 and sharpen_val <= 1
        assert thin_val >= 0 and thin_val <= 1

        self.img = copy.deepcopy(self.raw_img)
        self.landmarks = copy.deepcopy(self.raw_landmarks)

        self.face_obj = self.get_faces(self.img, self.landmarks)

        self.face_obj.whitening(whiten_val)
        self.face_obj.smooth(smooth_val)

        smooth_val *= 4 / 5
        whiten_val *= 4 / 5
        sharpen_val *= 4 / 5

        self.face_obj.organs['forehead'].whitening(whiten_val)
        self.face_obj.organs['forehead'].smooth(smooth_val)
        self.face_obj.organs['mouth'].brightening(lip_brighten_val)
        self.face_obj.organs['mouth'].smooth(smooth_val)
        self.face_obj.organs['mouth'].whitening(whiten_val)
        self.face_obj.organs['left eye'].whitening(whiten_val)
        self.face_obj.organs['right eye'].whitening(whiten_val)
        self.face_obj.organs['left eye'].sharpen(sharpen_val)
        self.face_obj.organs['right eye'].sharpen(sharpen_val)
        self.face_obj.organs['left eye'].smooth(smooth_val)
        self.face_obj.organs['right eye'].smooth(smooth_val)
        self.face_obj.organs['left brow'].whitening(whiten_val)
        self.face_obj.organs['right brow'].whitening(whiten_val)
        self.face_obj.organs['left brow'].sharpen(sharpen_val)
        self.face_obj.organs['right brow'].sharpen(sharpen_val)
        self.face_obj.organs['nose'].whitening(whiten_val)
        self.face_obj.organs['nose'].smooth(smooth_val)
        self.face_obj.organs['nose'].sharpen(sharpen_val)
        
        self.face_obj.sharpen(sharpen_val * 5 / 4)
        
        #pdb.set_trace()
        #dbg_tmp_img = self.img
        self.face_obj.face_thining(thin_val)

        return self.face_obj.im_bgr



if __name__=='__main__':
    parser = argparse.ArgumentParser(description="face beautification")
    parser.add_argument("--path", type=str, required=True)
    args = parser.parse_args()
    path = args.path

    raw_img = cv2.imread(path)
    makeup_obj = Makeup(raw_img)

    # testing
    original = copy.deepcopy(raw_img)
    results = makeup_obj.beautify()

    compa_img = np.concatenate((original, results), axis=1)
    
    cv2.imwrite("results.jpg", compa_img)
