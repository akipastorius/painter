from numpy import (array, argmin, arange, ceil, sqrt, empty, arctan2, ones, cos, sin,
                   maximum, clip, pi, uint8, min as npmin, round as npround, ravel, reshape)
from numpy.random import choice, randint, shuffle
import cv2
from sklearn.cluster import KMeans
import tqdm


class Painter:
    def __init__(self,
                 input_path: str,
                 output_path: str='data/output/canvas.png'):
        self.input_path = input_path
        self.output_path = output_path
        self.img = cv2.imread(input_path)
        if self.img is None:
            raise ValueError('file {} not found'.format(self.input_path))
        self.img_original = self.img
        self.canvas = None
        height, width, channels = self.img.shape
        self.heigth = height
        self.width = width
        self.channels = channels
    
    def paint(self, stroke_scale: int=2, enhance_colors=True):
        # use color enhancing
        if enhance_colors:
            self.enhance_colors(cliplimit=1.5)
            self.warming_filter(1.2, 1.2)
            palette = self.create_palette(40, 0.01)
            self.color_convert(palette)
    
        # create brush strokes (stroke points, size and directions=gradients) for the image
        strokes, stroke_scale = self.create_strokes(scale=stroke_scale)
        gradientx, gradienty = self.gradients()
        # execute painting
        self.do_paint(strokes, stroke_scale, gradientx, gradienty)
    
    def do_paint(self, strokes, stroke_scale, gradientx, gradienty):
        x = arange(self.width)
        y = array(arange(self.height)).reshape((self.height, 1))
        canvas = ones((self.height, self.width, 3)) * 255//2
        n = strokes.shape[0]
        # loop all brush strokes
        for i in tqdm.tqdm(range(n)):
            h = strokes[i]
            y0 = int(h[0])
            x0 = int(h[1])
            
            # get brush stroke direction and size based on gradent
            direction = arctan2(gradienty[y0, x0], gradientx[y0, x0])
            magnitude = sqrt(gradientx[y0, x0]**2 + gradienty[y0, x0]**2)
            angle = direction + pi/2
            
            # get brush stroke length
            length = maximum(npround(stroke_scale + sqrt(magnitude) * 0.5), stroke_scale * 1.5)    
            # get pixels for brush stroke
            idx = self.ellipse(x, y, length, stroke_scale, x0, y0, angle)
            flat_idx = ravel(idx)
            # get original color
            color = self.img[y0, x0]
            for j in range(3):
                flat = ravel(canvas[:,:,j])
                flat[flat_idx] = color[j]
                canvas[:,:,j] = reshape(flat, (self.height, self.width))
    
        canvas[canvas == -1] = 255
        self.canvas = canvas
    
    def create_strokes(self, scale=None):
        if scale is None:
            scale = int(max(ceil(min(self.height, self.width) / 250), 2))
            
        r = max(scale//2,1)
        strokes = empty(((int(ceil(self.height/scale))*int(ceil(self.width/scale))), 2))
        cntr = -1
        for i in range(0, self.height, scale):
            for j in range(0, self.width, scale):
               cntr += 1
               y = randint(-r, r) + i
               x = randint(-r, r) + j
               strokes[cntr, :] = (y % self.height, x % self.width)
    
        shuffle(strokes)
        return strokes, scale
    
    def gradients(self, blur: bool=True):
        # grayscale and blur
        gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        if blur:
            gray = cv2.medianBlur(gray, 11)
            
        # gradients for strokes
        gradientx = cv2.Sobel(gray, cv2.CV_32F, dx=1, dy=0)
        gradienty = cv2.Sobel(gray, cv2.CV_32F, dx=0, dy=1)
        
        return gradientx, gradienty
    
    def scale(self, percent):
        new_height = int(self.heigth * percent / 100)
        new_width = int(self.width * percent / 100)
        dim = (new_width, new_height)
        img = cv2.resize(self.img, dsize=dim)
        self.img = img
        self.height = new_height
        self.width = new_width
    
    def enhance_colors(self, cliplimit=2.5):
        lab = cv2.cvtColor(self.img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
    
        # CLAHE TO l-channel
        clahe = cv2.createCLAHE(clipLimit=cliplimit, tileGridSize=(8,8))
        cl = clahe.apply(l)
    
        limg = cv2.merge((cl, a, b))
        self.img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
        
    def warming_filter(self, sat, val):
        h, s, v= cv2.split(cv2.cvtColor(self.img, cv2.COLOR_BGR2HSV))
        s = clip(s * sat, a_min=0, a_max=255).astype(uint8)
        v = clip(v * val, a_min=0, a_max=255).astype(uint8)
        self.img = cv2.cvtColor(cv2.merge([h, s, v]), cv2.COLOR_HSV2BGR)
    
    def color_convert(self, palette):
        for i in range(0, self.height):
            for j in range(0, self.width):
                k = argmin(((self.img[i, j, :] - palette)**2).sum(axis=1))
                self.img[i, j, :] = palette[k]
    
    def create_palette(self, k, frac):
        flat = self.img.reshape(self.height * self.width, 3)
        n = int(flat.shape[0] * frac)
        smpl = flat[choice(flat.shape[0], n)]
        clstr = KMeans(n_clusters=k, n_init=1)
        clstr.fit(smpl)
        palette = clstr.cluster_centers_.astype(int)
        return palette
    
    def ellipse(self, x, y, a, b, x0, y0, angle):
        return ((x-x0)*cos(angle) + (y-y0)*sin(angle))**2 / a**2 + ((x-x0)*sin(angle) - (y-y0)*cos(angle))**2 / b**2 <= 1

    def save_canvas(self):
        cv2.imwrite(self.output_path, self.canvas)
