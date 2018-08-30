

import tensorflow as tf
import time
from time import time as timer
import subprocess as sp
import numpy as np
import json
import os
import sys
import cv2
import warnings
from os.path import basename
import os.path
import argparse, random, socket, sys
from time import localtime, strftime
import pandas as pd


from darkflow.net.ops import op_create, identity
from darkflow.net.ops import HEADER, LINE
from darkflow.net.framework import create_framework
from darkflow.dark.darknet import Darknet
from darkflow.cython_utils.cy_yolo2_findboxes import box_constructor
from darkflow.utils.process import cfg_yielder
from darkflow.dark.darkop import create_darkop





class argHandler(dict):
    
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
    
    
class TFNet(object):

    _TRAINER = dict({
        'rmsprop': tf.train.RMSPropOptimizer,
        'adadelta': tf.train.AdadeltaOptimizer,
        'adagrad': tf.train.AdagradOptimizer,
        'adagradDA': tf.train.AdagradDAOptimizer,
        'momentum': tf.train.MomentumOptimizer,
        'adam': tf.train.AdamOptimizer,
        'ftrl': tf.train.FtrlOptimizer,
        'sgd': tf.train.GradientDescentOptimizer
    })

   
    def __init__(self, FLAGS, darknet = None):
        print('>>__init__')
        self.ntrain = 0
        self.male=0
        self.female=0
        self.FLAGS=FLAGS
        self.frame_count=0
        self.record = np.array([['0000000000000000000'],[0],[0]])
        
        if isinstance(FLAGS, dict):
            newFLAGS = argHandler()
            newFLAGS.update(FLAGS)
            FLAGS = newFLAGS



        self.FLAGS = FLAGS
        if self.FLAGS.pbLoad and self.FLAGS.metaLoad:
            self.say('\nLoading from .pb and .meta')
            self.graph = tf.Graph()
            device_name = FLAGS.gpuName                 if FLAGS.gpu > 0.0 else None
            with tf.device(device_name):
                with self.graph.as_default() as g:
                    self.build_from_pb()
            return


        # init socket
        if self.FLAGS.socket == 'yes':
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.sock.connect((FLAGS.hostname, FLAGS.port))
            print('Client socket name is {}'.format(self.sock.getsockname()))  
            
            text='Client connected'
            print('Socket connected')
            data = text.encode('ascii')
            self.sock.send(data)       
            
        
        
        if darknet is None: 
            print('>>darknet = Darknet(FLAGS)')
            darknet = Darknet(FLAGS)

            self.ntrain = len(darknet.layers)

        self.darknet = darknet
        args = [darknet.meta, FLAGS]



        self.num_layer = len(darknet.layers)
        print('>>layers = ',self.num_layer)
        self.framework = create_framework(*args)
        
        # meta dict   
        #self.meta={'net': {'type': '[net]', 'batch': 1, 'subdivisions': 1, 'width': 608, 'height': 608, 'channels': 3, 'momentum': 0.9, 'decay': 0.0005, 'angle': 0, 'saturation': 1.5, 'exposure': 1.5, 'hue': 0.1, 'learning_rate': 0.001, 'burn_in': 1000, 'max_batches': 500200, 'policy': 'steps', 'steps': '400000,450000', 'scales': '.1,.1'}, 'type': '[region]', 'anchors': [0.57273, 0.677385, 1.87446, 2.06253, 3.33843, 5.47434, 7.88282, 3.52778, 9.77052, 9.16828], 'bias_match': 1, 'classes': 80, 'coords': 4, 'num': 5, 'softmax': 1, 'jitter': 0.3, 'rescore': 1, 'object_scale': 5, 'noobject_scale': 1, 'class_scale': 1, 'coord_scale': 1, 'absolute': 1, 'thresh': 0.3, 'random': 1, 'model': 'cfg/yolo.cfg', 'inp_size': [608, 608, 3], 'out_size': [19, 19, 425], 'name': 'yolo', 'labels': ['person', 'bicycle', 'car', 'motorbike', 'aeroplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'sofa', 'pottedplant', 'bed', 'diningtable', 'toilet', 'tvmonitor', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'], 'colors': [(254.0, 254.0, 254), (248.92, 228.6, 127), (243.84, 203.20000000000002, 0), (238.76, 177.79999999999998, -127), (233.68, 152.4, -254), (228.6, 127.0, 254), (223.52, 101.60000000000001, 127), (218.44, 76.20000000000002, 0), (213.35999999999999, 50.79999999999999, -127), (208.28000000000003, 25.399999999999995, -254), (203.20000000000002, 0.0, 254), (198.12, -25.400000000000023, 127), (193.04, -50.79999999999999, 0), (187.96, -76.20000000000002, -127), (182.88, -101.59999999999998, -254), (177.79999999999998, -127.0, 254), (172.71999999999997, -152.40000000000003, 127), (167.64, -177.79999999999998, 0), (162.56, -203.20000000000002, -127), (157.48, -228.59999999999997, -254), (152.4, -254.0, 254), (147.32000000000002, -279.40000000000003, 127), (142.24, -304.80000000000007, 0), (137.16, -330.19999999999993, -127), (132.08, -355.59999999999997, -254), (127.0, 254.0, 254), (121.92, 228.6, 127), (116.83999999999999, 203.20000000000002, 0), (111.75999999999999, 177.79999999999998, -127), (106.68, 152.4, -254), (101.60000000000001, 127.0, 254), (96.52, 101.60000000000001, 127), (91.44, 76.20000000000002, 0), (86.35999999999999, 50.79999999999999, -127), (81.27999999999999, 25.399999999999995, -254), (76.20000000000002, 0.0, 254), (71.12, -25.400000000000023, 127), (66.04, -50.79999999999999, 0), (60.96, -76.20000000000002, -127), (55.879999999999995, -101.59999999999998, -254), (50.79999999999999, -127.0, 254), (45.72000000000001, -152.40000000000003, 127), (40.64000000000001, -177.79999999999998, 0), (35.56, -203.20000000000002, -127), (30.48, -228.59999999999997, -254), (25.399999999999995, -254.0, 254), (20.31999999999999, -279.40000000000003, 127), (15.240000000000013, -304.80000000000007, 0), (10.160000000000009, -330.19999999999993, -127), (5.0800000000000045, -355.59999999999997, -254), (0.0, 254.0, 254), (-5.0800000000000045, 228.6, 127), (-10.160000000000009, 203.20000000000002, 0), (-15.240000000000013, 177.79999999999998, -127), (-20.320000000000018, 152.4, -254), (-25.400000000000023, 127.0, 254), (-30.480000000000025, 101.60000000000001, 127), (-35.559999999999974, 76.20000000000002, 0), (-40.63999999999998, 50.79999999999999, -127), (-45.719999999999985, 25.399999999999995, -254), (-50.79999999999999, 0.0, 254), (-55.879999999999995, -25.400000000000023, 127), (-60.96, -50.79999999999999, 0), (-66.04, -76.20000000000002, -127), (-71.12, -101.59999999999998, -254), (-76.20000000000002, -127.0, 254), (-81.28000000000002, -152.40000000000003, 127), (-86.36000000000001, -177.79999999999998, 0), (-91.44000000000003, -203.20000000000002, -127), (-96.51999999999997, -228.59999999999997, -254), (-101.59999999999998, -254.0, 254), (-106.67999999999998, -279.40000000000003, 127), (-111.75999999999999, -304.80000000000007, 0), (-116.83999999999999, -330.19999999999993, -127), (-121.92, -355.59999999999997, -254), (-127.0, 254.0, 254), (-132.08, 228.6, 127), (-137.16, 203.20000000000002, 0), (-142.24, 177.79999999999998, -127), (-147.32000000000002, 152.4, -254)]}
        self.meta=darknet.meta #auto define
        print(self.meta)
        
        print('>>meta class = ',len(self.meta['labels']))

        self.say('\nBuilding net ...')
        start = time.time()
        self.graph = tf.Graph()
        device_name = FLAGS.gpuName             if FLAGS.gpu > 0.0 else None
        with tf.device(device_name):
            with self.graph.as_default() as g:
                self.build_forward()
                self.setup_meta_ops()
        self.say('Finished in {}s\n'.format(
            time.time() - start))
    



    def build_from_pb(self):
        with tf.gfile.FastGFile(self.FLAGS.pbLoad, "rb") as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
        
        tf.import_graph_def(
            graph_def,
            name=""
        )
        with open(self.FLAGS.metaLoad, 'r') as fp:
            self.meta = json.load(fp)
        self.framework = create_framework(self.meta, self.FLAGS)

        # Placeholders
        self.inp = tf.get_default_graph().get_tensor_by_name('input:0')
        self.feed = dict() # other placeholders
        self.out = tf.get_default_graph().get_tensor_by_name('output:0')
        
        self.setup_meta_ops()
    
    def build_forward(self):
        verbalise = self.FLAGS.verbalise

        # Placeholders
        inp_size = [None] + self.meta['inp_size']
        self.inp = tf.placeholder(tf.float32, inp_size, 'input')
        self.feed = dict() # other placeholders

        # Build the forward pass
        state = identity(self.inp)
        roof = self.num_layer - self.ntrain
        self.say(HEADER, LINE)
        for i, layer in enumerate(self.darknet.layers):
            scope = '{}-{}'.format(str(i),layer.type)
            args = [layer, state, i, roof, self.feed]
            state = op_create(*args)   ####################################
            mess = state.verbalise()
            self.say(mess)
        self.say(LINE)

        self.top = state
        self.out = tf.identity(state.out, name='output')

    def setup_meta_ops(self):
        cfg = dict({
            'allow_soft_placement': False,
            'log_device_placement': False
        })

        utility = min(self.FLAGS.gpu, 1.)
        if utility > 0.0:
            self.say('GPU mode with {} usage'.format(utility))
            cfg['gpu_options'] = tf.GPUOptions(
                per_process_gpu_memory_fraction = utility)
            cfg['allow_soft_placement'] = True
        else: 
            self.say('Running entirely on CPU')
            cfg['device_count'] = {'GPU': 0}

        if self.FLAGS.train: self.build_train_op()
        
        if self.FLAGS.summary is not None:
            self.summary_op = tf.summary.merge_all()
            self.writer = tf.summary.FileWriter(self.FLAGS.summary + 'train')
        
        self.sess = tf.Session(config = tf.ConfigProto(**cfg))
        self.sess.run(tf.global_variables_initializer())

        if not self.ntrain: return
        self.saver = tf.train.Saver(tf.global_variables(), 
            max_to_keep = self.FLAGS.keep)
        if self.FLAGS.load != 0: self.load_from_ckpt()
        
        if self.FLAGS.summary is not None:
            self.writer.add_graph(self.sess.graph)



    def feed_once(self, im):
        
        assert isinstance(im, np.ndarray), \
                    'Image is not a np.ndarray'
        h, w, _ = im.shape
        im = self.framework.resize_input(im)

        this_inp = np.expand_dims(im, 0)
        feed_dict = {self.inp : this_inp}

        net_out = self.sess.run(self.out, feed_dict)[0]
        return net_out


    def savepb(self):
        """
        Create a standalone const graph def that 
        C++ can load and run.
        """
        darknet_pb = self.to_darknet()
        flags_pb = self.FLAGS
        flags_pb.verbalise = False
        
        flags_pb.train = False
        # rebuild another tfnet. all const.
        tfnet_pb = TFNet(flags_pb, darknet_pb)      
        tfnet_pb.sess = tf.Session(graph = tfnet_pb.graph)
        # tfnet_pb.predict() # uncomment for unit testing
        name = 'built_graph/{}.pb'.format(self.meta['name'])
        os.makedirs(os.path.dirname(name), exist_ok=True)
        #Save dump of everything in meta
        with open('built_graph/{}.meta'.format(self.meta['name']), 'w') as fp:
            json.dump(self.meta, fp)
        self.say('Saving const graph def to {}'.format(name))
        graph_def = tfnet_pb.sess.graph_def
        tf.train.write_graph(graph_def,'./', name, False)

    def load_from_ckpt(self):
        if self.FLAGS.load < 0: # load lastest ckpt
            with open(self.FLAGS.backup + 'checkpoint', 'r') as f:
                last = f.readlines()[-1].strip()
                load_point = last.split(' ')[1]
                load_point = load_point.split('"')[1]
                load_point = load_point.split('-')[-1]
                self.FLAGS.load = int(load_point)
        
        load_point = os.path.join(self.FLAGS.backup, self.meta['name'])
        load_point = '{}-{}'.format(load_point, self.FLAGS.load)
        self.say('Loading from {}'.format(load_point))
        try: self.saver.restore(self.sess, load_point)
        except: load_old_graph(self, load_point)

    def say(self, *msgs):
        if not self.FLAGS.verbalise:
            return
        msgs = list(msgs)
        for msg in msgs:
            if msg is None: continue
            print(msg)

    def _get_fps(self, frame): #get fps when detect on video
        elapsed = int()
        start = timer()
        preprocessed = self.preprocess(frame)
        feed_dict = {self.inp: [preprocessed]}
        net_out = self.sess.run(self.out, feed_dict)[0]
        processed = self.postprocess(net_out, frame, False)
        return timer() - start

    def camera(self):
        file = self.FLAGS.demo
        URL=self.FLAGS.URL
        SaveVideo = self.FLAGS.saveVideo

        if file == 'camera':
            file = 0
            camera = cv2.VideoCapture(file)
        elif file == 'stream':
            
            
            width=640
            height=480
            sp.call(["youtube-dl","--list-format",URL])
            run=sp.Popen(["youtube-dl","-f","94","-g", URL],stdout = sp.PIPE)
            VIDM3U8, _=run.communicate()


            VIDM3U8=str(VIDM3U8,'utf-8')
            VIDM3U8="".join(("hls://",str(VIDM3U8)))

            p1 = sp.Popen(['streamlink','--hls-segment-threads','10', VIDM3U8,'best','-o','-'],stdout = sp.PIPE)
            p2 = sp.Popen(['ffmpeg','-i', '-','-f', 'image2pipe',"-loglevel","quiet","-pix_fmt", "bgr24","-vcodec", "rawvideo",'-r','10', "-"],stdin=p1.stdout,stdout=sp.PIPE)


        else:
            assert os.path.isfile(file),             'file {} does not exist'.format(file)
            camera = cv2.VideoCapture(file) ## IMPORTANT HERE

        if file == 0:
            self.say('Press [ESC] to quit demo')

        #assert camera.isOpened(), \
        #'Cannot capture source' #trigger an error if the condition is false.

        if file == 0:#window setting for camera
            cv2.namedWindow('', 0)
            _, frame = camera.read()
            height, width, _ = frame.shape        
            cv2.resizeWindow('', width, height)
        elif file =='stream': #window setting for stream
            cv2.namedWindow('',0)
            height = 480
            width = 640
            cv2.resizeWindow('', width, height)
            
        else: #window setting for file
            _, frame = camera.read()
            height, width, _ = frame.shape

        if SaveVideo:
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            if file == 0:#camera window
              fps = 1 / self._get_fps(frame)
              if fps < 1:
                fps = 1
            else:
                fps = round(camera.get(cv2.CAP_PROP_FPS))
            videoWriter = cv2.VideoWriter(
                'output.avi', fourcc, fps, (width, height))


        # buffers for demo in batch
        buffer_inp = list()
        buffer_pre = list()
        
        elapsed = int()
        start = timer()
        self.say('Press [ESC] to quit demo')
        # Loop through frames
                
        while True:
            elapsed += 1
            if file == 0:
                _, frame = camera.read()  # if use camera ana
                #print(frame.shape)
            elif file =='stream':


                    raw_frame = p2.stdout.read(640*480*3) 
                    frame =  np.fromstring(raw_frame, dtype='uint8').reshape((480,640,3))


            if frame is None:
                print ('\nEnd of Video')
                break  

            preprocessed = self.preprocess(frame)
            #print(preprocessed.shape)
            buffer_inp.append(frame)
            buffer_pre.append(preprocessed)
            # Only process and imshow when queue is full
            


            if elapsed % self.FLAGS.queue == 0:
                feed_dict = {self.inp: buffer_pre}

                #####____FEDD_INPUT____#####
                net_out = self.sess.run(self.out, feed_dict)


                for img, single_out in zip(buffer_inp, net_out):
                    postprocessed = self.postprocess(
                        single_out, img, False)
                    if SaveVideo:
                        videoWriter.write(postprocessed)

 ################################################################____CAMERA____#################################################################################                       
                    if file == 0: #camera window                                                
                        cv2.imshow('', postprocessed)
                        print('male : {} female: {}'.format(self.male,self.female))
                        
                        

                        if self.FLAGS.socket == 'yes':
                            data=(self.male),(self.female)
                            data=str(data)
                            self.sock.send(data.encode('ascii'))
                        

                        if self.FLAGS.csv == 'yes':
                            # add number to numpy array
                            self.record=np.insert(self.record,len(self.record[1,:]),[strftime("%Y-%m-%d_%H:%M:%S", localtime()), self.male,self.female],axis=1)
                        
 
 #####################################################################____STREAM____##########################################################################################  
                    if file == 'stream':                                      
                        cv2.imshow('',postprocessed)
                        
                        print('male : {} female: {}'.format(self.male,self.female))
                        

                        if self.FLAGS.socket == 'yes':
                            data=(self.male),(self.female)
                            data=str(data)
                            self.sock.send(data.encode('ascii'))
                        
                        if self.FLAGS.csv == 'yes':
                            # add number to numpy array
                            self.record=np.insert(self.record,len(self.record[1,:]),[strftime("%Y-%m-%d_%H:%M:%S", localtime()), self.male,self.female],axis=1)
                    
                    if self.FLAGS.csv == 'yes':
                        self.frame_count+=1
                        #print('frame_count {}'.format(self.frame_count))                 

 #####################################################################################################################################################################                      
                    self.male=0   
                    self.female=0  

                    
                    self.save_csv() if (self.FLAGS.csv == 'yes' and self.frame_count == 100) else None


                             

                # Clear Buffers
                buffer_inp = list()
                buffer_pre = list()

            

            if elapsed % 5 == 0:
                sys.stdout.write('\r')
                sys.stdout.write('{0:3.3f} FPS'.format(
                    elapsed / (timer() - start)))
                sys.stdout.flush()

            if file == 0: #camera window
                choice = cv2.waitKey(1)
                if choice == 27: break
            if file == 'stream': #stream window
                choice = cv2.waitKey(1)
                if choice == 27: break

        sys.stdout.write('\n')
        self.save_csv() if (self.FLAGS.csv == 'yes') else None # save csv when hit Esc
        if SaveVideo:
            videoWriter.release()
        
        if file == 0: #camera window
            camera.release()
            cv2.destroyAllWindows()


        if file == 'stream': #stream window
            cv2.destroyAllWindows()  
        if self.FLAGS.socket == 'yes':
            text ='Client disconnected'
            data = text.encode('ascii')
            self.sock.send(data)   
            self.sock.close()
            
            
    def save_csv(self):
        


        df = pd.DataFrame(self.record[1:,1:].T, index=self.record[0,1:].T,columns=['male','female'])
        df.index.name = 'time'

        self.record = np.array([['0000000000000000000'],[0],[0]])




        if os.path.isfile('record.csv') and self.FLAGS.csv == 'yes':  ## if file already there
            with open('record.csv', 'a') as f:
                                df.to_csv(f, header=False)
            print('################################ record.csv appended ')

        else : # file doesn't exist

            df.to_csv('record.csv', sep=',')
            print('################################ record.csv created ')
        df=None   
        self.frame_count=0


                             
        
    def preprocess(self, im, allobj = None):
        if type(im) is not np.ndarray:
            im = cv2.imread(im)

        if allobj is not None: # in training mode
            result = imcv2_affine_trans(im)
            im, dims, trans_param = result
            scale, offs, flip = trans_param
            for obj in allobj:
                _fix(obj, dims, scale, offs)
                if not flip: continue
                obj_1_ =  obj[1]
                obj[1] = dims[0] - obj[3]
                obj[3] = dims[0] - obj_1_
            im = imcv2_recolor(im)

        im = self.resize_input(im)
        if allobj is None: return im
        return im#, np.array(im) # for unit testing
    def resize_input(self, im):
        
        h, w, c = self.meta['inp_size']
        imsz = cv2.resize(im, (w, h)) #resize with opencv to 418*418*3
        imsz = imsz / 255. #normalize
        imsz = imsz[:,:,::-1] #BGR to RGB
        return imsz

    def findboxes(self, net_out):
    # meta
        meta = self.meta
        boxes = list()
        boxes=box_constructor(meta,net_out)
        return boxes

    def process_box(self, b, h, w, threshold):
        max_indx = np.argmax(b.probs)
        max_prob = b.probs[max_indx]
        label = self.meta['labels'][max_indx]
        if max_prob > threshold:
            left  = int ((b.x - b.w/2.) * w)
            right = int ((b.x + b.w/2.) * w)
            top   = int ((b.y - b.h/2.) * h)
            bot   = int ((b.y + b.h/2.) * h)
            if left  < 0    :  left = 0
            if right > w - 1: right = w - 1
            if top   < 0    :   top = 0
            if bot   > h - 1:   bot = h - 1
            mess = '{}'.format(label)
            return (left, right, top, bot, mess, max_indx, max_prob)
        return None

    def postprocess(self, net_out, im, save = True):
        """
        Takes net output, draw net_out, save to disk
        """
        boxes = self.findboxes(net_out)

        # meta
        meta = self.meta
        threshold = meta['thresh']
        colors = [(249,68,61),(57,31,249)]
        labels = meta['labels']
        if type(im) is not np.ndarray:
            imgcv = cv2.imread(im)
        else: imgcv = im
        h, w, _ = imgcv.shape
      

        resultsForJSON = []
        for b in boxes:
            boxResults = self.process_box(b, h, w, threshold)
            if boxResults is None:
                continue
            left, right, top, bot, mess, max_indx, confidence = boxResults
            if mess == 'male':
                self.male+=1 
            if mess == 'female':
                self.female+=1 

            

            
                
            thick = int((h + w) // 300)
            cv2.rectangle(imgcv,
                (left, top), (right, bot),
                colors[max_indx], thick)
            cv2.putText(imgcv, mess+' '+str('%.2f' % confidence), (left, top - 12),
                0, 1e-3 * h, colors[max_indx],thick//3)
           

        if not save: return imgcv
           
  

class loader(object):
    """
    interface to work with both .weights and .ckpt files
    in loading / recollecting / resolving mode
    """
    VAR_LAYER = ['convolutional', 'connected', 'local', 
                 'select', 'conv-select',
                 'extract', 'conv-extract']

    def __init__(self, *args):
        self.src_key = list()
        self.vals = list()
        self.load(*args)

    def __call__(self, key):
        for idx in range(len(key)):
            val = self.find(key, idx)
            if val is not None: return val
        return None
    
    def find(self, key, idx):
        up_to = min(len(self.src_key), 4)
        for i in range(up_to):
            key_b = self.src_key[i]
            if key_b[idx:] == key[idx:]:
                return self.yields(i)
        return None

    def yields(self, idx):
        del self.src_key[idx]
        temp = self.vals[idx]
        del self.vals[idx]
        return temp

class weights_loader(loader):
    """one who understands .weights files"""
    
    _W_ORDER = dict({ # order of param flattened into .weights file
        'convolutional': [
            'biases','gamma','moving_mean','moving_variance','kernel'
        ],
        'connected': ['biases', 'weights'],
        'local': ['biases', 'kernels']
    })

    def load(self, path, src_layers):
        self.src_layers = src_layers
        walker = weights_walker(path)

        for i, layer in enumerate(src_layers):
            if layer.type not in self.VAR_LAYER: continue
            self.src_key.append([layer])
            
            if walker.eof: new = None
            else: 
                args = layer.signature
                new = create_darkop(*args)
            self.vals.append(new)

            if new is None: continue
            order = self._W_ORDER[new.type]
            for par in order:
                if par not in new.wshape: continue
                val = walker.walk(new.wsize[par])
                new.w[par] = val
            new.finalize(walker.transpose)

        if walker.path is not None:
            assert walker.offset == walker.size, \
            'expect {} bytes, found {}'.format(
                walker.offset, walker.size)
            print('Successfully identified {} bytes'.format(
                walker.offset))

class checkpoint_loader(loader):
    """
    one who understands .ckpt files, very much
    """
    def load(self, ckpt, ignore):
        meta = ckpt + '.meta'
        with tf.Graph().as_default() as graph:
            with tf.Session().as_default() as sess:
                saver = tf.train.import_meta_graph(meta)
                saver.restore(sess, ckpt)
                for var in tf.global_variables():
                    name = var.name.split(':')[0]
                    packet = [name, var.get_shape().as_list()]
                    self.src_key += [packet]
                    self.vals += [var.eval(sess)]


class weights_walker(object):
    """incremental reader of float32 binary files"""
    def __init__(self, path):
        self.eof = False # end of file
        self.path = path  # current pos
        if path is None: 
            self.eof = True
            return
        else: 
            self.size = os.path.getsize(path)# save the path
            major, minor, revision, seen = np.memmap(path,
                shape = (), mode = 'r', offset = 0,
                dtype = '({})i4,'.format(4))
            self.transpose = major > 1000 or minor > 1000
            self.offset = 16

    def walk(self, size):
        if self.eof: return None
        end_point = self.offset + 4 * size
        assert end_point <= self.size, \
        'Over-read {}'.format(self.path)

        float32_1D_array = np.memmap(
            self.path, shape = (), mode = 'r', 
            offset = self.offset,
            dtype='({})float32,'.format(size)
        )

        self.offset = end_point
        if end_point == self.size: 
            self.eof = True
        return float32_1D_array

def model_name(file_path):
    file_name = basename(file_path)
    ext = str()
    if '.' in file_name: # exclude extension
        file_name = file_name.split('.')
        ext = file_name[-1]
        file_name = '.'.join(file_name[:-1])
    if ext == str() or ext == 'meta': # ckpt file
        file_name = file_name.split('-')
        num = int(file_name[-1])
        return '-'.join(file_name[:-1])
    if ext == 'weights':
        return file_name
def create_loader(path, cfg = None):
    if path is None:
        load_type = weights_loader
    elif '.weights' in path:
        load_type = weights_loader
    else: 
        load_type = checkpoint_loader
    
    return load_type(path, cfg)
        






if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Darkflow re-arrange version with stream video and socket')

    parser.add_argument('--demo',default='stream',help='stream or camera')
    parser.add_argument('--url',default='https://www.youtube.com/watch?v=an1_CXsBkKk',help='youtube url')

    parser.add_argument('--socket',default='no',help='yes or no')
    parser.add_argument('--hostname',default='127.0.0.1',help='socket server ip')
    parser.add_argument('--port',type=int, default=1060,help='server port')

    parser.add_argument('--gpu',type=float, default='0',help='gpu usage in percent')
    parser.add_argument('--csv',default='no',help='save record in csv file')
    
    
    args = parser.parse_args()
 


    FLAGS={'imgdir': './sample_img/', 'binary': './bin/', 'config': './cfg/', 'dataset': '../pascal/VOCdevkit/IMG/', \
    'labels': 'labels.txt', 'backup': './ckpt/', 'summary': './summary/', 'annotation': '../pascal/VOCdevkit/ANN/', \
    'threshold': 0.7, 'model': 'cfg/yolo-female.cfg', 'trainer': 'adam', 'momentum': 0.0, 'verbalise': True, \
    'train': False, 'load': 20400, 'savepb': False, 'gpu': args.gpu, 'gpuName': '/gpu:0', 'lr': 1e-05, \
    'keep': 20, 'batch': 16, 'epoch': 1000, 'save': 2000, 'demo': args.demo, 'URL': args.url,\
    'queue': 1, 'json': False, 'saveVideo': False,\
    'pbLoad': '', 'metaLoad': '', 'socket': args.socket,'hostname': args.hostname,'port': args.port,'csv': args.csv}


    

    tfnet = TFNet(FLAGS)

    tfnet.camera()




