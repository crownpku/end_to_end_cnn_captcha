# pylint: disable=C0111,too-many-arguments,too-many-instance-attributes,too-many-locals,redefined-outer-name,fixme
# pylint: disable=superfluous-parens, no-member, invalid-name
import sys
sys.path.insert(0, "../../python")
import mxnet as mx
import numpy as np
import cv2, random

from io import BytesIO
from generate_captcha import *
#from captcha.image import ImageCaptcha

char_dic = {}


class OCRBatch(object):
    def __init__(self, data_names, data, label_names, label):
        self.data = data
        self.label = label
        self.data_names = data_names
        self.label_names = label_names

    @property
    def provide_data(self):
        return [(n, x.shape) for n, x in zip(self.data_names, self.data)]

    @property
    def provide_label(self):
        return [(n, x.shape) for n, x in zip(self.label_names, self.label)]

def gen_rand():
    num = random.randint(0, 9999)
    buf = str(num)
    while len(buf) < 4:
        buf = "0" + buf
    return buf

def get_label(buf):
    global char_dic
    char_num = len(char_dic)
    for x in buf:
	if x not in char_dic:
	    char_dic[x] = char_num
	    print str(char_num) + ':' + str(x) 
	    char_num = char_num + 1


    return np.array([int(char_dic[x]) for x in buf])



class OCRIter(mx.io.DataIter):
    def __init__(self, count, batch_size, num_label, height, width):
        super(OCRIter, self).__init__()
        #self.captcha = ImageCaptcha(fonts=['./data/OpenSans-Regular.ttf']) 
	self.batch_size = batch_size
        self.count = count
        self.height = height
        self.width = width
        self.provide_data = [('data', (batch_size, 3, height, width))]
        self.provide_label = [('softmax_label', (self.batch_size, num_label))]

    def __iter__(self):
        for k in range(self.count / self.batch_size):
            data = []
            label = []
            for i in range(self.batch_size):
                num = gen_rand()
                #img = self.captcha.generate(num)
                img, num = create_validate_code()
		#img = binarization(img)
		#pic_num = str(i%10)
		#print img
		#print num
		#img.save('test.png')
		#img = np.fromstring(img.getvalue(), dtype='uint8')
		#img.save("./org" + pic_num + ".png")
                img = np.asarray(img)#, dtype='uint8')
		
		#img = cv2.imdecode(img, cv2.IMREAD_COLOR)
                #img = img[:,:,::-1] 
		#Guan 20161108: Convert BGR to RGB
		img = cv2.resize(img, (self.width, self.height))

                
		#cv2.imwrite("./tmp" + pic_num + ".png", img)
                img = np.multiply(img, 1/255.0)
                img = img.transpose(2, 0, 1)
                data.append(img)
                label.append(get_label(num))

            data_all = [mx.nd.array(data)]
            label_all = [mx.nd.array(label)]
            data_names = ['data']
            label_names = ['softmax_label']

            data_batch = OCRBatch(data_names, data_all, label_names, label_all)
            yield data_batch

    def reset(self):
        pass

def get_ocrnet():
    data = mx.symbol.Variable('data')
    label = mx.symbol.Variable('softmax_label')
    conv1 = mx.symbol.Convolution(data=data, kernel=(5,5), num_filter=32)
    pool1 = mx.symbol.Pooling(data=conv1, pool_type="max", kernel=(2,2), stride=(1, 1))
    relu1 = mx.symbol.Activation(data=pool1, act_type="relu")

    conv2 = mx.symbol.Convolution(data=relu1, kernel=(5,5), num_filter=32)
    pool2 = mx.symbol.Pooling(data=conv2, pool_type="avg", kernel=(2,2), stride=(1, 1))
    relu2 = mx.symbol.Activation(data=pool2, act_type="relu")

    conv3 = mx.symbol.Convolution(data=relu2, kernel=(3,3), num_filter=32)
    pool3 = mx.symbol.Pooling(data=conv3, pool_type="avg", kernel=(2,2), stride=(1, 1))
    relu3 = mx.symbol.Activation(data=pool3, act_type="relu")

    conv4 = mx.symbol.Convolution(data=relu3, kernel=(3,3), num_filter=32)
    pool4 = mx.symbol.Pooling(data=conv4, pool_type="avg", kernel=(2,2), stride=(1, 1))
    relu4 = mx.symbol.Activation(data=pool4, act_type="relu")


    flatten = mx.symbol.Flatten(data = relu4)
    fc1 = mx.symbol.FullyConnected(data = flatten, num_hidden = 256)
    fc21 = mx.symbol.FullyConnected(data = fc1, num_hidden = 43)
    fc22 = mx.symbol.FullyConnected(data = fc1, num_hidden = 43)
    fc23 = mx.symbol.FullyConnected(data = fc1, num_hidden = 43)
    fc24 = mx.symbol.FullyConnected(data = fc1, num_hidden = 43)
    fc25 = mx.symbol.FullyConnected(data = fc1, num_hidden = 43)
    fc26 = mx.symbol.FullyConnected(data = fc1, num_hidden = 43)


    fc2 = mx.symbol.Concat(*[fc21, fc22, fc23, fc24, fc25, fc26], dim = 0)
    label = mx.symbol.transpose(data = label)
    label = mx.symbol.Reshape(data = label, target_shape = (0, ))
    return mx.symbol.SoftmaxOutput(data = fc2, label = label, name = "softmax")


def Accuracy(label, pred):
    label = label.T.reshape((-1, ))
    hit = 0
    total = 0
    for i in range(pred.shape[0] / 6):
        ok = True
        for j in range(6):
            k = i * 6 + j
            if np.argmax(pred[k]) != int(label[k]):
	        ok = False
                break
        if ok:
            hit += 1
        total += 1
    return 1.0 * hit / total


def train():
    network = get_ocrnet()
    devs = [mx.gpu(0)]
    model = mx.model.FeedForward(ctx = devs,
                                 symbol = network,
                                 num_epoch = 200,
                                 learning_rate = 0.001,
                                 wd = 0.00001,
                                 initializer = mx.init.Xavier(factor_type="in", magnitude=2.34),
                                 momentum = 0.9)
    batch_size = 200
    data_train = OCRIter(150000, batch_size, 6, 25, 96)
    data_test = OCRIter(10000, batch_size, 6, 25, 96)
    
    import logging
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(level=logging.DEBUG, format=head)
    
    model.fit(X = data_train, eval_data = data_test, eval_metric = Accuracy, batch_end_callback=mx.callback.Speedometer(batch_size, 50),)
    model.save("cnn-ocr-20161109")

if __name__ == '__main__':
    train()

