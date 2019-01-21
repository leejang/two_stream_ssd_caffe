import caffe
import numpy as np
#import glog

class RobustL2L1LossLayer(caffe.Layer):

    @classmethod
    def parse_args(cls, argsStr):
      parser = argparse.ArgumentParser(description='Python L1 Loss Layer')
      parser.add_argument('--loss_weight', default=1.0, type=float)
      args   = parser.parse_args(argsStr.split())
      print('Using Config:')
      pprint.pprint(args)
      return args
		
    def setup(self, bottom, top):
      assert len(bottom) == 2, 'There should be two bottom blobs'
      self.scale = 2.0
      #self.c_val = 1.0*256*25*25
      #self.c2 = self.c_val ** 2
      predShape = bottom[0].data.shape
      gtShape   = bottom[1].data.shape
      for i in range(len(predShape)):
        assert predShape[i] == gtShape[i], 'Mismatch: %d, %d' % (predShape[i], gtShape[i])
      assert bottom[0].data.squeeze().ndim == bottom[1].data.squeeze().ndim, 'Shape Mismatch'
      assert len(top)==1, 'There should be only one output blob'
      top[0].reshape(1,1,1,1)
      
      #f=open('/data/chenyou/caffe_SSD/examples/robot/procedures/dpn/out.txt','w')
      #f.close()
    
    def forward(self, bottom, top):
      self.batchSz_ = bottom[0].data.shape[0]
    
      #c=self.c_val
      
      l1_val = np.abs(bottom[0].data[...] - bottom[1].data[...])
      #l1_val = np.clip(l1_val,0,self.scale)
      l2_val = l1_val ** 2
      
      top[0].data[...] = np.sum(l2_val) / float(2.0) / float(self.batchSz_)
      
      #print 'Loss is %f' % top[0].data[0]

    def backward(self, top, propagate_down, bottom):
  
      diff = (bottom[0].data[...] - bottom[1].data[...])
      diff = np.clip(diff, -self.scale, self.scale)
      
      bottom[0].diff[...] =  diff / float(self.batchSz_) 

  
    def reshape(self, bottom, top):
      top[0].reshape(1,1,1,1)
      pass