import tensorflow as tf
from lucid.modelzoo.vision_base import Model
import warnings 
import lucid.optvis.render as render
warnings.filterwarnings('ignore')
graph_file='nasnet_mobile_graphdef.pb'
graph_def=tf.GraphDef()
with open(graph_file,'rb') as f:
    graph_def.ParseFromString(f.read())
for node in graph_def.node:
    print(node.name)
class NasNetMobile(Model):
    model_path='nasnet_mobile_graphdef_frozen.pb.modelzoo'
    image_shape=[224,224,3]
    image_value_range=(0,1)
    input_name='input'
if __name__=='__main__':
	nasnet=NasNetMobile()
	_=render.render_vis(nasnet,'cell_0/cell_output/concat:0')
	