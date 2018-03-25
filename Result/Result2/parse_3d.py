import numpy as np
import cv2
import sklearn.model_selection as sk
from analysis import parse_protobufs
from ipdb import set_trace as debug
import os
import tensorflow as tf

class struct():
	pass
def parse_data(save=True):

	data_dict_x = {}#rgb classified by object
	data_dict_d = {}#rgb classified by object    
	data_dict_y = {}#rgb classified by object
	batch_dict = {}
	save_path_train = '/home/zhouzixuan/notebooks/proj_new/data/train/'
	save_path_test =  '/home/zhouzixuan/notebooks/proj_new/data/test/'
	# example data extraction of x value of object/item 0 in training example 0: data.states[0].items[0].x
	num_examples = 18000 # number or screenshots
	num_items = []  # number of items in each example
	labels = []
	labels_rgb = []
	X_rgb = np.empty([0,299,299,3])
    
	X_d = np.empty([0,299,299])
	batch_size = 32
	path = struct()
	path.data_name = '_SessionStateData.proto'
	path.data_folder = 'TeleOpVRSession_2018-02-05_15-44-11/'
	data0 = parse_protobufs(path)
	path.data_folder = 'TeleOpVRSession_2018-03-07_14-38-06_Camera1/'
	data1 = parse_protobufs(path)    
	path.data_folder = 'TeleOpVRSession_2018-03-07_14-38-06_Camera2/'
	data2 = parse_protobufs(path)    
	path.data_folder = 'TeleOpVRSession_2018-03-07_14-38-06_Camera3/'
	data3 = parse_protobufs(path)    
	path.data_folder = 'TeleOpVRSession_2018-03-07_14-38-06_Camera4/'
	data4 = parse_protobufs(path)    
	# format labels into n x 6 array
	for i in range(18000):
		print i
		path = struct()
		if i < 10000:
			t = i
			data = data0
		else:
			if i < 12000:
				t = i - 10000
				data = data1
			else:
				if i < 14000:
					t = i - 12000                
					data = data2
				else:
					if i < 16000:
						t = i - 14000                                 
						data = data3
					else:
						t = i - 16000                         
						data = data4
		num_items.append(len(data.states[i].items))
		img_name = str(data.states[i].snapshot.name)
		depth_name = img_name[:-4] + '-Depth.jpg'

		# read in rgb and depth images and add a new axis to them to indicate which snapshot index for each image
		rgb_img = np.expand_dims(cv2.imread(img_name, 1), axis=0)
		depth_img = np.expand_dims(cv2.imread(depth_name, 0), axis = 0)
        
		for j in range(num_items[i]):
            
			item_id = str(data.states[i].items[j].id)
			item_id_int = data.states[i].items[j].id            
			if item_id_int != 35: 
				continue
			else:
				print 666
			'''
			RGB label, classified by name
			input label (X)
			D label, classified by name
			input label (X)
			'''            
			if item_id not in data_dict_x:
				data_dict_x[item_id] = np.empty([0,299,299,3])
				data_dict_d[item_id] = np.empty([0,299,299])                
			data_dict_x[item_id] = np.vstack([data_dict_x[item_id], rgb_img])
			data_dict_d[item_id] = np.vstack([data_dict_d[item_id], depth_img])
            
			'''
			RGB-D label, classified by name
			Batch split
			'''
			if item_id not in batch_dict:
				batch_dict[item_id] = 0
                              
			# Output label (Y)
			rlabel = data.states[i].items[j] 
			current_label = [data.states[i].snapshot.name, rlabel.x, rlabel.y, rlabel.z, rlabel.roll, rlabel.pitch, rlabel.yaw]
			#print data.states[i].items[j].id
			labels.append(current_label)
			'''
			RGB label
			'''
			current_label_rgb = [rlabel.x, rlabel.y, rlabel.z]
			labels_rgb.append(current_label_rgb)
            
			'''
			RGB label, classified by name
			Output label (Y)
			'''
            
			if item_id not in data_dict_y:
				data_dict_y[item_id] = []
			data_dict_y[item_id].append(current_label_rgb)
            
			if len(data_dict_x[item_id]) == batch_size:
				batch = batch_dict[item_id]
				if i % 10 != 0:
					tmp_path = save_path_train
				else:
					tmp_path = save_path_test                   
				if not os.path.exists(tmp_path):
					os.makedirs(tmp_path)
                           
				np.save(tmp_path +"/"+ str(batch) +"_x.npy", data_dict_x[item_id])
				np.save(tmp_path +"/"+ str(batch) +"_d.npy", data_dict_d[item_id])       
				np.save(tmp_path +"/"+ str(batch) +"_y.npy", np.array(data_dict_y[item_id]))  
                
				train_path = "/home/zhouzixuan/notebooks/proj_new/3ddata/train/"
				test_path = "/home/zhouzixuan/notebooks/proj_new/3ddata/test/"
				if not os.path.exists(train_path):
					os.makedirs(train_path)
				if not os.path.exists(test_path):
					os.makedirs(test_path)
                    
				d_batch = data_dict_d[item_id]
				x_batch = data_dict_x[item_id]
				d_round = np.floor(d_batch/25.5)
				sess = tf.InteractiveSession()
				v = tf.transpose(tf.one_hot(d_round, depth=10, axis=2, on_value=1.0, off_value=0.0), perm=[0,1,3,2])
				v = v.eval()
				combine = np.empty([32, 299, 299, 3, 0])
				for i in range(10):
					i = 1
					v_tmp = v[:,:,:,i]
					v_tmp = np.transpose(np.broadcast_to(v_tmp,(3, 32, 299, 299)),(1,2,3,0))
					v_tmp = v_tmp == 1
					x_tmp = np.multiply(x_batch, v_tmp)
					x_cur = np.expand_dims(x_tmp, axis = 4)
					combine = np.concatenate((combine, x_cur), axis=4)
				np.save(train_path +"/"+ str(batch) +".npy", combine)       
				np.save(test_path +"/"+ str(batch) +"_y.npy", np.array(data_dict_y[item_id]))  
                
				data_dict_x[item_id] = np.empty([0,299,299,3])
				data_dict_d[item_id] = np.empty([0,299,299])                
				data_dict_y[item_id] = []               
				batch_dict[item_id] = 1 + batch
				
if __name__ == '__main__':
	parse_data(save=True)				
                       
