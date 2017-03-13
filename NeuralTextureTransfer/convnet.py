import scipy.misc
import numpy as np
import math
import tensorflow as tf
import constants
import cv2
import glob


class convnet(object):

    def __init__(self,batch_size=64,filter_size=5,first_layer_filters=64,filter_growth_factor=4,pool_factor=2):
        self.batch_size = batch_size
        self.filter_size = filter_size
        self.input_images = tf.placeholder(tf.float32,shape=[None,constants.IMAGE_SIZE,constants.IMAGE_SIZE,3])
        self.batch_labels = tf.placeholder(tf.float32,shape=[None,constants.NUM_CATEGORIES])
        self.first_layer_filters = first_layer_filters
        self.filter_growth_factor = filter_growth_factor
        self.pool_factor = pool_factor
        self.kp = tf.placeholder(tf.float32)
        self.filter_map = {}
        self.layer_map = {}


    def conv2d(self,input,in_channels,out_channels,name='NONE',non_linearity=tf.nn.relu):
        filters = tf.Variable(tf.truncated_normal([self.filter_size,self.filter_size,in_channels,out_channels],stddev=0.01))
        self.filter_map[name] = filters
        biases = tf.Variable(tf.truncated_normal([out_channels],stddev=0.01))
        return non_linearity(tf.nn.conv2d(input,filters,strides=[1, 1, 1, 1], padding='SAME')+biases)


    def max_pool(self,input,size=2):
        return tf.nn.max_pool(input,ksize=[1, size, size, 1],strides=[1, size, size, 1], padding='SAME')


    def fc(self,input,in_size,out_size,non_linearity=tf.nn.sigmoid):
        weights = tf.Variable(tf.truncated_normal([in_size,out_size],stddev=0.01))
        biases = tf.Variable(tf.truncated_normal([out_size],stddev=0.01))
        return non_linearity(tf.matmul(input,weights)+biases)


    def build(self):
        f0 = self.first_layer_filters
        f1 = f0*self.filter_growth_factor
        f2 = f1*self.filter_growth_factor
        result_dim = constants.IMAGE_SIZE/(self.pool_factor**3)

        self.h0 = self.conv2d(self.input_images,3,f0,name='h0') #32x32xf0
        self.h0_pool = self.max_pool(self.h0) #16x16xf0
        self.layer_map['h0']=self.h0_pool
        self.h1 = self.conv2d(self.h0_pool,f0,f1,name='h1') #16x16xf1
        self.h1_pool = self.max_pool(self.h1) #8x8xf1
        self.layer_map['h1'] = self.h1_pool
        self.h2 = self.conv2d(self.h1_pool,f1,f2,name='h2') #8x8xf2
        self.h2_pool = self.max_pool(self.h2) #4x4xf2
        self.layer_map['h2'] = self.h2_pool
        self.h2_flat = tf.reshape(self.h2_pool,[-1,result_dim*result_dim*f2])
        self.fc_layer = tf.nn.dropout(self.fc(self.h2_flat,result_dim*result_dim*f2,1024,tf.nn.relu),keep_prob=self.kp)
        self.out = self.fc(self.fc_layer,1024,constants.NUM_CATEGORIES,tf.identity)


    def build_loss(self):
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.out, labels=self.batch_labels))
        correct_prediction = tf.equal(tf.argmax(self.out,1), tf.argmax(self.batch_labels,1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


    def build_train_op(self):
        self.train_op = tf.train.AdamOptimizer(1e-4).minimize(self.loss)


    #load up the cifar-10 training data and test data, both images and labels
    def load_training_data(self):
        #self.images = np.empty([constants.NUM_IMAGES,constants.IMAGE_SIZE,constants.IMAGE_SIZE,3])
        #self.labels = np.zeros([constants.NUM_IMAGES,constants.NUM_CATEGORIES])
        #self.test_images = np.empty([constants.NUM_TEST_IMAGES, constants.IMAGE_SIZE, constants.IMAGE_SIZE, 3])
        #self.test_labels = np.zeros([constants.NUM_TEST_IMAGES, constants.NUM_CATEGORIES])

        # LOAD IN IMAGES AND CORRESPONDING LABELS
        #print('Loading png images from ' + constants.IMAGE_DIRECTORY)

        #im_num = 0
        category_num = 0
        fnames=[]
        iminds=[]
        for category in constants.CATEGORIES:
            catnames = glob.glob(constants.IMAGE_DIRECTORY + category + '*.png')
            #iminds.append(cumsum)
            iminds+=[category_num for name in catnames]
            fnames+=catnames
            category_num+=1
            
            #for fname in fnames:
            #    self.images[im_num, :, :, :] = np.float32(cv2.imread(fname)) / 128.0 - 1
            #    self.labels[im_num, category_num] = 1
            #    im_num += 1
            #category_num += 1

        self.im_names=fnames
        self.im_inds=iminds
        #im_num = 0
        category_num = 0
        test_names=[]
        test_inds=[]
        #cumsum=0
        for category in constants.CATEGORIES:
            catnames = glob.glob(constants.TEST_DIRECTORY + category + '*.png')
            #iminds.append(cumsum)
            test_inds+=[category_num for name in catnames]
            test_names+=catnames
            category_num+=1
            #for fname in fnames:
            #    self.test_images[im_num, :, :, :] = np.float32(cv2.imread(fname)) / 128.0 - 1
            #    self.test_labels[im_num, category_num] = 1
            #    im_num += 1
            #category_num += 1
        self.test_inds=test_inds
        self.test_names=test_names

    #fetch a batch of training data, index tells us which batch we're getting
    def get_batch(self,index):
        batch_images = np.empty([self.batch_size,constants.IMAGE_SIZE,constants.IMAGE_SIZE,3])
        batch_labels = np.zeros([self.batch_size,constants.NUM_CATEGORIES])
        #self.images[self.batch_size*index:self.batch_size*(index+1),:,:,:]
        #batch_labels = self.labels[self.batch_size*index:self.batch_size*(index+1),:]
        
        for i in range(self.batch_size):
            name=self.im_names[index*self.batch_size+i]
            batch_images[i,:,:,:]=np.float32(cv2.imread(name))/128.0-1
            batch_labels[i,self.im_inds[index*self.batch_size+i]]=1
        return batch_images,batch_labels

    def get_test_batch(self,index):
        batch_images = np.empty([self.batch_size,constants.IMAGE_SIZE,constants.IMAGE_SIZE,3])
        batch_labels = np.zeros([self.batch_size,constants.NUM_CATEGORIES])
        #self.images[self.batch_size*index:self.batch_size*(index+1),:,:,:]
        #batch_labels = self.labels[self.batch_size*index:self.batch_size*(index+1),:]
        for i in range(self.batch_size):
            name=self.test_names[index*self.batch_size+i]
            batch_images[i,:,:,:]=np.float32(cv2.imread(name))/128.0-1
            batch_labels[i,self.test_inds[index*self.batch_size+i]]=1
        return batch_images,batch_labels

    #http://stackoverflow.com/questions/4601373/better-way-to-shuffle-two-numpy-arrays-in-unison
    #We want to randomize the order of the training data in each epoch, but we have to be careful
    #to shuffle the labels and the samples in the same way, or else we'll just have nonsense
    def shuffle_data(self):
        rng_state = np.random.get_state()
        np.random.shuffle(self.im_names)
        np.random.set_state(rng_state)
        np.random.shuffle(self.im_inds)


    def save_activations(self,name,count,sess,batch_images):
        activation_values = sess.run(self.layer_map[name],feed_dict={self.input_images: batch_images,self.kp: 1})
        print(np.amax(activation_values))
        padded_activations = np.lib.pad(activation_values,((0,0),(1,1),(1,1),(0,0)),'constant',constant_values=-1)
        padded_batch = np.lib.pad(batch_images,((0,0),(1,1),(1,1),(0,0)),'constant',constant_values=-1)

        num_images = batch_images.shape[0]
        num_filters = activation_values.shape[3]
        scale_factor = 2
        dim = padded_batch.shape[1]*scale_factor

        output_image = np.zeros((num_images*dim,(1+num_filters)*dim,3))

        for i in range(num_images):

            b_image = scipy.misc.toimage(padded_batch[i,:,:],cmin=-1,cmax=1)
            b_resized = scipy.misc.imresize(b_image,(dim,dim,3),interp='nearest')

            output_image[i*dim:(i+1)*dim,0:dim,:] = b_resized

            for j in range(num_filters):
                a_image = scipy.misc.toimage(padded_activations[i,:,:,j],cmin=0,cmax=np.amax(activation_values))
                a_resized = scipy.misc.imresize(a_image,(dim,dim),interp='nearest')
                a_3channel = np.empty((dim,dim,3))
                a_3channel[:,:,0] = a_resized
                a_3channel[:,:,1] = a_resized
                a_3channel[:,:,2] = a_resized

                output_image[i*dim:(i+1)*dim,(1+j)*dim:(2+j)*dim,:] = a_3channel

        image = scipy.misc.toimage(output_image,cmin=0,cmax=255)
        image.save("{:s}_{:05d}_activations.png".format(name,count));


    def save_filters(self,name,count,sess):
        filter_values = sess.run(self.filter_map[name])

        padded = np.lib.pad(filter_values,((1,1),(1,1),(0,0),(0,0)),'constant',constant_values=-0.1)
        scale_factor = 4
        scaled_size = scale_factor*padded.shape[0]
        depth = filter_values.shape[3]
        output_image = np.zeros((scaled_size * 8,scaled_size * int(math.ceil(depth/8.0)),3))
        for i in range(depth):
            x = i%8
            y = i//8
            image = scipy.misc.toimage(padded[:,:,:,i],cmin=-0.1,cmax=0.1)
            resized = scipy.misc.imresize(image,(scaled_size,scaled_size,3),interp='nearest')
            output_image[x*scaled_size:(x+1)*scaled_size,y*scaled_size:(y+1)*scaled_size,:] = resized

        image = scipy.misc.toimage(output_image,cmin=0,cmax=255)
        image.save("{:s}_{:05d}_filters.png".format(name,count));


    def train(self,epochs,sess):
        tf.global_variables_initializer().run()

        sample_batch,sample_labels = self.get_batch(0)
        sample_batch = np.copy(sample_batch)

        count = 0
        for ep in range(epochs):
            self.shuffle_data()
            num_batches = constants.NUM_IMAGES//self.batch_size
            for batch_index in range(num_batches):
                batch_images,batch_labels = self.get_batch(batch_index)
                _,loss,output = sess.run([self.train_op,self.loss,self.out],feed_dict={self.input_images: batch_images,self.batch_labels: batch_labels,self.kp: 0.5})
                count+=1
                if(count%10==0 and count<150):
                    print ("count = "+str(count))
                if(count%1000==0):
                    print ("Starting Validation")
                    self.save_filters('h0',count,sess)
                    #self.save_activations('h0',count,sess,sample_batch)
                    #self.save_activations('h1',count,sess,sample_batch)
                    num_test_batches=constants.NUM_VALIDATION_IMAGES
                    accuracy=0.0
                    for test_batch_index in range(num_test_batches):
                        if (test_batch_index%25==0):
                            print ("test batch index = "+str(test_batch_index))
                        test_batch,test_labels=self.get_test_batch(test_batch_index)
                        batch_accuracy= sess.run(self.accuracy,feed_dict={self.input_images: test_batch,self.batch_labels: test_labels,self.kp: 1})
                        accuracy+=batch_accuracy/num_test_batches
                    print("Epoch {:3d}, batch {:3d} - Accuracy={:0.4f} Batch_Loss={:0.4f}".format(ep,batch_index,accuracy,loss))


with tf.Session() as sess:
    network = convnet(first_layer_filters=64,filter_growth_factor=2)
    network.build()
    network.load_training_data()
    network.build_loss()
    network.build_train_op()
    network.train(50,sess)
