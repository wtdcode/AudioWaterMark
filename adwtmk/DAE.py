from pydub import AudioSegment
import sys
import tensorflow as tf
import numpy as np
from adwtmk.audio import Audio
from adwtmk.encoder import *
from adwtmk.decoder import *
class DAE(object):
    def __init__(self,model_name):
        self.model_name = model_name
        self.process = 0
        self.loss = 0
        self.origin_loss = 0
        self.core_size = 3
        self.batch_size = 600
        self.Epoches = 100

    def _get_batches(self,batch_size,data,core_size):
        assert batch_size % core_size == 0
        dim_0 = len(data)
        #print("dim_0:",dim_0)
        length = len(data[0])
        num_batches = length // batch_size
        remainder_length = length % batch_size
        res = list()
        for i in range(num_batches):
            res.append(data[:,i*batch_size:(i+1)*batch_size])
        res = [np.array(x,np.float64).reshape(dim_0,batch_size//core_size,core_size) for x in res]
        remainder = data[:,-remainder_length:]
        return res,remainder 

        


    #np.set_printoptions(threshold=1e6)
    #def _my_config():
        #core_size = 5
        #batch_size = 500
        #Epoches = 200

    def fast_training(self,sound):
        self.core_size = 100
        self.batch_size = 1000
        self.Epoches = 50
        self._main(sound,100,1000,50)

    def medium_training(self,sound):
        self.core_size = 5
        self.batch_size = 500
        self.Epoches = 100
        self._main(sound,5,500,100)

    def slow_training(self,sound):
        self.core_size = 3
        self.batch_size = 300
        self.Epoches = 100
        self._main(sound,3,300,150)

    def get_train_result_music_file(self):
        if (self.new_sound):
            return self.new_sound
        else:
            raise Exception("You should run training firstly !")

    def get_current_training_process(self):
        return self.process

    def test(self,sound):
        audio_matrix = sound.get_reshaped_samples()
        #max_value = np.max(audio_matrix)
        #min_value = np.min(audio_matrix)
        #audio_matrix = (audio_matrix-min_value) / (max_value-min_value)
        mean_value = np.mean(audio_matrix)
        std_value = np.std(audio_matrix)
        audio_matrix = (audio_matrix-mean_value) / std_value
        channels = len(audio_matrix)
        batches,remainder = self._get_batches(batch_size=self.batch_size,core_size=self.core_size,data=audio_matrix)
        losses = list()
        for i in range(len(batches)):
            dropout_indicator = np.random.rand()
            if (dropout_indicator <= 0.2):
                losses.append(np.sum(abs(batches[i])))
                batches[i] *= 0.00
        losses.append(0)
        sum_losses = np.sum(np.array(losses).reshape(-1))
        #print("losses:")
        #print(np.array(losses).reshape(-1))
        #print(sum_losses)
        test_batches = np.array(batches,np.float64).reshape(channels,-1)
        test_batches = np.concatenate((test_batches,remainder),axis=1)
        count = audio_matrix.shape
        count = count[0]*count[1]
        self.origin_loss = sum_losses/(float)(count)
        test_batches = test_batches * std_value + mean_value
        test_sound = sound.spawn(test_batches)
        self._main(test_sound,self.core_size,self.batch_size,1,1.0)
        return test_sound,self.new_sound



    def _main(self,sound,core_size,batch_size,Epoches,drop_out_rate=0.9):
        self.new_sound = None
        self.process = 0
        self.loss = 0

        #print(sound.frame_rate,sound.duration_seconds, len(sound.get_array_of_samples()))

        audio_matrix = sound.get_reshaped_samples()
        #max_value = np.max(audio_matrix)
        #min_value = np.min(audio_matrix)
        #audio_matrix = (audio_matrix-min_value) / (max_value-min_value)
        mean_value = np.mean(audio_matrix)
        std_value = np.std(audio_matrix)
        audio_matrix = (audio_matrix-mean_value) / std_value


        batches,remainder = self._get_batches(batch_size=batch_size,core_size=core_size,data=audio_matrix)

        steps = batch_size // core_size
        channels = len(audio_matrix)


        best_output = ""

        with tf.Session() as sess:
            fw_cell = tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.BasicLSTMCell(core_size),drop_out_rate)
            fw_rnn_cell = tf.contrib.rnn.MultiRNNCell([fw_cell]*2) 
            bw_cell = tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.BasicLSTMCell(core_size),drop_out_rate)
            bw_rnn_cell = tf.contrib.rnn.MultiRNNCell([bw_cell]*2) 
            input_data = tf.placeholder(shape=[channels,steps,core_size],dtype=tf.float64)
            in_weights = tf.get_variable(name="in_weight",shape=[steps*core_size,steps*core_size],dtype=tf.float64)
            in_bias = tf.get_variable(name="in_bias",shape=[core_size*steps],dtype=tf.float64)
            hidden_data = tf.tanh(tf.nn.xw_plus_b(tf.reshape(input_data,(channels,-1)),in_weights,in_bias))
            hidden_data_out = tf.reshape(hidden_data,[channels,steps,core_size])
            bi_outputs,last_state = tf.nn.bidirectional_dynamic_rnn(fw_rnn_cell,bw_rnn_cell,hidden_data_out,dtype=tf.float64)
            out_weights = tf.get_variable(name="out_weight",shape=[steps*core_size*2,steps*core_size],dtype=tf.float64)
            out_bias = tf.get_variable(name="out_bias",shape=[core_size*steps],dtype=tf.float64)
            outputs = tf.nn.xw_plus_b(tf.reshape(tf.concat(bi_outputs,2),(channels,-1)),out_weights,out_bias)
            #outputs,last_state = tf.nn.dynamic_rnn(fw_rnn_cell,input_data,dtype=tf.float64)
            loss = tf.reduce_mean(tf.sqrt(tf.squared_difference(tf.reshape(input_data,(channels,-1)),outputs)))
            train = tf.train.AdamOptimizer(0.001).minimize(loss)
            saver = tf.train.Saver()
            train_loss = 999999999
            try:
                saver.restore(sess,self.model_name)
                print("model restored")
            except:
                sess.run(tf.global_variables_initializer())
                print("restore failed, randomly initialize")
            for i in range(Epoches):
                loss_temp = 0
                outputs_temp = list()
                for item in batches:
                   if (drop_out_rate < 1):
                           epoch_outputs,epoch_loss,_ = sess.run([outputs,loss,train],feed_dict={
                               input_data:item
                               }) 
                   else:
                           epoch_outputs,epoch_loss = sess.run([outputs,loss],feed_dict={
                               input_data:item
                               }) 
                   loss_temp += epoch_loss
                   outputs_temp.append(epoch_outputs)
                loss_temp /= len(batches)
                if (i == 0 and drop_out_rate<1):
                    self.origin_loss = loss_temp
                self.process = i/Epoches
                self.loss = loss_temp
                #print("process:%f,loss:%f" % (i/Epoches,loss_temp))
                if (loss_temp < train_loss):
                    train_loss = loss_temp
                    if (drop_out_rate < 1):
                        saver.save(sess,self.model_name)
                    best_output = outputs_temp
            #best_output = best_output.append(remainder)
            best_output = np.array(best_output,np.float64).reshape(channels,-1)
            best_output = np.concatenate((best_output,remainder),axis=1)
            #best_output = best_output.T
            #best_output = best_output.reshape(-1)
            best_output = best_output*std_value+mean_value
            #best_output *= max_value-min_value
            #best_output += min_value

        self.new_sound = sound.spawn(best_output)
        #new_sound.export("test.flac","flac")
        #ex.add_artifact(filename="./test.flac")
        #ex.add_artifact(filename="./rnn_model_key_multirnn_bi_input.ckpt*")
        #audio_matrix = np.array(audio_matrix,np.float64).reshape(channels,-1)
        #audio_matrix = audio_matrix.T
        #audio_matrix = audio_matrix.reshape(-1)
        #audio_matrix = audio_matrix * (max_value-min_value)+min_value
        audio_matrix = audio_matrix * std_value + mean_value
        new_sound = sound.spawn(audio_matrix)
        #new_sound.export("test2.flac","flac")

    #sound = Audio.from_file("./mark.flac", format="flac")
    #fast_training(sound)
