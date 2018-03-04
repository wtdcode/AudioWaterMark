from adwtmk.encoder import *
from adwtmk.decoder import *
from adwtmk.DAE import DAE
from threading import Thread
import time

# read sound file
sound = Audio.from_file("./mark.flac", format="flac")
dae = DAE("./Neural_sign.ckpt")
# recommend to train the model in another thread in order not to crush the UI thread.
try:
    t = Thread(target=dae.slow_training,args=(lsb_marked,))
    t.start()
    # The exact neural sign is the file , the name of which begins with "Neural_sign.ckpt"
    # you can use fast/slow/medium_training, fast - high speed but most noisy
except:
# in case if there is any problem when the training thread runs
    pass

#time.sleep(100)
#An asynchronous example for getting the training process, you can use it for processbar (hhhh)
print(dae.get_current_training_process())
try:
    #syncronize()
    t.join()
    # The test_sound is the sound that suffer from the cutting attack, and the output_sound is the denoised sound
    test_sound,output_sound = dae.test(sound) 
    #To get mathematically exact test loss for the sound, remember to call dae.test(sound) firstly
    print("attack error:%f, output error rate:%f"%(dae.original_loss,dae.loss))
except Exception as e:
    print("During the test:")
    print(e)
    
