import keras
import time

class LeenaLogger(keras.callbacks.Callback):

    def __init__(self, fname_prepend="log"):
        self.logname = "./logs/" + fname_prepend + "_" + time.strftime("%Y%m%d_%H%M%S") + ".csv"
        self.f = open(self.logname, 'w')
        self.header_written = False

#    def on_train_begin(self, logs={}):
#        return
#
#    def on_train_end(self, logs={}):
#        return
#
#    def on_epoch_begin(self, logs={}):
#        return

    def on_epoch_end(self, epoch, logs={}):
        if not self.header_written:
            self.header_keys = sorted(logs.keys())
            self.f.write("epoch," + (','.join(self.header_keys)) + "\n")
            self.header_written = True
        batch_line = [str(epoch)]
        for k in self.header_keys:
            batch_line.append(str(logs[k]))
        self.f.write(','.join(batch_line) + "\n")
        #print(logs)
        #self.losses.append(logs.get('loss'))
        return

#    def on_batch_begin(self, batch, logs={}):
#        return

    def on_batch_end(self, batch, logs={}):
        return
