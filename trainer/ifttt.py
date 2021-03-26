"""
IFTTT Callbacks
"""
from requests import post
from tensorflow.keras.callbacks import Callback


class IFTTTTrainingCompleteCallback(Callback):
    ifttt_key = 'bxoTj4tM935zzKaQcJUQtO'
    event_name = 'training_complete'

    def __init__(self, jobname):
        self.jobname = jobname

    def _call_ifttt_webhook(self, value1=None, value2=None, value3=None):
        post(f'https://maker.ifttt.com/trigger/{self.event_name}/with/key/{self.ifttt_key}', 
            json={'value1': value1, 'value2': value2, 'value3': value3})

    def on_train_end(self, logs):
        print('Sending IFTTT message...')
        logstr = ', '.join(f'{key}: {value:.02f}' for key,value in logs.items())
        self._call_ifttt_webhook(value1=self.jobname, value2=logstr)