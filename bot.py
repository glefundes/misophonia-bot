import os
import sys
import bisect
import logging

sys.path.append('../')
from model import MisophoniaModel
from utils import normalize_waveform

import torch
import torchaudio

import telegram
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters


def build_message(score):
    if score == -1:        
        return 'I think the audio is silent, but might as well not listen just to be safe ;)'

    msg_index = bisect.bisect(confidence_thresholds, score) - 1
    #     pred_class = 'noise' if int(round(score)) == 1 else 'clean'
    return messages[msg_index] + f'\n({score:.04f})' 
    
def run_inference(model, audio_file):
    waveform, sample_rate = torchaudio.load(audio_file)
    # Is audio silent?
    if waveform.var() < 1e-6:
        return -1
    # Normalize waveform
    waveform = normalize_waveform(waveform).permute(1,0)
    # Pad waveform if shortan than the minimum length acceptable for the model to handle
    if waveform.shape[0] < model.min_len:
        padded = torch.zeros(model.min_len, waveform.shape[1])
        padded[:waveform.shape[0], :] = waveform            
        waveform = padded
    # Run inference
    length = torch.tensor(waveform.shape[0])
    with torch.no_grad():
        pred = model(waveform.unsqueeze(0), length.unsqueeze(0))
    pred = pred.squeeze().item()
    return pred

def parse_audio(update, context):
    try:
        audio_obj = [update.message.audio, update.message.voice]
        audio_obj = next(item for item in audio_obj if item is not None)
        audio_file = audio_obj.get_file().download('tmpfile.oga')  
        
        score = run_inference(model, audio_file)
        msg = build_message(score)       
        
        context.bot.send_message(chat_id=update.effective_chat.id, reply_to_message_id=update.message.message_id, text=msg)
    except StopIteration:
        context.bot.send_message(chat_id=update.effective_chat.id, text='No valid audio data could be retrieved :(')

def start(bot, update):
    update.effective_message.reply_text("Hi! I'm Misophonia bot. Send me an audio or voice message and I'll try to tell you if it's safe to listen. Add me to groups and I'll do the same for every audio and voice message :)")

def error(bot, update, error):
    logger.warning('Update "%s" caused error "%s"', update, error)
    
if __name__ == "__main__":
    # Set these variable to the appropriate values
    TOKEN = "1282105586:AAFKV1hssrLR5JXmkImU4io6KvtzMiUhqb8"
    NAME = "The name of your app on Heroku"
    MODEL_PATH = 'weights.pth'
    
    confidence_thresholds = [0, 0.2, 0.45, 0.6, 0.8, 1]
    messages = [
        "I'm pretty sure this audio is safe :D",
        "This audio is probably safe :)",
        "Sorry, I can't be certain about this one :/",
        "This audio is probably unsafe :(",
        "Oof, that is some terrible noise D:"
    ]

    # Port is given by Heroku
#     PORT = os.environ.get('PORT')

    # Enable logging
    logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                        level=logging.INFO)
    logger = logging.getLogger(__name__)

    model = MisophoniaModel()
    state_dict = torch.load(MODEL_PATH,  map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)
    model.eval()
    
    # Set up the Updater
    updater = Updater(token=TOKEN, use_context=True)
    dp = updater.dispatcher
    # Add handlers
    dp.add_handler(CommandHandler('start', start))
    dp.add_error_handler(error)
    dp.add_handler(MessageHandler(Filters.audio | Filters.voice, parse_audio))

    # Start the webhook
#     updater.start_webhook(listen="0.0.0.0",
#                           port=int(PORT),
#                           url_path=TOKEN)
#     updater.bot.setWebhook("https://{}.herokuapp.com/{}".format(NAME, TOKEN))
#     updater.idle()
    updater.start_polling()
    updater.idle()
    # updater.stop()