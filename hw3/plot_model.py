
import argparse
from keras.utils import plot_model
from keras.models import load_model
import matplotlib.pyplot as plt
def main():
    parser = argparse.ArgumentParser(prog='plot_model.py',
            description='Plot the model.')
    parser.add_argument('--model',type=str,default='CNN_params/CNN_model.h5')
    args = parser.parse_args()

    emotion_classifier = load_model(args.model)
    emotion_classifier.summary()
    plot = plot_model(emotion_classifier,to_file='model.png')

if __name__=='__main__':
    main()