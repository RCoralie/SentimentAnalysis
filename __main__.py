import sys
import argparse
import logging
import tensorflow as tf

from os.path import isfile
from .defaults import Config
from .utils import GloveDict
from .model import SentimentAnalysis

tf.logging.set_verbosity(tf.logging.ERROR)


def process_args(args, defaults):

    parser = argparse.ArgumentParser()
    parser.prog = 'sentimentAnalysis'
    subparsers = parser.add_subparsers(help='Subcommands.')

    # Global arguments
    parser.add_argument('--log-path', dest='log_path',
                        type=str, default=defaults.LOG_PATH,
                        help=('Log file path, default=%s' % (defaults.LOG_PATH)))

    # Init resources
    parser_init = subparsers.add_parser('init', help='Create all resources needed if they don\'t already exist.')
    parser_init.set_defaults(phase='init')

    # Training (on tweets corpus)
    parser_train = subparsers.add_parser('trainOnTweetsCorpus', help='Create all resources needed if they don\'t already exist.')
    parser_train.set_defaults(phase='trainOnTweetsCorpus')

    # Prediction (on twitter)
    parser_predict = subparsers.add_parser('opinionAnalysisOnTwitter', help='Public opinion of the latest tweets on the subjects.')
    parser_predict.set_defaults(phase='opinionAnalysisOnTwitter')
    parser_predict.add_argument('filters', metavar='filters', nargs='+', type=str, help=('Topic and filters for twitter search.'))

    parameters = parser.parse_args(args)
    return parameters


def main(args=None):

    if args is None:
        args = sys.argv[1:]

    parameters = process_args(args, Config)
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)-15s %(name)-5s %(levelname)-8s %(message)s',
        filename=parameters.log_path)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)-15s %(name)-5s %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

    if parameters.phase == 'init':
        # If the file is not present in the resource, creates it 
        if not isfile(Config.GLOVE_DICT_FILE):
          print 'Creation of the glove dictionnary file...'
          GloveDict.createGloveDic()

    if parameters.phase == 'trainOnTweetsCorpus':
        if not isfile(Config.SENTIMENT_ANALYSIS_MODEL):   
          print 'Create the sentiment analysis model on tweets corpus :'
          SentimentAnalysis.createModel()

    if parameters.phase == 'opinionAnalysisOnTwitter':
        # Give public opinion on this topic
        publicOpinion = SentimentAnalysis.opinionAnalysisOnTwitter(parameters.filters)
        print "Number of tweets finded : %d" %(publicOpinion["tweetsNb"])
        print "Opinion score : %f" %(publicOpinion["score"])
        if publicOpinion["score"] >= 0.5 :
            print "Good opinion !"
        else :
            print "Bad opinion !"


if __name__ == "__main__":
    main()
