# Franklin Hu, Sunil Pedapudi
# CS 194-10 Machine Learning
# Fall 2011
# Assignment 4
import pickle
import os


#########################################
  
def munge_Boolean(email_file,features):
    """
    Returns a vector representation of the email based on features

    Arguments:
    email_file -- name of a preprocessed email file
    features_file -- name of a file containing pickled list of tokens
    """
    features = _get_features(features_file)
    freq = _munge_frequency(email_file)
    output = []
    for f in features:
    if freq[f] == 0
      output.append(0)
    else:
      output.append(1)
    return output

def munge_NTF(email_file,features):
    """
    Returns a vector representation of the email based on features

    Arguments:
    email_file -- name of a preprocessed email file
    features_file -- name of a file containing pickled list of tokens
    """
    features = _get_features(features_file)
    freq = _munge_frequency(email_file)
    output = []
    for f in features:
    output.append(freq[f])
    return output

def _munge_frequency(email_file):
    """
    Counts the frequency of terms an email file
    """
    email = open(email_file, 'r')
    freq = defaultdict(int)
    for line in email:
        tok = line.strip(" ").split(" ")
        for t in tok:
            freq[t] += 1
    email.close()

    return freq

def _get_features(features_file):
    handle = open(features_file, 'r')
    features = pickle.load(handle)
    handle.close()
    return features

def NBclassify_Boolean(example,model,cost_ratio):
    return 0

def NBclassify_NTF(example,model,cost_ratio):
    return 0


#########################################


def get_files(path):
    for f in os.listdir(path):
        f = os.path.abspath( os.path.join(path, f ) )
        if os.path.isfile( f ):
            yield f


class NaiveBayesModel:
    
    def __init__(self, features_file, model_file):
        self.features = pickle.load(open(features_file,'rb'))
        self.model = pickle.load(open(model_file,'rb'))

    def test(self, spam_dir, ham_dir, cost_ratio):
        N = 0
        loss = 0.
        for f in get_files(spam_dir):
            N += 1
            classification = self.classify(self.munge(f),cost_ratio)
            if not (classification==1):
                loss += 1
    
        for f in get_files(ham_dir):
            N += 1
            classification = self.classify(self.munge(f),cost_ratio)
            if not (classification==0):
                loss += cost_ratio
        
        print "Classifier average loss: %f" % (loss/N)


class NB_Boolean(NaiveBayesModel):
    
    def classify(self,example,cost_ratio):
        return NBclassify_Boolean(example,self.model,cost_ratio)
        
    def munge(self,email_file):
        return munge_Boolean(email_file,self.features)


class NB_NTF(NaiveBayesModel):    
    
    def classify(self,example,cost_ratio):
        return NBclassify_NTF(example,self.model,cost_ratio)
    
    def munge(self,email_file):
        return munge_NTF(email_file,self.features)


#########################################
