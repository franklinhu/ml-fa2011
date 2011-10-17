import pickle
import os


#########################################
  
def munge_Boolean(email_file,features):
    return []

def munge_NTF(email_file,features):
    return []

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
