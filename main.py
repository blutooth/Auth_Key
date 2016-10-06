from sklearn.metrics import log_loss,accuracy_score
from sklearn.metrics import confusion_matrix
import numpy as np
import auth_algorithms as aa
import sys

def test_bed(train,test):
    # log-likelihood, accuracy and confusion matrix counts
    ll       = 0;
    acc      = 0;
    cm_total = np.array([[0,0],[0,0]])


    # get the trained models, the test data, the users, and the output values
    templates,instances,users,answers=aa.train_test(train,test)

    # loop over users computing test statistics
    for user in users:
        print()
        print('testing classifier on user ', user)

        # test data
        X_testing = instances[user]
        # score and thresh-hold
        score, thresh = aa.authenticate(X_testing,user,templates)
        # actual Y values
        Y_testing = answers[user]

        ## compute user statistics
        # log likelihood
        ll_u = log_loss(Y_testing,score)
        ll  +=  ll_u
        # accuracies
        predictions = [int(x>thresh) for x in score[:,1]]
        acc_u       = accuracy_score(predictions,Y_testing)
        acc        += acc_u
        # confusion matrix
        cm_user  = confusion_matrix(Y_testing,predictions)
        cm_total += cm_user

        # print user stats
        print('accuracy: ', acc_u)
        print('log-likelihood: ', ll_u)
        print('Confusion matrix: elem (i,j) i - actual label, j- predicted')
        print(cm_user)

    # average stats
    acc /= len(users)
    ll /= len(users)

    # print total stats
    print()
    print('Average Over all Users')
    print('Ave accuracy: ', acc)
    print('Ave log-likelihood: ', ll)
    print('Total Confusion matrix: elem (i,j) i - actual label, j- predicted')
    print(cm_user)


if __name__ =='__main__':
    if len(sys.argv)==3:
        test_bed(train=sys.argv[1],test=sys.argv[2])
    else:
        test_bed(train="dataset_training.csv",test="dataset_testing.csv")