import os

for i in range(10):
    print('fold %d' % (i))
    cmd = "python train.py --fold %d" % (i)
    os.system(cmd)
#print("best_Test_acc_fold", best_Test_acc_fold)
