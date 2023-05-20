# Import some modules
import sys, os, pickle, argparse, importlib, datetime
sys.path.insert(0,os.path.join('.','lib'))          # add a directory to search functions
sys.path.insert(0,os.path.join('.','detections'))    # add a directory to search functions
import functions, gc
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # disable tensorflow warnings
import tensorflow as tf, numpy as np, matplotlib.pyplot as plt, pandas as pd
from tensorflow import keras
from keras.utils import np_utils
from tensorflow.keras import optimizers
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import Dense, Activation, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from DenseNoise import DenseNoise # noise that will affect NN
from ConvolutionalNoise import Conv2DNoise # noise that will affect NN
from tensorflow.python.ops.parallel_for import jacobian as tf_jacobian
from tensorflow.python.ops.parallel_for import batch_jacobian as tf_batch_jacobian
from PIL import Image

seed = 1337
np.random.seed(seed) # for reproducibility
sess = tf.compat.v1.InteractiveSession()

# Use argparse to check what the user is trying to do
parser = argparse.ArgumentParser(add_help=False)
parser.add_argument('-ds', '--dataset', action='store', dest='dataset', type=str, default=None)
parser.add_argument('-m', '--model', action='store', dest='model', type=str, default=None)

parser.add_argument('-a', '--attack', action='store', dest='attack', type=str, default=None)
parser.add_argument('-eta', '--eta', action='store', dest='eta', type=float, default=0.0)
parser.add_argument('-eps', '--epsilon', action='store', dest='epsilon', type=float, default=0.0)
parser.add_argument('-kap', '--kappa', action='store', dest='kappa', type=float, default=0.0)
parser.add_argument('-cte', '--cte_init', action='store', dest='cte_init', type=float, default=0.0)
parser.add_argument('-cteupd', '--cte_update', action='store', dest='cte_update', type=float, default=0.1)
parser.add_argument('-ns', '--n_succ', action='store', dest='n_succ', type=int, default=50)
parser.add_argument('-mi', '--max_iter', action='store', dest='max_iter', type=int, default=20)

parser.add_argument('-d', '--detection', action='store', dest='detection', type=str, default=None)
parser.add_argument('-xs', '--x_start', action='store', dest='x_start', type=float, default=0.0)
parser.add_argument('-xe', '--x_end', action='store', dest='x_end', type=float, default=1.8)
parser.add_argument('-dx', '--dx_plot', action='store', dest='dx_plot', type=float, default=0.2)
parser.add_argument('-ys', '--y_start', action='store', dest='y_start', type=float, default=0.0)
parser.add_argument('-ye', '--y_end', action='store', dest='y_end', type=float, default=0.18)
parser.add_argument('-dy', '--dy_plot', action='store', dest='dy_plot', type=float, default=0.02)
parser.add_argument('-ni', '--noise_iter', action='store', dest='noise_iter', type=int, default=100)

parser.add_argument('-s', '--save', action='store', dest='save', type=str, default="yes")
parser.add_argument('-ow', '--overwrite', action='store', dest='overwrite', type=str, default="no")
args = parser.parse_args()

# Check args typed by user
datasets = ["MNIST","CIFAR10","DR", "smallDR"]
attacks = ["deepfool", "FGSM", "CW2"]
detections = ["stoGauNet"]
save = True
overwrite = False
print("\n\nChecking arguments errors...")
if not (args.dataset in datasets):
    sys.exit("Dataset argument should be one the following: {}".format(datasets))
if not args.model:
    sys.exit("Model argument cannot be empty")
if (not (args.attack in attacks)) and args.attack != None:
    sys.exit("Attack argument should be one the following: {}".format(attacks))
if (not (args.detection in detections)) and args.detection != None:
    sys.exit("Detection argument should be one of the following: {}".format(detections))
if not args.attack and args.detection:
    sys.exit("You cannot specify defence without attack")
if args.save.lower() == "false" or args.save.lower() == "no" or args.save == "0":
    save = False
if args.overwrite.lower() == "true" or args.overwrite.lower() == "yes" or args.overwrite == "1":
    overwrite = True

# Define some paths
path_code_model_rel = "/models/"+args.dataset+"-model_" + args.model + ".py"
path_code_model = os.path.abspath(os.curdir) + path_code_model_rel
if args.attack:
    path_code_attack = os.path.abspath(os.curdir) + "/attacks/" + args.attack + ".py"

path_train_data_rel = "/gen_data/"+args.dataset+"-model_"+args.model
model_name = "-model.h5"
history_name = "-history.pickle"
conf_mat_name = "-conf_mat.npy"
path_model = os.path.abspath(os.curdir) + path_train_data_rel + model_name
path_history = os.path.abspath(os.curdir) + path_train_data_rel + history_name
path_conf_mat = os.path.abspath(os.curdir) + path_train_data_rel + conf_mat_name
if args.attack:
    path_attack_rel = "/gen_data/"+args.dataset+"-model_"+args.model+"-"+args.attack
    if args.attack == "deepfool":
        if args.eta == 0:
            sys.exit("You have to specify parameter eta when using deepfool attack")
        path_attack_rel += "_eta_" + functions.float_to_str(args.eta)
        path_attack_rel += "_mi_" + str(args.max_iter)
    elif args.attack == "FGSM":
        if args.epsilon == 0:
            sys.exit("You have to specify parameter epsilon when using FGSM attack")
        path_attack_rel += "_eps_" + functions.float_to_str(args.epsilon)
    elif args.attack == "CW2":
        if args.kappa == 0 or args.cte_init == 0:
            sys.exit("You have to specify parameters kappa and cte_init when using CW2 attack")
        path_attack_rel += "_kap_" + functions.float_to_str(args.kappa)
        path_attack_rel += "_cte_" + functions.float_to_str(args.cte_init)
        path_attack_rel += "_cteupd_" + functions.float_to_str(args.cte_update)
        path_attack_rel += "_mi_" + str(args.max_iter)
    path_attack_rel += "_ns_" + str(args.n_succ)
    path_detection_rel = path_attack_rel
    path_attack_rel += ".npz"
    path_attack = os.path.abspath(os.curdir) + path_attack_rel
if args.detection:
    path_detection_rel += "-" + args.detection
    if args.detection == "stoGauNet":
        noise_iter = args.noise_iter
        x_start, x_end, dx_plot = args.x_start, args.x_end, args.dx_plot   # Dense
        y_start, y_end, dy_plot = args.y_start, args.y_end, args.dy_plot   # Convolutional
        path_detection_rel += "_xs_" + functions.float_to_str(x_start)
        path_detection_rel += "_xe_" + functions.float_to_str(x_end)
        path_detection_rel += "_dx_" + functions.float_to_str(dx_plot)
        path_detection_rel += "_ys_" + functions.float_to_str(y_start)
        path_detection_rel += "_ye_" + functions.float_to_str(y_end)
        path_detection_rel += "_dy_" + functions.float_to_str(dy_plot)
        path_detection_rel += "_ni_" + str(noise_iter)
        path_detection_rel += ".npz"
    path_detection = os.path.abspath(os.curdir) + path_detection_rel

# Print what the program is going to do
os.system('clear')
print("dataset:\t\t\t\t{}".format(args.dataset))
print("model:\t\t\t\t\t{}".format(args.model))
print("path of model used:\t\t\t.{}".format(path_code_model_rel))
print("training data will be saved in:\t\t.{}".format(path_train_data_rel+model_name))
print("training data will be saved in:\t\t.{}".format(path_train_data_rel+history_name))
print("training data will be saved in:\t\t.{}".format(path_train_data_rel+conf_mat_name))
print("")
if args.attack:
    print("attack:\t\t\t\t\t{}".format(args.attack))
    print("attack data will be saved in:\t\t.{}".format(path_attack_rel))
    print("")
if args.detection:
    print("detection method:\t\t\t{}".format(args.detection))
    print("detection data will be saved in:\t.{}".format(path_detection_rel))
    print("")
print("The results will be saved" if save else "The results will NOT be saved")
print("The results will NOT overwrite others with same name" if not overwrite else "The results will overwrite others with same name")
print("")

# Load dataset
if args.dataset == "MNIST": # load MNIST dataset
    from tensorflow.keras.datasets import mnist
    
    nb_classes = 10
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_train = X_train[..., np.newaxis]
    X_test = X_test[..., np.newaxis]

    # Data normalization
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255

    # Convert class vectors to binary class matrices
    Y_train = tf.keras.utils.to_categorical(y_train, nb_classes)
    Y_test = tf.keras.utils.to_categorical(y_test, nb_classes)


    img_rows, img_cols, img_channels = X_train.shape[1:4]
    input_shape = (img_rows, img_cols, img_channels)

elif args.dataset == "CIFAR10": # load CIFAR10 dataset
    from tensorflow.keras.datasets import cifar10

    nb_classes = 10
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()

    # Data normalization
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255

    # Convert class vectors to binary class matrices
    Y_train = tf.keras.utils.to_categorical(y_train, nb_classes)
    Y_test = tf.keras.utils.to_categorical(y_test, nb_classes)

    img_rows, img_cols, img_channels = X_train.shape[1:4]
    input_shape = (img_rows, img_cols, img_channels)

elif args.dataset == "DR" or args.dataset == "smallDR": # load Diabetic Retinopathy dataset
    nb_classes = 5
    Nimages_by_class = 708  # Remember that the class with least ammount of examples (class 4) has 708
    porcTrain = .8

    if args.dataset == "DR":
        data_folder = os.path.abspath(os.curdir)+'/../Datasets/DR/train_small_200_300'
    elif args.dataset == "smallDR":
        data_folder = os.path.abspath(os.curdir)+'/../Datasets/DR/train_small_100_150'
    labels_dir = os.path.abspath(os.curdir) +'/../Datasets/DR/trainLabels.csv'

    patients = pd.read_csv(labels_dir)

    patients_0 = patients[patients['level']==0]['image'].values
    patients_1 = patients[patients['level']==1]['image'].values
    patients_2 = patients[patients['level']==2]['image'].values
    patients_3 = patients[patients['level']==3]['image'].values
    patients_4 = patients[patients['level']==4]['image'].values

    seed +=1
    np.random.seed(seed)
    np.random.shuffle(patients_0)

    seed +=1
    np.random.seed(seed)
    np.random.shuffle(patients_1)

    seed +=1
    np.random.seed(seed)
    np.random.shuffle(patients_2)

    seed +=1
    np.random.seed(seed)
    np.random.shuffle(patients_3)

    seed +=1
    np.random.seed(seed)
    np.random.shuffle(patients_4)

    patients_0 = patients_0[:Nimages_by_class]
    patients_1 = patients_1[:Nimages_by_class]
    patients_2 = patients_2[:Nimages_by_class]
    patients_3 = patients_3[:Nimages_by_class]
    patients_4 = patients_4[:Nimages_by_class]

    label_0 = np.zeros(patients_0.size, dtype=np.uint8)
    label_1 = np.ones(patients_1.size, dtype=np.uint8)
    label_2 = 2 * np.ones(patients_2.size, dtype=np.uint8)
    label_3 = 3 * np.ones(patients_3.size, dtype=np.uint8)
    label_4 = 4 * np.ones(patients_4.size, dtype=np.uint8)

    ntrain = round(patients_0.size * porcTrain)

    names_train = np.hstack([patients_0[:ntrain], patients_1[:ntrain],
        patients_2[:ntrain], patients_3[:ntrain], patients_4[:ntrain]])

    names_test = np.hstack([patients_0[ntrain:], patients_1[ntrain:],
        patients_2[ntrain:], patients_3[ntrain:], patients_4[ntrain:]])

    label_train = np.hstack([label_0[:ntrain], label_1[:ntrain],
        label_2[:ntrain], label_3[:ntrain], label_4[:ntrain]])

    label_test = np.hstack([label_0[ntrain:], label_1[ntrain:],
        label_2[ntrain:], label_3[ntrain:], label_4[ntrain:]])

    idtrain = np.arange(names_train.size)
    idtest = np.arange(names_test.size)

    seed +=1
    np.random.seed(seed)
    np.random.shuffle(idtrain)

    seed +=1
    np.random.seed(seed)
    np.random.shuffle(idtest)

    names_train = names_train[idtrain]
    label_train = label_train[idtrain]

    names_test = names_test[idtest]
    label_test = label_test[idtest]

    X_train = []
    for i in range(names_train.size):
        fname = os.path.join(data_folder, names_train[i]+'.jpeg-small')
        I = np.array(Image.open(fname))
        I = I / 255
        X_train.append(I)

    X_test = []
    for i in range(names_test.size):
        fname = os.path.join(data_folder, names_test[i]+'.jpeg-small')
        I = np.array(Image.open(fname))
        I = I / 255
        X_test.append(I)

    #validation of loading data
    #labels = np.array(pd.read_csv(labels_dir))
    #for i in range(names_train.shape[0]):
    #    fidx = np.where(labels == names_train[i])[0][0]
    #    trueVal = labels[fidx,1]
    #    if(trueVal!=label_train[i]):
    #        print('Error')
    #for i in range(names_test.shape[0]):
    #    fidx = np.where(labels == names_test[i])[0][0]
    #    trueVal = labels[fidx,1]
    #    if(trueVal!=label_test[i]):
    #        print('Error')
    y_train = label_train
    y_test = label_test

    # Data normalization
    X_train = np.array(X_train).astype('float32')
    X_test = np.array(X_test).astype('float32')

    # Convert class vectors to binary class matrices
    Y_train = np_utils.to_categorical(y_train, nb_classes)
    Y_test = np_utils.to_categorical(y_test, nb_classes)

    img_channels = I.shape[-1]
    img_rows, img_cols = I.shape[:2]
    input_shape = (img_rows, img_cols, img_channels)

# Check if model exists (model, history and confussion matrix)
# if exists and overwrite is false: load model
# else: exec corresponding file with model and train it (params and training should be in file, but saving of model and history should be here)

# Load or train neural network
if os.path.isfile(path_model) and os.path.isfile(path_history) and os.path.isfile(path_conf_mat) and overwrite == False:
    model = load_model(path_model)
    history_fname = open(path_history,"rb")
    history = pickle.load(history_fname)
    confusion_matrix = np.load(path_conf_mat)
else:
    verbose = 2 # 0 = silent, 1 = progress bar, 2 = one line per epoch
    exec(open(path_code_model).read())

    # Calculate and save confusion matrix
    confusion_matrix = np.zeros((nb_classes,nb_classes)).astype(int)
    predLabel = model.predict(X_test)
    maxLabel = np.argmax(predLabel,axis=1)
    maxLabel = maxLabel.reshape((maxLabel.shape[0], 1))
    for i in range(nb_classes):
        idi = y_test==i
        for j in range(nb_classes):
            confusion_matrix[i,j] = (maxLabel[idi]==j).sum()
    
    if save:
        model.save(path_model)
        history_fname = open(path_history,"wb")
        history = history.history
        pickle.dump(history,history_fname)
        history_fname.close()
        np.save(path_conf_mat, confusion_matrix)
        print("Training data saved")

print("Training finished")
if not args.attack:
    sys.exit()

# Attack
dist_adv_ex_mat = np.zeros((nb_classes,nb_classes)).astype(int)
exec(open(path_code_attack).read())

print("Attack finished")
if not args.detection:
    sys.exit()

# Switch detection method
if args.detection == "stoGauNet":
    # Generate two grids for x and y for the colorgraph
    y_plot, x_plot = np.mgrid[slice(y_start - (dy_plot/2), y_end + dy_plot, dy_plot),
                            slice(x_start - (dx_plot/2), x_end + dx_plot, dx_plot)]
    z_rocauc_set_plot=np.zeros((np.size(x_plot,0)-1,np.size(x_plot,1)-1))
    z_rocauc_test_plot=np.zeros((np.size(x_plot,0)-1,np.size(x_plot,1)-1))
    z_min_error_set_plot=np.zeros((np.size(x_plot,0)-1,np.size(x_plot,1)-1))
    z_min_error_test_plot=np.zeros((np.size(x_plot,0)-1,np.size(x_plot,1)-1))

    # Values to show in the graph
    my_xTicks = x_plot[0,:]
    my_xTicks = my_xTicks[0:-1] + dx_plot/2
    my_yTicks = y_plot[:,0]
    my_yTicks = my_yTicks[0:-1] + dy_plot/2

    totalDots = (np.size(x_plot,0)-1) * (np.size(x_plot,1)-1)     # Number of dots in the colorgraph
    iDots = 0
    best_rocauc = 0
    best_min_error = 1

    n_set = int(i_succ * 0.8)
    n_test = i_succ - n_set
    X_test_nat_noisy_pred = np.zeros((i_succ, noise_iter))    # winner class with noise (array because of noise_iter)
    X_test_adv_noisy_pred = np.zeros((i_succ, noise_iter))    # winner class with noise (array because of noise_iter)
    np_eye = np.eye(nb_classes,nb_classes)

    # NN with noise inside a for loop in order to make a color graph
    for i_x_plot in range (0,np.size(x_plot,0)-1):
        for i_y_plot in range (0,np.size(x_plot,1)-1):
            iDots = iDots + 1
            sigmanoiseD = x_plot[i_x_plot,i_y_plot] + dx_plot/2     # dense noise
            sigmanoiseC = y_plot[i_x_plot,i_y_plot] + dy_plot/2     # convolutional noise
            
            print('\n{:.0f} of {:.0f} to finish colorgraph ; snC = {:.2f} ; snD = {:.2f}'.format(iDots,totalDots,sigmanoiseC,sigmanoiseD))
            
            if sigmanoiseD == 0 and sigmanoiseC == 0:
                z_rocauc_set_plot[i_x_plot,i_y_plot] = 0.5
                z_min_error_set_plot[i_x_plot,i_y_plot] = 0.5
                z_rocauc_test_plot[i_x_plot,i_y_plot] = 0.5
                z_min_error_test_plot[i_x_plot,i_y_plot] = 0.5
                continue
            
            # The next line executes a code to get model 1
            exec(open(path_code_model).read())
            
            # In the next lines we cause the model 1 to have the same weights as the original model
            weights = model.get_weights()
            model1.set_weights(weights)

            # Detection method stoGauNet
            frs_nat = np.zeros(shape=(i_succ))
            frs_adv= np.zeros(shape=(i_succ))

            # When this for loop finishes, the arrays X_test_nat_noisy_pred and X_test_adv_noisy_pred are full
            np.random.seed(1) 
            
            for col in range(noise_iter):
                # Data needed to set and test threshold
                input_tensor = model1.input
                output_tensor = model1.get_layer("last_layer").output # "dense" because is the name of the last layer (before the softmax)
                model_aux = keras.Model(input_tensor, output_tensor)
                # predict without printing the progress bar
                get_layer_output = model_aux.predict
                layer_outputs = get_layer_output(X_test_nat, verbose = 0)
                layer_outputs = np.argsort(layer_outputs, axis = 1)[:,nb_classes-1]
                X_test_nat_noisy_pred[:,col] = layer_outputs
                layer_outputs = get_layer_output(X_test_adv, verbose = 0)
                layer_outputs = np.argsort(layer_outputs, axis = 1)[:,nb_classes-1]
                X_test_adv_noisy_pred[:,col] = layer_outputs
            
            cond = X_test_nat_pred[:,np.newaxis]
            frs_nat = np.sum(X_test_nat_noisy_pred==cond, axis = 1) / noise_iter
            cond = X_test_adv_pred[:,np.newaxis]
            frs_adv = np.sum(X_test_adv_noisy_pred==cond, axis = 1) / noise_iter
            frs_nat_set = frs_nat[:n_set]   # nat frs to set threshold
            frs_adv_set = frs_adv[:n_set]   # adv frs to set threshold
            frs_nat_test = frs_nat[n_set:]  # nat frs to test threshold
            frs_adv_test = frs_adv[n_set:]  # adv frs to test threshold

            # Calculate rocauc and min_error with the portion of frs array that is to set the threshold (similar to the train portion of the dataset when training a network)
            rocauc_set = functions.roc_auc(frs_nat_set,frs_adv_set)
            (min_error_set,best_threshold) = functions.min_error(frs_nat_set,frs_adv_set)

            # Same as before, but now with the portion of frs array that is to test the threshold (similar to the test portion of the dataset when training a network)
            # Note: No calculation are made with the test part. Its only to see if the threshold behaves good with other examples
            rocauc_test = functions.roc_auc(frs_nat_test,frs_adv_test)
            succ_nat = np.sum(frs_nat_test>best_threshold)
            succ_adv = np.sum(frs_adv_test<best_threshold)
            min_error_test = 1 - (succ_nat + succ_adv) / 2 / n_test
            
            print("rocauc(set):  {:.4f}".format(rocauc_set))
            print("rocauc(test): {:.4f}".format(rocauc_test))
            print("min_error(set):  {:.4f}".format(min_error_set))
            print("min_error(test): {:.4f}".format(min_error_test))

            if(rocauc_set>best_rocauc):
                frs_nat_set_best_rocauc = np.copy(frs_nat_set)
                frs_adv_set_best_rocauc = np.copy(frs_adv_set)
                frs_nat_test_best_rocauc = np.copy(frs_nat_test)
                frs_adv_test_best_rocauc = np.copy(frs_adv_test)
                best_rocauc = rocauc_set

            if(min_error_set<best_min_error):
                frs_nat_set_best_min_error = np.copy(frs_nat_set)
                frs_adv_set_best_min_error = np.copy(frs_adv_set)
                frs_nat_test_best_min_error = np.copy(frs_nat_test)
                frs_adv_test_best_min_error = np.copy(frs_adv_test)
                best_min_error = min_error_set
            
            z_rocauc_set_plot[i_x_plot,i_y_plot] = rocauc_set
            z_min_error_set_plot[i_x_plot,i_y_plot] = min_error_set
            z_rocauc_test_plot[i_x_plot,i_y_plot] = rocauc_test
            z_min_error_test_plot[i_x_plot,i_y_plot] = min_error_test

    # Save detection data to make colorgraphs and to plot the best histogram (the histogram corresponding with the best score)
    if (not (os.path.isfile(path_detection) and overwrite == False)) and save:
        np.savez(path_detection, x_plot = x_plot, y_plot = y_plot, z_rocauc_set_plot = z_rocauc_set_plot, z_rocauc_test_plot = z_rocauc_test_plot, z_min_error_set_plot = z_min_error_set_plot, z_min_error_test_plot = z_min_error_test_plot, frs_nat_set_best_rocauc = frs_nat_set_best_rocauc, frs_adv_set_best_rocauc = frs_adv_set_best_rocauc, frs_nat_test_best_rocauc = frs_nat_test_best_rocauc, frs_adv_test_best_rocauc = frs_adv_test_best_rocauc, frs_nat_set_best_min_error = frs_nat_set_best_min_error, frs_adv_set_best_min_error = frs_adv_set_best_min_error, frs_nat_test_best_min_error = frs_nat_test_best_min_error, frs_adv_test_best_min_error = frs_adv_test_best_min_error)
        print("Detection data saved")

print("Detection finished")