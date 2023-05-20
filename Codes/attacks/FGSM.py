# target = K.placeholder(shape=(nb_classes,))
# output = model.layers[-1].output
# loss = K.categorical_crossentropy(target,output,from_logits=True,axis=-1) # Caution: this must be the same function used in the training phase
# batch_jacobian_loss = tf_batch_jacobian(loss, model.layers[0].input, use_pfor=False)

from PIL import Image

loss_object = tf.keras.losses.CategoricalCrossentropy(from_logits=True) # apply from logits = True

def create_adversarial_pattern(input_image, input_label):
  with tf.GradientTape() as tape:
    tape.watch(input_image)
    prediction = model(input_image)
    loss = loss_object(input_label, prediction)


  # Get the gradients of the loss w.r.t to the input image.
  gradient = tape.gradient(loss, input_image)
  # Get the sign of the gradients to create the perturbation
  signed_grad = tf.sign(gradient)
  return signed_grad

def preprocess(image):
  image = tf.cast(image, tf.float32)
#   image = tf.image.resize(image, (28, 28))
#   image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
  image /= 255.0  # normalize to [0,1] range  
  image = image[None, ...]
  return image

idx = 0                     # Index of images
i_succ = 0                  # Successful generated adv examples
n_succ = args.n_succ        # Number of adversarial examples to generate
n_classified = 0            # Number of correct classified images, that means that i_succ/n_classified will be the prob of generating an adv example
n_test = X_test.shape[0]    # Number of images in test set
epsilon = args.epsilon      # Perturbation size FGSM

l2=np.zeros(shape=(n_succ))

if not os.path.isfile(path_attack): # Generate and save adversarial examples corresponding to X_test
    # Index idx of test set
    while i_succ != n_succ:
        # print("")
        if idx == n_test:
            break
        img0 = np.squeeze(X_test[idx])
        img = img0.reshape((1, img_rows, img_cols, img_channels))

        iout = np.argmax(model(img).numpy()) # Winner class
        iobj = np.argsort(model(img).numpy()[0])[nb_classes-2]
        
        if np.argmax(Y_test[idx]) != iout: # Wrong classification
            print("NN wrong classification")
            idx += 1
            continue

        n_classified += 1
        iter=0
        
        label = tf.one_hot(iout, nb_classes)
        label = tf.reshape(label, (1, nb_classes))
        img0 = (img0 * 255).astype(np.uint8)
        Image.fromarray(img0).save('temp.jpg')

        # leer la imagen de nuevo con tf.image.decode_jpeg
        img = tf.io.read_file('temp.jpg')
        # image = tf.image.decode_jpeg(img, channels=img_channels) 

        # borrar el archivo temporal
        os.remove('temp.jpg')
        image = tf.image.decode_image(img)
        image = preprocess(image)
        perturbations = create_adversarial_pattern(image, label)

        # Calculates the gradient of the output with respect to the input (idx element of the test set)
        # grad = sess.run(batch_jacobian_loss, feed_dict={target: Y_test[idx] , model.input: X_test[idx].reshape(1,img_rows, img_cols,img_channels) })
        # sgrad = np.sign(grad)*epsilon
        # Input vector correction
        X_adver = image + perturbations * epsilon
        # X_adver /= 255.0  # normalize to [0,1] range
        X_adver = np.clip(X_adver, 0, 1) # Clip the image to be between 0 and 1

        # Outputs with the new input
        layer_output = model(X_adver)
        iadv = np.argsort(layer_output[0])[nb_classes-1] # Winner class
                
        # If the adversarial example was successful
        if iobj == iadv:
            print("Adv example ({} of {}) successfully generated".format(i_succ+1, n_succ))
            dist_adv_ex_mat[iout,iadv] += 1

            # l2 distortion (needed to calculate rho)
            aa=np.sum((X_test[idx:idx+1]-X_adver)**2)
            bb=np.sum(X_test[idx:idx+1]**2)
            l2[i_succ] = np.sqrt(aa/bb)
            i_succ += 1
            
            np_image = np.array(image)
            np_adver = np.array(X_adver)

            if(i_succ==1):
                X_test_nat = np_image
                X_test_adv = np_adver
                X_test_nat_adv_pred = np.array([[iout],[iadv]])
            else:
                X_test_nat = np.concatenate((X_test_nat,np_image))
                X_test_adv = np.concatenate((X_test_adv,np_adver))
                X_test_nat_adv_pred = np.hstack((X_test_nat_adv_pred,np.array([[iout],[iadv]])))
        else:
            pass
            # print("Attack failed in generating an adv example for this image")
        idx += 1

    l2 = l2[0:i_succ]
    rho = np.sum(l2)/i_succ
    X_test_nat_adv = np.stack((X_test_nat,X_test_adv))
    print("rho: ",rho)
else:
    data = np.load(path_attack)
    # Load adversarial examples corresponding to X_test
    X_test_nat_adv = data['X_test_nat_adv']
    X_test_nat = X_test_nat_adv[0]
    X_test_adv = X_test_nat_adv[1]
    X_test_nat_adv_pred = data['X_test_nat_adv_pred']
    n_classified = data['n_classified']
    i_succ = data['i_succ']

X_test_nat_pred = X_test_nat_adv_pred[0]
X_test_adv_pred = X_test_nat_adv_pred[1]

p_succ = i_succ / n_classified

# Save attack data
if (not (os.path.isfile(path_attack) and overwrite == False)) and save:
    np.savez(path_attack, dist_adv_ex_mat = dist_adv_ex_mat, rho = rho, p_succ = p_succ, i_succ = i_succ, n_classified = n_classified, epsilon = epsilon, X_test_nat_adv = X_test_nat_adv, X_test_nat_adv_pred = X_test_nat_adv_pred)
    print("Attack data saved")