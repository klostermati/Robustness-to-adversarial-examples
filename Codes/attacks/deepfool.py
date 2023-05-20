# Create a new tensor as input
list_shape = np.array(model.layers[0].input.shape[1:]).tolist()
list_shape.insert(0,1)
input_tensor = tf.constant(np.random.random(list_shape), dtype=tf.float32)

# Create a GradientTape to record the operations
with tf.GradientTape() as tape:
    tape.watch(input_tensor)
    output = model(input_tensor)
    jacobian = tape.jacobian(output, input_tensor)

idx = 0                     # Index of images
i_succ = 0                  # Successful generated adv examples
n_succ = args.n_succ        # Number of adversarial examples to generate
n_classified = 0            # Number of correct classified images, that means that i_succ/n_classified will be the prob of generating an adv example
n_test = X_test.shape[0]    # Number of images in test set
max_iter = args.max_iter    # Maximum number of times the input is updated
eta = args.eta              # Value of the overshoot variable

l2=np.zeros(shape=(n_succ))

if not os.path.isfile(path_attack): # Generate and save adversarial examples corresponding to X_test
    # Index idx of test set
    while i_succ != n_succ:
        print("")
        if idx == n_test:
            break

        img0 = np.squeeze(X_test[idx])
        img = img0.reshape((1, img_rows, img_cols, img_channels))

        iout = np.argmax(model(img).numpy()) # Winner class
        iobj = np.argsort(model(img).numpy()[0])[nb_classes-2] # Runner up class
        
        if np.argmax(Y_test[idx]) != iout: # Wrong classification
            print("NN wrong classification")
            idx += 1
            continue
        
        n_classified += 1
        iter = 0
        iadv = -1     # At least one execution of while loop
        X_adver = np.copy(img)
        X_adver_tf = tf.convert_to_tensor(X_adver)
        
        while iadv != iobj and iter<max_iter:
            print("trying")
            # intermediate_model = tf.keras.models.Model(inputs=model.input,
            #                                 outputs=model.layers[-2].output)
            # layer_output = intermediate_model(X_adver_tf)
            
            with tf.GradientTape() as tape:
                tape.watch(X_adver_tf)
                intermediate_model = tf.keras.models.Model(inputs=model.input, outputs=model.layers[-2].output)
                layer_output = intermediate_model(X_adver_tf)
                grad_iout = tape.gradient(layer_output[0, iout], X_adver_tf)
            
            with tf.GradientTape() as tape:
                tape.watch(X_adver_tf)
                intermediate_model = tf.keras.models.Model(inputs=model.input, outputs=model.layers[-2].output)
                layer_output = intermediate_model(X_adver_tf)
                grad_iobj = tape.gradient(layer_output[0, iobj], X_adver_tf)

            dgrad = grad_iout.numpy() - grad_iobj.numpy()
            
            norm2 = np.dot(dgrad.flatten(), dgrad.flatten())
            delta_output = layer_output[0][iout]-layer_output[0][iobj]
            # Input vector correction
            X_adver -= (1+eta)*delta_output*dgrad/norm2
            X_adver_tf = tf.convert_to_tensor(X_adver)
            # Outputs with the new input
            layer_output = model(X_adver_tf)
            iadv = np.argmax(layer_output[0]) # Winner class
            iter += 1
                
        # If the adversarial example was successful
        if iobj == iadv:
            print("Adv example ({} of {}) successfully generated".format(i_succ+1, n_succ))
            dist_adv_ex_mat[iout,iadv] += 1

            # l2 distortion (needed to calculate rho)
            aa=np.sum((X_test[idx:idx+1]-X_adver)**2)
            bb=np.sum(X_test[idx:idx+1]**2)
            l2[i_succ] = np.sqrt(aa/bb)
            i_succ += 1
            
            if(i_succ==1):
                X_test_nat = img
                X_test_adv = X_adver
                X_test_nat_adv_pred = np.array([[iout],[iadv]])
            else:
                X_test_nat = np.concatenate((X_test_nat,img))
                X_test_adv = np.concatenate((X_test_adv,X_adver))
                X_test_nat_adv_pred = np.hstack((X_test_nat_adv_pred,np.array([[iout],[iadv]])))
        else:
            print("Attack failed in generating an adv example for this image")
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
    np.savez(path_attack, dist_adv_ex_mat = dist_adv_ex_mat, rho = rho, p_succ = p_succ, i_succ = i_succ, n_classified = n_classified, max_iter = max_iter, eta = eta, X_test_nat_adv = X_test_nat_adv, X_test_nat_adv_pred = X_test_nat_adv_pred)
    print("Attack data saved")