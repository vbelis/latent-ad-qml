svm = SVC(kernel='rbf', gamma=args["gamma"], C=args["c_param"])
qsvm.fit(train_features, train_labels)
#train_acc = qsvm.score(train_features, train_labels)