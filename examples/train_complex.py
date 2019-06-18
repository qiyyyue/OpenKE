import os

import config
import models
import tensorflow as tf
import numpy as np

con = config.Config()
#Input training files from benchmarks/FB15K/ folder.
con.set_in_path("../benchmarks/kg_100k/")
#True: Input test files from the same folder.
con.set_test_link_prediction(True)
con.set_test_triple_classification(True)

con.set_work_threads(4)
con.set_train_times(500)
con.set_nbatches(100)
con.set_alpha(0.1)
con.set_lmbda(0.0001)
con.set_bern(0)
con.set_dimension(100)
con.set_ent_neg_rate(1)
con.set_rel_neg_rate(0)
con.set_opt_method("Adagrad")


save_path = '../res/kg_100k/complEx'
if not os.path.exists(save_path):
    os.makedirs(save_path)

#Models will be exported via tf.Saver() automatically.
con.set_export_files(os.path.join(save_path, "model.vec.tf"), 0)
#Model parameters will be exported to json files automatically.
con.set_out_files(os.path.join(save_path, "embedding.vec.json"))
#Initialize experimental settings.
con.init()
#Set the knowledge embedding model
con.set_model(models.ComplEx)
#Train the model.
con.run()
#To test models after training needs "set_test_flag(True)".
con.test()

