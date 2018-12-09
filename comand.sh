#command for prepairing dadaset for trainingg the network
python dataset_tool_tf.py --input-dir datasets/ILSVRC2012_img_val --out=datasets/imagenet_val_raw.tfrecords

#downloading kodak dataset
python download_kodak.py --output-dir=datasets/kodak


#training the netwotk
python config.py --desc='-test' train --train-tfrecords=datasets/imagenet_val_raw.tfrecords --long-train=true --noise=gaussian


#testing or valiation of the network
python config.py validate --dataset-dir=datasets/kodak --network-snapshot=results/00001-autoencoder-1gpu-L-n2n/network_final.pickle
