mkdir CNN_params
wget -P CNN_params https://github.com/mikekd106/ML2017FALL/releases/download/0.0.0/CNN_model.h5

python3 hw3_cnn.py --infer --test_data_path=$1 --output_path=$2