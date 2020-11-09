python main.py --model GfEASR --data_train DIV2K  --scale 2  --data_range 1-800/801-900 --n_resblocks 16 --n_feats 64  --data_test DIV2K --save gfeasr_r16f64_x2 --lr 0.00005 --decay 20-100-180 --patch_size 96 --epochs 230 --pre_train ../experiment/gfear_r16f64x2/model/model_latest.pt 
python main.py --model GfEASR --data_train DIV2K  --scale 3  --data_range 1-800/801-900 --n_resblocks 16 --n_feats 64  --data_test DIV2K --save gfeasr_r16f64_x3 --lr 0.0001 --decay 30-80-160-240 --patch_size 144
python main.py --model GfEASR --data_train DIV2K  --scale 4  --data_range 1-800/801-900 --n_resblocks 16 --n_feats 64  --data_test DIV2K --save gfeasr_r16f64_x4 --lr 0.0001 --decay 30-80-160-240 --patch_size 192

