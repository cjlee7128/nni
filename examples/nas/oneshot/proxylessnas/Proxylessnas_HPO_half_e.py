### Changjae Lee @ 2022-09-17 
import logging
import os
import sys
from argparse import ArgumentParser
# https://stackoverflow.com/questions/67320700/how-to-run-multiple-scripts-in-a-python-script-with-args 
import subprocess 

logger = logging.getLogger('nni_proxylessnas_hpo')

if __name__ == "__main__":
    parser = ArgumentParser("nni_proxylessnas_hpo")
    parser.add_argument("--CUDA_VISIBLE_DEVICES", default=0, type=int)
    parser.add_argument("--reference_latency", default=None, type=float, help='the reference latency in specified hardware')
    parser.add_argument("--n_worker", default=0, type=int) 
    parser.add_argument("--train_trainer_epochs", default=150, type=int) 
    parser.add_argument("--retrain_trainer_epochs", default=30, type=int) 
    ### Changjae Lee @ 2022-09-17 
    # default='24,40,80,96,192,320' -> default='8,8,16,16,24,40' 
    #parser.add_argument("--width_stages_list", default='8,16,24,32,40,48,56,64', type=str)
    ### Changjae Lee @ 2022-09-21 
    #parser.add_argument("--output_size", default=16, type=int)

    args = parser.parse_args()
    CUDA_VISIBLE_DEVICES=args.CUDA_VISIBLE_DEVICES 
    reference_latency=args.reference_latency 
    n_worker=args.n_worker 
    train_trainer_epochs=args.train_trainer_epochs 
    retrain_trainer_epochs=args.retrain_trainer_epochs 
    #width_stages_list=[int(i) for i in args.width_stages_list.split(',')]
    
    ### Changjae Lee @ 2022-09-18 
    i = 0

    # 0 1 2 3 4 5 
    # 2 2 3 3 3 3 
    for w1 in [2, 4, 8, 16, 32]: 
        width_stages = f'{w1}' 
        for c1 in [1, 2, 3, 4]: 
            n_cell_stages = f'{c1}' 
            for s1 in [1, 2]: 
                stride_stages=f'{s1}'
                for output_size in [2, 4, 8, 12, 16, 24, 32, 40, 48]: 
                    # if i < 3: 
                    #     i = i + 1 
                    #     continue 
                    # This is like running `python script1 -c config/config_file1.json -m import` and your other commands
                    # https://stackoverflow.com/questions/70235696/checking-folder-and-if-it-doesnt-exist-create-it 
                    PATH = f'./checkpoint_width_stages_{width_stages}_{n_cell_stages}_{stride_stages}_{output_size}/' 
                    if not os.path.exists(PATH):
                        os.makedirs(PATH)
                    # search  
                    # https://stackoverflow.com/questions/8365394/set-environment-variable-in-python-script 
                    #os.environ['CUDA_VISIBLE_DEVICES'] = f"{CUDA_VISIBLE_DEVICES}" # visible in this process + all children 
                    cmd_str = ["python", "main_half_e.py", "--applied_hardware", "cortexA76cpu_tflite21", "--reference_latency", f"{reference_latency}", "--n_worker", f"{n_worker}", "--width_stages", f"{width_stages}", "--n_cell_stages", f'{n_cell_stages}', "--stride_stages", f'{stride_stages}', "--output_size", f'{output_size}', "--checkpoint_path", f'{PATH}', "--train_trainer_epochs", f"{train_trainer_epochs}"] 
                    subprocess.call(cmd_str, shell=False)
                    # retrain 
                    cmd_str = ["python", "main_half_e.py", "--applied_hardware", "cortexA76cpu_tflite21", "--reference_latency", f"{reference_latency}", "--n_worker", f"{n_worker}", "--train_mode", "retrain", "--width_stages", f"{width_stages}", "--n_cell_stages", f'{n_cell_stages}', "--stride_stages", f'{stride_stages}', "--output_size", f'{output_size}', "--checkpoint_path", f"{PATH}", "--exported_arch_path", f"{PATH}checkpoint.json", "--retrain_trainer_epochs", f"{retrain_trainer_epochs}"] 
                    subprocess.call(cmd_str, shell=False) 