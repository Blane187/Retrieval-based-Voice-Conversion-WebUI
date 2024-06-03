import os
import numpy as np
import faiss
import json
import pathlib
from random import shuffle
from subprocess import Popen, PIPE, STDOUT
import argparse

def train_index(exp_dir1, version19):
    exp_dir = f"logs/{exp_dir1}"
    os.makedirs(exp_dir, exist_ok=True)
    feature_dir = f"{exp_dir}/3_feature256" if version19 == "v1" else f"{exp_dir}/3_feature768"
    
    if not os.path.exists(feature_dir):
        return "请先进行特征提取!"
    
    listdir_res = list(os.listdir(feature_dir))
    if len(listdir_res) == 0:
        return "请先进行特征提取！"
    
    infos = []
    npys = []
    for name in sorted(listdir_res):
        phone = np.load(f"{feature_dir}/{name}")
        npys.append(phone)
    big_npy = np.concatenate(npys, 0)
    big_npy_idx = np.arange(big_npy.shape[0])
    np.random.shuffle(big_npy_idx)
    big_npy = big_npy[big_npy_idx]
    
    if big_npy.shape[0] > 2e5:
        infos.append(f"Trying doing kmeans {big_npy.shape[0]} shape to 10k centers.")
        yield "\n".join(infos)
        try:
            big_npy = MiniBatchKMeans(
                n_clusters=10000,
                verbose=True,
                batch_size=256 * os.cpu_count(),
                compute_labels=False,
                init="random",
            ).fit(big_npy).cluster_centers_
        except Exception as e:
            infos.append(str(e))
            yield "\n".join(infos)
    
    np.save(f"{exp_dir}/total_fea.npy", big_npy)
    n_ivf = min(int(16 * np.sqrt(big_npy.shape[0])), big_npy.shape[0] // 39)
    infos.append(f"{big_npy.shape},{n_ivf}")
    yield "\n".join(infos)
    
    index = faiss.index_factory(256 if version19 == "v1" else 768, f"IVF{n_ivf},Flat")
    infos.append("training")
    yield "\n".join(infos)
    
    index_ivf = faiss.extract_index_ivf(index)
    index_ivf.nprobe = 1
    index.train(big_npy)
    faiss.write_index(index, f"{exp_dir}/trained_IVF{n_ivf}_Flat_nprobe_{index_ivf.nprobe}_{exp_dir1}_{version19}.index")
    
    infos.append("adding")
    yield "\n".join(infos)
    batch_size_add = 8192
    for i in range(0, big_npy.shape[0], batch_size_add):
        index.add(big_npy[i: i + batch_size_add])
    faiss.write_index(index, f"{exp_dir}/added_IVF{n_ivf}_Flat_nprobe_{index_ivf.nprobe}_{exp_dir1}_{version19}.index")
    infos.append(f"成功构建索引，added_IVF{n_ivf}_Flat_nprobe_{index_ivf.nprobe}_{exp_dir1}_{version19}.index")
    yield "\n".join(infos)

def click_train(
    exp_dir1,
    sr2,
    if_f0_3,
    spk_id5,
    save_epoch10,
    total_epoch11,
    batch_size12,
    if_save_latest13,
    pretrained_G14,
    pretrained_D15,
    gpus16,
    if_cache_gpu17,
    if_save_every_weights18,
    version19,
):
    exp_dir = f"{now_dir}/logs/{exp_dir1}"
    os.makedirs(exp_dir, exist_ok=True)
    gt_wavs_dir = f"{exp_dir}/0_gt_wavs"
    feature_dir = f"{exp_dir}/3_feature256" if version19 == "v1" else f"{exp_dir}/3_feature768"
    
    if if_f0_3:
        f0_dir = f"{exp_dir}/2a_f0"
        f0nsf_dir = f"{exp_dir}/2b-f0nsf"
        names = (
            set([name.split(".")[0] for name in os.listdir(gt_wavs_dir)])
            & set([name.split(".")[0] for name in os.listdir(feature_dir)])
            & set([name.split(".")[0] for name in os.listdir(f0_dir)])
            & set([name.split(".")[0] for name in os.listdir(f0nsf_dir)])
        )
    else:
        names = set([name.split(".")[0] for name in os.listdir(gt_wavs_dir)]) & set(
            [name.split(".")[0] for name in os.listdir(feature_dir)]
        )
    
    opt = []
    for name in names:
        if if_f0_3:
            opt.append(
                f"{gt_wavs_dir}/{name}.wav|{feature_dir}/{name}.npy|{f0_dir}/{name}.wav.npy|{f0nsf_dir}/{name}.wav.npy}|{spk_id5}"
            )
        else:
            opt.append(
                f"{gt_wavs_dir}/{name}.wav|{feature_dir}/{name}.npy}|{spk_id5}"
            )
    
    fea_dim = 256 if version19 == "v1" else 768
    if if_f0_3:
        for _ in range(2):
            opt.append(
                f"{now_dir}/logs/mute/0_gt_wavs/mute{sr2}.wav|{now_dir}/logs/mute/3_feature{fea_dim}/mute.npy|{now_dir}/logs/mute/2a_f0/mute.wav.npy|{now_dir}/logs/mute/2b-f0nsf/mute.wav.npy}|{spk_id5}"
            )
    else:
        for _ in range(2):
            opt.append(
                f"{now_dir}/logs/mute/0_gt_wavs/mute{sr2}.wav|{now_dir}/logs/mute/3_feature{fea_dim}/mute.npy}|{spk_id5}"
            )
    
    shuffle(opt)
    with open(f"{exp_dir}/filelist.txt", "w") as f:
        f.write("\n".join(opt))
    
    print("Write filelist done")
    print("Use gpus:", str(gpus16))
    if pretrained_G14 == "":
        print("No pretrained Generator")
    if pretrained_D15 == "":
        print("No pretrained Discriminator")
    
    if version19 == "v1" or sr2 == "40k":
        config_path = f"configs/v1/{sr2}.json"
    else:
        config_path = f"configs/v2/{sr2}.json"
    
    config_save_path = os.path.join(exp_dir, "config.json")
    if not pathlib.Path(config_save_path).exists():
        with open(config_save_path, "w", encoding="utf-8") as f:
            with open(config_path, "r") as config_file:
                config_data = json.load(config_file)
                json.dump(
                    config_data,
                    f,
                    ensure_ascii=False,
                    indent=4,
                    sort_keys=True,
                )
            f.write("\n")
    
    cmd = (
        f'python infer/modules/train/train.py -e "{exp_dir1}" -sr {sr2} -f0 {1 if if_f0_3 else 0} -bs {batch_size12} -g {gpus16} -te {total_epoch11} -se {save_epoch10} '
        f'{"-pg %s" % pretrained_G14 if pretrained_G14 != "" else ""} {"-pd %s" % pretrained_D15 if pretrained_D15 != "" else ""} '
        f'-l {1 if if_save_latest13 else 0} -c {1 if if_cache_gpu17 else 0} -sw {1 if if_save_every_weights18 else 0} -v {version19}'
    )
    
    p = Popen(cmd, shell=True, cwd=now_dir, stdout=PIPE, stderr=STDOUT, bufsize=1, universal_newlines=True)
    
    for line in p.stdout:
        print(line.strip())
    
    p.wait()
    return "训练结束, 您可查看控制台训练日志或实验文件夹下的train.log"

def main():
    parser = argparse.ArgumentParser(description="CLI for voice conversion training")
    parser.add_argument('--model_name', type=str, required=True, help='Model name')
    parser.add_argument('--dataset', type=str, required=True, help='Dataset path')
    parser.add_argument('--sample_rate', type=str, choices=['48000', '40000'], default='48000', help='Sample rate')
    parser.add_argument('--version', type=str, choices=['v1', 'v2'], default='v2', help='Version')
    parser.add_argument('--f0method', type=str, default='rmvpe_gpu', help='F0 method')
    parser.add_argument('--save_frequency', type=int, default=50, help='Save frequency')
    parser.add_argument('--epoch', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch_size', type=str, default='7', help='Batch size')
    parser.add_argument('--cache_gpu', action='store_true', help='Cache GPU')
    
    args = parser.parse_args()
    
    model_name = args.model_name
    exp_dir = model_name
    dataset = args.dataset
    sample_rate = args.sample_rate
    ksample_rate = "48k" if sample_rate == "48000" else "40k"
    version = args.version
    version19 = version
    f0method = args.f0method
    save_frequency = args.save_frequency
    epoch = args.epoch
    batch_size = args.batch_size
    cache_gpu = args.cache_gpu
    
    now_dir = os.path.join('Retrieval-based-Voice-Conversion-WebUI')
    
    os.makedirs(f"{now_dir}/logs/{exp_dir}", exist_ok=True)
    open(f"{now_dir}/logs/{exp_dir}/preprocess.log", "w").close()
    open(f"{now_dir}/logs/{exp_dir}/extract_f0_feature.log", "w").close()
    
    # Process Data
    command = f"python infer/modules/train/preprocess.py '{dataset}' {sample_rate} 2 '{now_dir}/logs/{exp_dir}' False 3.0"
    print(command)
    os.system(command)
    
    # Feature Extraction
    if f0method != "rmvpe_gpu":
        command = f"python infer/modules/train/extract/extract_f0_print.py '{now_dir}/logs/{exp_dir}' 2 '{f0method}'"
    else:
        command = f"python infer/modules/train/extract/extract_f0_rmvpe.py 1 0 0 '{now_dir}/logs/{exp_dir}' True"
    print(command)
    os.system(command)
    
    command = f"python infer/modules/train/extract_feature_print.py cuda:0 1 0 0 '{now_dir}/logs/{exp_dir}' '{version}' False"
    print(command)
    os.system(command)
    
    # Train Feature Index
    result_generator = train_index(exp_dir, version)
    for result in result_generator:
        print(result)
    
    # Train Model
    if version == 'v1':
        if ksample_rate == '40k':
            G_path = 'assets/pretrained/f0G40k.pth'
            D_path = 'assets/pretrained/f0D40k.pth'
        elif ksample_rate == '48k':
            G_path = 'assets/pretrained/f0G48k.pth'
            D_path = 'assets/pretrained/f0D48k.pth'
    elif version == 'v2':
        if ksample_rate == '40k':
            G_path = 'assets/pretrained_v2/f0G40k.pth'
            D_path = 'assets/pretrained_v2/f0D40k.pth'
        elif ksample_rate == '48k':
            G_path = 'assets/pretrained_v2/f0G48k.pth'
            D_path = 'assets/pretrained_v2/f0D48k.pth'
    
    result_generator = click_train(
        exp_dir,
        ksample_rate,
        True,
        0,
        save_frequency,
        epoch,
        batch_size,
        True,
        G_path,
        D_path,
        0,
        cache_gpu,
        False,
        version,
    )
    print(result_generator)

if __name__ == "__main__":
    main()
