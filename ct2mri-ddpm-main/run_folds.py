"""Train (and optionally sample/score) several CV folds in parallel, one GPU per job.
Replaces example_exp.py. Folds wait in a queue, so 5 folds on 4 GPUs works: the fifth
fold starts as soon as a GPU frees up. Unknown arguments are forwarded to train.py.
 
    python run_folds.py --data_path $DATA --exp_name ddpm_avg_eq --folds 1 2 3 4 5 \
        --gpus 0 1 2 3 --sample --test_file /path/avg_eq_seg_test.npz --epochs 500 --amp
 
train.py always runs with --resume, so re-running the same command after a preemption
continues unfinished folds and skips the training of finished ones.
"""
import argparse
import os
import queue
import subprocess
import sys
import threading
 
 
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_path", required=True, help="fold file prefix")
    ap.add_argument("--exp_name", required=True)
    ap.add_argument("--folds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    ap.add_argument("--gpus", type=int, nargs="+", default=[0])
    ap.add_argument("--out_dir", default="runs")
    ap.add_argument("--sample", action="store_true", help="sample + score the held-out fold after training")
    ap.add_argument("--test_file", default=None, help="separate test set .npz to sample + score as well")
    ap.add_argument("--sampler", default="ddpm", choices=["ddpm", "ddim"])
    a, train_extra = ap.parse_known_args()
 
    py = sys.executable
    jobs = queue.Queue()
    for f in a.folds:
        jobs.put(f)
 
    def worker(gpu):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
        while True:
            try:
                fold = jobs.get_nowait()
            except queue.Empty:
                return
            run_dir = os.path.join(a.out_dir, a.exp_name, f"fold{fold}")
            cmds = [[py, "train.py", "--data_path", a.data_path, "--fold", str(fold),
                     "--exp_name", a.exp_name, "--out_dir", a.out_dir, "--resume", *train_extra]]
            if a.sample:
                cmds.append([py, "sample.py", "--run_dir", run_dir, "--data_path", a.data_path,
                             "--fold", str(fold), "--sampler", a.sampler])
            if a.test_file:
                cmds.append([py, "sample.py", "--run_dir", run_dir, "--data_file", a.test_file,
                             "--sampler", a.sampler])
            for c in cmds:
                print(f"[gpu {gpu}] fold {fold}: {' '.join(c)}", flush=True)
                if subprocess.run(c, env=env).returncode != 0:
                    print(f"[gpu {gpu}] fold {fold} FAILED at {c[1]}", flush=True)
                    break
 
    threads = [threading.Thread(target=worker, args=(g,)) for g in a.gpus]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
 
 
if __name__ == "__main__":
    main()
 