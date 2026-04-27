import argparse
import time
import datetime
from torch_ac.torch_ac import algos
import tensorboardX
import sys
import os
import utils

from model import ACModel, RNDACModel
import numpy as np
import os


import torch
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]= "0"
parser = argparse.ArgumentParser()


parser.add_argument("--algo", required=True,
                    help="algorithm to use: a2c | ppo (REQUIRED)")
parser.add_argument("--env", required=True,
                    help="name of the environment to train on (REQUIRED)")
parser.add_argument("--model", default=None,
                    help="name of the model (default: {ENV}_{ALGO}_{TIME})")
parser.add_argument("--seed", type=int, default=1,
                    help="random seed (default: 1)")
parser.add_argument("--log-interval", type=int, default=1,
                    help="number of updates between two logs (default: 1)")
parser.add_argument("--save-interval", type=int, default=10,
                    help="number of updates between two saves (default: 10, 0 means no saving)")
parser.add_argument("--procs", type=int, default=16,
                    help="number of processes (default: 16)")
parser.add_argument("--frames", type=int, default=10**7,
                    help="number of frames of training (default: 1e7)")


parser.add_argument("--epochs", type=int, default=4,
                    help="number of epochs for PPO (default: 4)")
parser.add_argument("--batch-size", type=int, default=256,
                    help="batch size for PPO (default: 256)")
parser.add_argument("--frames-per-proc", type=int, default=None,
                    help="number of frames per process before update (default: 5 for A2C and 128 for PPO)")
parser.add_argument("--discount", type=float, default=0.99,
                    help="discount factor (default: 0.99)")
parser.add_argument("--lr", type=float, default=0.001,
                    help="learning rate (default: 0.001)")
parser.add_argument("--gae-lambda", type=float, default=0.95,
                    help="lambda coefficient in GAE formula (default: 0.95, 1 means no gae)")
parser.add_argument("--entropy-coef", type=float, default=0.01,
                    help="entropy term coefficient (default: 0.01)")
parser.add_argument("--value-loss-coef", type=float, default=0.5,
                    help="value loss term coefficient (default: 0.5)")
parser.add_argument("--max-grad-norm", type=float, default=0.5,
                    help="maximum norm of gradient (default: 0.5)")
parser.add_argument("--optim-eps", type=float, default=1e-8,
                    help="Adam and RMSprop optimizer epsilon (default: 1e-8)")
parser.add_argument("--optim-alpha", type=float, default=0.99,
                    help="RMSprop optimizer alpha (default: 0.99)")
parser.add_argument("--clip-eps", type=float, default=0.2,
                    help="clipping epsilon for PPO (default: 0.2)")
parser.add_argument("--recurrence", type=int, default=1,
                    help="number of time-steps gradient is backpropagated (default: 1). If > 1, a LSTM is added to the model to have memory.")
parser.add_argument("--text", action="store_true", default=False,
                    help="add a GRU to the model to handle text input")
parser.add_argument("--heatmap", action="store_true",
                    help="draw heatmap")
parser.add_argument("--noisy_tv", help="whether to add a noisy tv or not")
parser.add_argument("--noise_beta", type=float, default=0.0,
                    help="noise injection beta (default: 0.0)")


def find_index_with_index(arr, value):
    for i, row in enumerate(arr):
        if value in row:
            return (i, row.index(value))
        
        
if __name__ == "__main__":
    args = parser.parse_args()

    args.mem = args.recurrence > 1
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    date = datetime.datetime.now().strftime("%y-%m-%d-%H-%M-%S")
    default_model_name = f"{args.env}_{args.algo}_seed{args.seed}_{date}"

    model_name = args.model or default_model_name
    model_dir = utils.get_model_dir(model_name)

    txt_logger = utils.get_txt_logger(model_dir)
    csv_file, csv_logger = utils.get_csv_logger(model_dir)
    tb_writer = tensorboardX.SummaryWriter(model_dir)

    txt_logger.info("{}\n".format(" ".join(sys.argv)))
    txt_logger.info("{}\n".format(args))

    utils.seed(args.seed)

    if not os.path.exists('./storage/'+args.model + '/state/'):
        os.makedirs('./storage/'+args.model + '/state/')
        os.makedirs('./storage/'+args.model + '/int_fft/')
    txt_logger.info(f"Device: {device}\n")

    envs = []
    for i in range(args.procs):
        env = utils.make_env(args.env, args.seed)
        from minigrid.wrappers import ReseedWrapper
        env = ReseedWrapper(env, seeds=[0,0], seed_idx=0)
        envs.append(env)
    txt_logger.info("Environments loaded\n")

    try:
        status = utils.get_status(model_dir)
    except OSError:
        status = {"num_frames": 0, "update": 0}
    txt_logger.info("Training status loaded\n")

    obs_space, preprocess_obss = utils.get_obss_preprocessor(envs[0].observation_space)
    if "vocab" in status:
        preprocess_obss.vocab.load_vocab(status["vocab"])
    txt_logger.info("Observations preprocessor loaded")

    if args.algo == "rnd_rev":
        print("rnd ac model enter")
        acmodel = RNDACModel(obs_space, envs[0].action_space, args.mem, args.text)
    elif args.algo == "TeCLE":
        print("rnd ac model enter")
        acmodel = RNDACModel(obs_space, envs[0].action_space, args.mem, args.text)
    else:
        print("acmodel enter")
        acmodel = ACModel(obs_space, envs[0].action_space, args.mem, args.text)    
    
    if "model_state" in status:
        acmodel.load_state_dict(status["model_state"])
    acmodel.to(device)
    txt_logger.info("Model loaded\n")
    txt_logger.info("{}\n".format(acmodel))

    if args.algo == "a2c":
        algo = algos.A2CAlgo(envs, acmodel, args.noisy_tv, device, args.frames_per_proc, args.discount, args.lr, args.gae_lambda,
                                args.entropy_coef, args.value_loss_coef, args.max_grad_norm, args.recurrence,
                                args.optim_alpha, args.optim_eps, preprocess_obss)
    elif args.algo == "ppo":
        algo = algos.PPOAlgo(envs, acmodel, args.noisy_tv, device, args.frames_per_proc, args.discount, args.lr, args.gae_lambda,
                                args.entropy_coef, args.value_loss_coef, args.max_grad_norm, args.recurrence,
                                args.optim_eps, args.clip_eps, args.epochs, args.batch_size, preprocess_obss)
    elif args.algo == "TeCLE":
        TeCLE_learning_rate = args.lr
        rnd_obs_clip = 5
        int_discount = 0.99
        clip_grad = False
        algo = algos.PPO_TeCLE_Algo(args, envs, acmodel, args.noise_beta, args.noisy_tv, TeCLE_learning_rate, rnd_obs_clip, int_discount, clip_grad, 
                                  device, args.frames_per_proc, args.discount, args.lr, args.gae_lambda,
                                args.entropy_coef, args.value_loss_coef, args.max_grad_norm, args.recurrence,
                                args.optim_eps, args.clip_eps, args.epochs, args.batch_size, preprocess_obss)
    elif args.algo == "rnd_rev":
        
        algo = algos.PPO_rnd_rev_Algo(envs, acmodel, args.noisy_tv, False, 0.1, False, device, args.frames_per_proc, args.discount, args.lr, args.gae_lambda,
                                args.entropy_coef, args.value_loss_coef, args.max_grad_norm, args.recurrence,
                                args.optim_eps, args.clip_eps, args.epochs, args.batch_size, preprocess_obss)
    elif args.algo == "icm":
        algo = algos.PPO_icm_Algo(envs, acmodel, args.noisy_tv, 1, False, device, args.frames_per_proc, args.discount, args.lr, args.gae_lambda,
                                args.entropy_coef, args.value_loss_coef, args.max_grad_norm, args.recurrence,
                                args.optim_eps, args.clip_eps, args.epochs, args.batch_size, preprocess_obss)
    else:
        raise ValueError("Incorrect algorithm name: {}".format(args.algo))

    if "optimizer_state" in status:
        algo.optimizer.load_state_dict(status["optimizer_state"])
    txt_logger.info("Optimizer loaded\n")

    num_frames = status["num_frames"]
    update = status["update"]
    
    total_visitation_counts = np.zeros(
            (envs[0].unwrapped.width, envs[0].unwrapped.height)
        )
    
    start_time = time.time()
    import time
    start_time = time.time()
    
    num = 1
    while num_frames < args.frames:

        update_start_time = time.time()
        exps, logs1, visit_count = algo.collect_experiences()
        logs2 = algo.update_parameters(exps)
        logs = {**logs1, **logs2}
        update_end_time = time.time()
        
        num_frames += logs["num_frames"]
        update += 1
        
        
        if update % args.log_interval == 0:
            
            fps = logs["num_frames"] / (update_end_time - update_start_time)
            duration = int(time.time() - start_time)
            return_per_episode = utils.synthesize(logs["return_per_episode"])
            rreturn_per_episode = utils.synthesize(logs["reshaped_return_per_episode"])
            num_frames_per_episode = utils.synthesize(logs["num_frames_per_episode"])

            header = ["update", "frames", "FPS", "duration"]
            data = [update, num_frames, fps, duration]
            header += ["rreturn_" + key for key in rreturn_per_episode.keys()]
            data += rreturn_per_episode.values()
            header += ["num_frames_" + key for key in num_frames_per_episode.keys()]
            data += num_frames_per_episode.values()
            
            if args.algo == 'rnd_rev':
                header += ["entropy", "ext_value", "int_value", "policy_loss", "value_loss", "grad_norm" , "rnd_loss", "novel_states_visited", "int_reward"]
                data += [logs["entropy"], logs["ext_value"], logs["int_value"], logs["policy_loss"], logs["value_loss"], logs["grad_norm"], logs["rnd_loss"], logs["novel_states_visited"].mean().item(), logs["intrinsic_rewards"].mean()]
                txt_logger.info(
                    "U {} | F {:06} | FPS {:04.0f} | D {} | rR:μσmM {:.2f} {:.2f} {:.2f} {:.2f} | F:μσmM {:.1f} {:.1f} {} {} | H {:.3f} | Ext_V {:.3f} | Int_V {:.3f} | pL {:.3f} | vL {:.3f} | ∇ {:.3f} | rnd_loss {:.3f} | novel_state {:.3f} | int_reward {:.8f}"
                    .format(*data))
            elif args.algo == 'TeCLE':
                header += ["entropy", "ext_value", "int_value", "policy_loss", "value_loss", "grad_norm" , "TeCLE_loss", "inv_loss", "novel_states_visited", "int_reward", "reward_ratio"]
                data += [logs["entropy"], logs["ext_value"], logs["int_value"], logs["policy_loss"], logs["value_loss"], logs["grad_norm"], logs["TeCLE_loss"], logs["inv_loss"], logs["novel_states_visited"].mean().item(), logs["intrinsic_rewards"].mean(), logs["reward_ratio"]]
                txt_logger.info(
                    "U {} | F {:06} | FPS {:04.0f} | D {} | rR:μσmM {:.2f} {:.2f} {:.2f} {:.2f} | F:μσmM {:.1f} {:.1f} {} {} | H {:.3f} | Ext_V {:.3f} | Int_V {:.3f} | pL {:.3f} | vL {:.3f} | ∇ {:.3f} | TeCLE_loss {:.3f} | inv_loss {:.3f} | novel_state {:.3f} | int_reward {:.8f} | ratio {:.5f}"
                    .format(*data))
            elif args.algo == "icm":
                header += ["entropy", "value", "policy_loss", "value_loss", "grad_norm", "inverse_loss", "forward_loss", "int_reward", "novel_states_visited"]
                data += [logs["entropy"], logs["value"], logs["policy_loss"], logs["value_loss"], logs["grad_norm"], logs["inverse_loss"], logs["forward_loss"], logs["intrinsic_reward"], logs["novel_states_visited"].mean().item()]
                txt_logger.info(
                    "U {} | F {:06} | FPS {:04.0f} | D {} | rR:μσmM {:.2f} {:.2f} {:.2f} {:.2f} | F:μσmM {:.1f} {:.1f} {} {} | H {:.3f} | V {:.3f} | pL {:.3f} | vL {:.3f} | ∇ {:.3f} | inv_L {:.3f} | for_L {:.3f} | int_r {:.8f} | novel_state {:.3f}"
                    .format(*data))
            else:
                header += ["entropy", "value", "policy_loss", "value_loss", "grad_norm", "novel_states_visited"]
                data += [logs["entropy"], logs["value"], logs["policy_loss"], logs["value_loss"], logs["grad_norm"], logs["novel_states_visited"].mean().item()]
                txt_logger.info(
                    "U {} | F {:06} | FPS {:04.0f} | D {} | rR:μσmM {:.2f} {:.2f} {:.2f} {:.2f} | F:μσmM {:.1f} {:.1f} {} {} | H {:.3f} | V {:.3f} | pL {:.3f} | vL {:.3f} | ∇ {:.3f} | novel_state {:.3f}"
                    .format(*data))

            header += ["return_" + key for key in return_per_episode.keys()]
            data += return_per_episode.values()

            if status["num_frames"] == 0:
                csv_logger.writerow(header)
            csv_logger.writerow(data)
            csv_file.flush()

            for field, value in zip(header, data):
                tb_writer.add_scalar(field, value, num_frames)

        if args.save_interval > 0 and update % args.save_interval == 0:
            status = {"num_frames": num_frames, "update": update,
                      "model_state": acmodel.state_dict(), "optimizer_state": algo.optimizer.state_dict()}
            if hasattr(preprocess_obss, "vocab"):
                status["vocab"] = preprocess_obss.vocab.vocab
            utils.save_status(status, model_dir)
            txt_logger.info("Status saved")
    
    end_time = time.time()
    txt_logger.info("total train time : {}".format(end_time-start_time))
