from AlphaConnectFour import AlphaZeroConnectFourNN, AlphaZero
from connect_four_env import ConnectFourEnv
import torch
import torch.multiprocessing as mp
from torch import optim
from collections import deque
from torch.optim.lr_scheduler import CosineAnnealingLR

def reset_lr(optim, lr):
    for pg in optim.param_groups:
        pg['lr'] = lr

def run_worker(rank, model, args, return_queue):
    env = ConnectFourEnv()
    alpha = AlphaZero(model, None, env, args)
    result = []
    for _ in range(args['num_self_play_iterations']):
        result += alpha.self_play()
    return_queue.put(result)

def main():
    mp.set_start_method("spawn", force=True)
    load_model = True
    args = {
        'C': 2,
        'num_searches': 1000,
        'num_iterations': 20,
        'num_self_play_iterations': 20,
        'epochs': 2,
        'num_processes': 12,
        'res_blocks': 40,
        'num_hidden': 256,
        'batch_size': 256,
        'base-lr': 0.0001,
    }

    env = ConnectFourEnv()
    model = AlphaZeroConnectFourNN(num_resBlocks=args['res_blocks'], num_hidden=args['num_hidden'])
    if load_model:
        model.load_state_dict(torch.load("BestConnect4Model.pt", map_location=torch.device('cpu')))
    model.share_memory()



    optimizer = optim.AdamW(model.parameters(), lr=5e-3)
    if load_model:
        optimizer.load_state_dict(torch.load("BestConnect4Optim.pt", map_location=torch.device('cpu')))
    alpha = AlphaZero(model, optimizer, env, args)
    memory = deque(maxlen=50000)
    if load_model:
        memory += torch.load(f"LastData.pt", weights_only=False)
    scheduler = CosineAnnealingLR(optimizer, T_max=50) 
    for iteration in range(args['num_iterations']):
        model.eval()
        return_queue = mp.Queue()
        processes = []

        for rank in range(args['num_processes']):
            p = mp.Process(target=run_worker, args=(rank, model, args, return_queue))
            p.start()
            processes.append(p)

        for _ in range(args['num_processes']):
            memory += return_queue.get()

        for p in processes:
            p.join()
        memory_list = list(memory)
        model.train()
        reset_lr(optimizer, args['base-lr'])

        # re-create scheduler for this cycle so it decays over `epochs`
        scheduler = CosineAnnealingLR(optimizer, T_max=args['epochs'], eta_min=1e-5)
        for epoch in range(args['epochs']):
            avg_loss, avg_policy, avg_value = alpha.train(memory_list)
            scheduler.step()
            print(f"[Iter {iteration+1}] Epoch {epoch+1}: Total Loss = {avg_loss}, Policy Loss: {avg_policy}, Value Loss: {avg_value}")

        torch.save(model.state_dict(), f"AlphaModel{iteration}.pt")
        torch.save(optimizer.state_dict(), f"AlphaOptim{iteration}.pt")
        torch.save(memory, f"LastData.pt")
    torch.save(model.state_dict(), f"LatestAlphaConnect4.pt")
    torch.save(optimizer.state_dict(), f"LatestOptimizer.pt")
    torch.save(memory, f"LastData.pt")


if __name__ == "__main__":
    main()

