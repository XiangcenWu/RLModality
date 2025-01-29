import torch



def test_agent(env, agent, num_steps, device='cpu', random=False):
    obs = env.reset()
    obs = obs.unsqueeze(0).to(device)
    _list = []
    for _ in range(num_steps):
        with torch.no_grad():
            if random:
                action = (
                    torch.randint(0, 4, size=(1, )).item(),
                    torch.randint(0, 4, size=(1, )).item()
                )
            else:
                agent.actor.eval()
                action, probs, value = agent.choose_action(obs, noise=0)
            print(action)
            next_obs, reward= env.step_train(action)
            
            obs = next_obs.unsqueeze(0).to(device)
            _list.append(env.calculate_current_accuracy())

    return _list
        