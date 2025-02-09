import torch



def test_agent(env, agent, num_steps, device='cpu', random=False):
    obs = env.reset()
    obs = obs.unsqueeze(0).to(device)
    _list = []
    last_action = None
    for _ in range(num_steps):
        with torch.no_grad():
            if random:
                action =torch.randint(0, 32, size=(1, )).item()
            else:
                agent.actor.eval()
                action = agent.choose_action_inference(obs, last_action=last_action)
            print(action)
            next_obs, reward= env.step_train(action)
            
            obs = next_obs.unsqueeze(0).to(device)
            last_action = 9999999
            _list.append(env.calculate_current_accuracy())

    return _list
        