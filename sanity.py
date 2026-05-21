from env import ApartmentEnv

if __name__=="__main__":
    apt = ApartmentEnv(4, 4)
    apt.reset()
    while True:
        if apt.state[1] > 2:
            action = 1
        else:
            action = 0
        
        obs, reward, terminated, truncated, info = apt.step(action)
        
        print(obs[0], obs[1], int(reward), terminated, truncated, info)
        if terminated == True:
            break
    