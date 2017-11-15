
def _learning_rate_(iter):
    if iter < 601:
        learning_rate = 0.001
    elif 600 < iter < 32001:
        learning_rate = 0.1
    elif 32000 < iter < 48001:
        learning_rate = 0.01
    elif 48000 < iter < 60001:
        learning_rate = 0.001
    elif 60000 < iter < 80001:
        learning_rate = 0.0001
    elif 80000 < iter < 90001:
        learning_rate = 0.00005
    else:
        learning_rate = 0.00001
    return learning_rate
