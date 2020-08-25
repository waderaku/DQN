class Parameter_Set:
    def __init__(self, eps_init, eps_anneal, eps_min, lr, gamma, cap, q_update):
        self.eps_init = eps_init
        self.eps_anneal = eps_anneal
        self.eps_min = eps_min
        self.lr = lr
        self.gamma = gamma
        self.cap = cap
        self.q_update = q_update

    def update(self, eps_init, eps_anneal, eps_min, lr, gamma, cap, q_update):
        if eps_init:
            self.eps_init = eps_init

        if eps_anneal:
            self.eps_anneal = eps_anneal

        if eps_min:
            self.eps_min = eps_min

        if lr:
            self.lr = lr

        if gamma:
            self.gamma = gamma

        if cap:
            self.cap = cap

        if q_update:
            self.q_update = q_update
