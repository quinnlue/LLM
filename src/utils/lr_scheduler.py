import matplotlib.pyplot as plt

class LRScheduler:
    def __init__(self, warmup_steps: int, total_steps: int, min_lr: float, max_lr: float, final_lr: float):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.final_lr = final_lr

        self.anneal_steps = self.total_steps - self.warmup_steps
        self.final_dip_steps = max(1, int(self.anneal_steps * 0.1))
        self.anneal_steps -= self.final_dip_steps

    def __call__(self, step: int):
        if step < self.warmup_steps:
            return self.min_lr + (self.max_lr - self.min_lr) * step / self.warmup_steps
        elif step < self.warmup_steps + self.anneal_steps:
            anneal_step = step - self.warmup_steps
            return self.max_lr - (self.max_lr - self.min_lr) * anneal_step / self.anneal_steps
        elif step < self.total_steps:
            dip_step = step - self.warmup_steps - self.anneal_steps
            return self.min_lr - (self.min_lr - self.final_lr) * dip_step / self.final_dip_steps
        else:
            return self.final_lr


if __name__ == "__main__":
    scheduler = LRScheduler(
        warmup_steps=1000,
        total_steps=10000,
        min_lr=1e-5,
        max_lr=3e-4,
        final_lr=1e-6,
    )

    steps = list(range(10000))
    lrs = [scheduler(step) for step in steps]

    plt.plot(steps, lrs)
    plt.title("1Cycle LR Schedule with Final Dip")
    plt.xlabel("Step")
    plt.ylabel("Learning Rate")
    plt.grid(True)
    plt.show()

