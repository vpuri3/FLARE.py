#
import math
import torch

__all__ = [
    'DecayScheduler',
]

#======================================================================#
class DecayScheduler:
    def __init__(
        self,
        init_val=0.5,
        min_val=0.0,
        total_steps=10000,
        decay_type='cosine',
    ):
        assert decay_type in ['linear', 'constant', 'cosine', 'exp', 'step']
        assert 0 <= init_val <= 1
        assert 0 <= min_val <= 1
        assert 0 <= total_steps

        self.init_val = init_val 
        self.min_val = min_val         
        self.total_steps = total_steps     
        self.decay_type = decay_type       
        self.step_num = 0                  
        
    def reset(self):
        self.step_num = 0
        
    def set_current_step(self, step_num):
        self.step_num = step_num
        
    def step(self):
        self.step_num += 1

    def get_current_val(self):
        progress = self.step_num / self.total_steps
        
        if progress > 1:
            return self.min_val
        
        if self.decay_type == 'linear':
            val = self.init_val * (1 - progress) + self.min_val * progress
        elif self.decay_type == 'constant':
            val = self.init_val
        elif self.decay_type == 'cosine':
            val = self.min_val + 0.5 * (self.init_val - self.min_val) * (1 + math.cos(math.pi * progress))
        elif self.decay_type == 'exp':
            min_val = math.fabs(self.min_val) + 1e-9
            init_val = math.fabs(self.init_val) + 1e-9
            decay_rate = -math.log(min_val / init_val) / self.total_steps
            val = init_val * math.exp(-decay_rate * self.step_num)
        elif self.decay_type == 'step':
            # Example: halve every 25% of steps
            steps_per_drop = self.total_steps // 4
            drop_factor = 0.5 ** (self.step_num // steps_per_drop)
            val = self.init_val * drop_factor
        else:
            raise ValueError(f"Unknown decay_type: {self.decay_type}")
        
        return max(val, self.min_val)

#======================================================================#
if __name__ == '__main__':
    scheduler = DecayScheduler(
        init_val=0.5, min_val=0.0,
        total_steps=10000, decay_type='linear',
    )

    for step in range(0, 10001):
        scheduler.step()
        if step % 1000 == 0:
            val = scheduler.get_current_val()
            print(f"Step {step}: Val = {val:.4f}")

#======================================================================#
#
