from loop import Trainer
from cfg import d
dev = d()
Begins = Trainer(dev.data_dir, dev.batch_size, dev.epochs, dev.save_dir, dev.learning_rate, 'train')
# Trainer.train()
