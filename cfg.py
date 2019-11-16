class d:
    def __init__(self):
        self.img_height = 64
        self.img_width = 64
        self.noise_dim = 100
        self.emb_dim = 1024
        self.projected_em_dim = 128
        self.batch_size = 64
        self.learning_rate = 0.5
        self.epochs = 100
        self.cuda = False
        self.data_dir = r'birds.hdf5'
        self.save_dir = r'SavedModels'
