
from torchvision import transforms

class Config():
    #####################################################
    # Structural Hyperparameters
    #####################################################
    epoch = 0               # epoch to start training from
    num_epochs = 200        # number of epochs of training
    batch_size = 16          # size of the batches
    dataroot = 'data'       # root directory of the dataset
    lr = .0002              # initial learning rate
    b1 = .5                 # momentum term of adam
    b2 = .999               # adam: decay of first order momentum of gradient
    decay_epoch = 100       # epoch from which to start lr decay
    size = 256              # size of the data crop (squared assumed)
    input_num_channels = 3  # number of channels in input images
    output_num_channels = 3 # number of channels in output images
    cuda = True             # use GPU?
    num_cpu = 8             # number of cpu threads to use during batch generation

    #####################################################
    # Loss Function Hyperparameters (tinker with these!!)
    #####################################################
    # Which loss functions do you want?
    IDENTITY_LOSS_INCLUDED=True             # default: True (tinker with this)
    CYCLE_LOSS_INCLUDED=True                # default: True
    GAN_LOSS_INCLUDED=True                  # default: True   

    # How do you want to prioritize those loss functions?
    IDENTITY_WEIGHT = 5.0                   # default: 5.0
    CYCLE_WEIGHT = 10.0                     # default: 10.0
    GAN_WEIGHT = 1.0                        # default: 1.0

    GOOD_TRUST_DISCR_WEIGHT = 0.5           # default: 0.5
    BAD_GULLIBILITY_DISCR_WEIGHT = 0.5      # default: 0.5

    #####################################################
    #  WHERE TO SAVE THE MODEL
    #####################################################
    # Paths
    train_x_dir = './data/train/input'
    train_y_dir = './data/train/output'
    val_input_dir = './data/val/input'
    val_target_dir = './data/val/output'
    # Checkpoint and output directories
    checkpoint_dir = './runs/red_delicious/checkpoints'
    sample_dir = './runs/red_delicious/samples'
    log_dir = './runs/red_delicious/logs'

    # Data transformations
    data_transforms = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
