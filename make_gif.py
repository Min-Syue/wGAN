from model.funs import make_gif

if __name__ == '__main__':
    
    # file_path = '/photo_cGAN/photo_mnist/Images_Epochs_'\
    file_path = '/photo_wGAN/photo_mnist/Images_Epochs_'
    make_gif(file_path, 'wGAN_4000epochs_mnist', interval=200, top_range=4000)