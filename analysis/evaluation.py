import ssim2
import os
import matplotlib.pyplot as plt

def evaluate_folder_ssim(folder_path, 
                    WINDOW_SIZE=None,
                    ALPHA_chosen=None,BETA_chosen=None, GAMMA_chosen=None,
                    plotAll=False,
                    plotScores=False):

    #### Designate Image Source

    # find all images with a name containing 'real'
    real_images = [fileName for fileName in os.listdir(folder_path) if 'real' in fileName]
    real_images.sort()

    # find all images with a name containing 'cycle'
    cycle_images = [fileName for fileName in os.listdir(folder_path) if 'cycle' in fileName]
    cycle_images.sort()

    # find all images with a name containing 'pix2pix'
    pix2pix_images = [fileName for fileName in os.listdir(folder_path) if 'pix2pix' in fileName]
    pix2pix_images.sort()

    # find all images with a name containing 'ground_truth'
    ground_truth_images = [fileName for fileName in os.listdir(folder_path) if 'ground_truth' in fileName]

    ##### Get the SSIM scores for each image
    ssim_scores_pix2pix = []
    ssim_scores_cycle = []
    
    for i in range(len(real_images)):
        image_number = real_images[i].split('_')[0]

        ssim_scores_pix2pix.append(ssim2.ssim_score(os.path.join(folder_path, real_images[i]), 
                                                    os.path.join(folder_path, pix2pix_images[i]),
                                                    WINDOW_SIZE=WINDOW_SIZE,
                                                    ALPHA_chosen=ALPHA_chosen, BETA_chosen=BETA_chosen, GAMMA_chosen=GAMMA_chosen,
                                                    plotOn=plotAll,
                                                    image_number=image_number,
                                                    generation_type='PIX'))
        ssim_scores_cycle.append(ssim2.ssim_score(os.path.join(folder_path, real_images[i]),
                                                    os.path.join(folder_path, cycle_images[i]),
                                                    WINDOW_SIZE=WINDOW_SIZE,
                                                    ALPHA_chosen=ALPHA_chosen, BETA_chosen=BETA_chosen, GAMMA_chosen=GAMMA_chosen,
                                                    plotOn=plotAll,
                                                    image_number=image_number,
                                                    generation_type='CYC'))
        
    if plotScores:
        plt.plot(ssim_scores_pix2pix, label='pix2pix')
        plt.plot(ssim_scores_cycle, label='CycleGAN')
        plt.legend()
        plt.show()
        
    return ssim_scores_pix2pix, ssim_scores_cycle

folder_path = 'analysis/cycle_pix_maps_compared'

ssim_scores_pix2pix, ssim_scores_cycle = evaluate_folder_ssim(folder_path, plotScores=True)

