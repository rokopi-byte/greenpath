"""Test script.
Usage:
  test.py <hparams> <checkpoint_dir> <dataset_root> [--cuda=<id>]
  test.py -h | --help

Options:
  -h --help     Show this screen.
  --cuda=<id>   id of the cuda device [default: 0].
"""

import os
import json
import torch
import numpy as np
from torch.utils.data.dataset import Dataset
from tqdm import tqdm
from docopt import docopt
from os.path import join
from dataset import process_data
from irl_dcb.config import JsonConfig
from sklearn.model_selection import train_test_split

import cv2 as cv

from irl_dcb.data import LHF_IRL
from irl_dcb.models import LHF_Policy_Cond_Small, LHF_Policy_Cond_Big
from irl_dcb.environment import IRL_Env4LHF
from irl_dcb import utils
from irl_dcb import metrics

torch.manual_seed(42620)
np.random.seed(42620)


def gen_scanpaths(generator,
                  env_test,
                  test_img_loader,
                  patch_num,
                  max_traj_len,
                  im_w,
                  im_h,
                  num_sample=10,
                  fixations=None):
    all_actions = []
    for i_sample in range(num_sample):
        progress = tqdm(test_img_loader,
                        desc='trial ({}/{})'.format(i_sample + 1, num_sample))
        for i_batch, batch in enumerate(progress):
            img_names_batch = batch['img_name']
            cat_names_batch = batch['cat_name']
            with torch.no_grad():
                env_test.set_data(batch)
                trajs = utils.collect_trajs(env_test,
                                            generator,
                                            patch_num,
                                            max_traj_len,
                                            is_eval=True,
                                            sample_action=True)

                all_actions.extend([(cat_names_batch[i], img_names_batch[i],
                                     'present', trajs['actions'][:, i], trajs['init'][i])
                                    for i in range(env_test.batch_size)])

    scanpaths = utils.actions2scanpaths(all_actions, patch_num, im_w, im_h, trajs)
    utils.cutFixOnTarget(scanpaths, bbox_annos)

    return scanpaths


if __name__ == '__main__':
    args = docopt(__doc__)
    device = torch.device('cuda:{}'.format(args['--cuda']) if torch.cuda.is_available() else "cpu")
    hparams = args["<hparams>"]
    dataset_root = args["<dataset_root>"]
    checkpoint = args["<checkpoint_dir>"]
    hparams = JsonConfig(hparams)

    # dir of pre-computed beliefs
    DCB_dir_HR = join(dataset_root, 'DCBs/HR/')
    DCB_dir_LR = join(dataset_root, 'DCBs/LR/')
    data_name = '{}x{}'.format(hparams.Data.im_w, hparams.Data.im_h)

    # bounding box of the target object (for search efficiency evaluation)
    bbox_annos = np.load(join(dataset_root, 'bbox_annos.npy'),
                         allow_pickle=True).item()

    with open(join(dataset_root,
                   'dataset_train.json')) as json_file:
        human_scanpaths_train = json.load(json_file)

    with open(join(dataset_root,
                   'dataset_valid.json')) as json_file:
        human_scanpaths_valid = json.load(json_file)

    with open(join(dataset_root,
                   'dataset_test.json')) as json_file:
        test_set = json.load(json_file)

    with open(join(dataset_root,
                   'first_fixations.json')) as json_file:
        fixations = json.load(json_file)

        # target_init_fixs[key] = (traj['X'][0] / hparams.Data.im_w[traj['name']],
        #                         traj['Y'][0] / hparams.Data.im_h[traj['name']])

    cat_names = list(np.unique([x['task'] for x in human_scanpaths_train]))
    catIds = dict(zip(cat_names, list(range(len(cat_names)))))

    dataset = process_data(human_scanpaths_train, human_scanpaths_valid,
                           DCB_dir_HR, DCB_dir_LR, fixations, None, bbox_annos, hparams)

    tasks = np.unique([traj['task'] for traj in human_scanpaths_train + human_scanpaths_valid])
    names = np.unique([traj['name'] for traj in human_scanpaths_train + human_scanpaths_valid + test_set])

    # tasks = np.array(["beer"])
    # test_set = [x for x in test_set if x["task"] == "beer"]

    target_init_fixs = {}
    train_task_img_pair = []
    for i in tasks:
        for n in names:
            train_task_img_pair.append(i + "_" + n)
            target_init_fixs[i + "_" + n] = (fixations[n[:-4]][0] / 46,
                                             fixations[n[:-4]][1] / 46)

    # train_task_img_pair = np.unique(train_task_img_pair)

    test_dataset = LHF_IRL(DCB_dir_HR, DCB_dir_LR, target_init_fixs, train_task_img_pair, bbox_annos,
                           hparams.Data, catIds)
    num_workers = 0 if device.type == "cpu" else 6
    dataloader = torch.utils.data.DataLoader(test_dataset,
                                             batch_size=16,
                                             shuffle=False,
                                             num_workers=num_workers)

    _, first_fixations, _ = utils.preprocess_fixations(
        test_set + human_scanpaths_valid,
        hparams.Data.patch_size,
        hparams.Data.patch_num,
        hparams.Data.im_h,
        hparams.Data.im_w,
        truncate_num=hparams.Data.max_traj_length)

    # load trained model
    input_size = 30  # number of belief maps
    task_eye = torch.eye(len(dataset['catIds'])).to(device)
    generator = LHF_Policy_Cond_Small(hparams.Data.patch_count,
                                      len(dataset['catIds']), task_eye,
                                      input_size).to(device)

    # state = torch.load(join(checkpoint, 'trained_generator.pkg'), map_location=device)

    # generator.eval()
    # generator.load_state_dict(state["model"])
    # utils.load('best', generator, 'generator_big', pkg_dir=checkpoint)
    utils.load('best', generator, 'generator', pkg_dir=checkpoint, device=device)
    generator.eval()

    # build environment
    env_test = IRL_Env4LHF(hparams.Data,
                           max_step=hparams.Data.max_traj_length,
                           mask_size=hparams.Data.IOR_size,
                           status_update_mtd=hparams.Train.stop_criteria,
                           device=device,
                           inhibit_return=True, init_mtd='real')
    env_test.set_first_fixations(first_fixations)

    label = np.zeros((len(test_set)))
    test_1, test_2, _, _ = train_test_split(test_set, label, test_size=0.5)

    # evaluate predictions
    real_mean_cdf, _ = utils.compute_search_cdf(test_set, bbox_annos,
                                                hparams.Data.max_traj_length, False)
    print("Human Mean CDF:", real_mean_cdf)
    # CDF-AUC
    real_cdf_auc = metrics.compute_cdf_auc(real_mean_cdf)
    print("Human CDF AUC:", real_cdf_auc)

    real_lcs, real_lcs_count = metrics.averageLCS(test_1, test_2, hparams.Data.patch_size)
    print("Human LCS:", real_lcs, real_lcs_count)
    real_value = metrics.dtw(test_1, test_2, hparams.Data.im_w, hparams.Data.im_h)
    print("Human DTW:", real_value)

    value = {"real": np.array([]), "first": np.array([]), "random": np.array([])}
    lcs = {"real": np.array([]), "first": np.array([]), "random": np.array([])}
    lcs_count = {"real": np.array([]), "first": np.array([]), "random": np.array([])}
    cdf = {"real": np.array([]), "first": np.array([]), "random": np.array([])}
    cdf_auc = {"real": np.array([]), "first": np.array([]), "random": np.array([])}
    #
    random_generator = LHF_Policy_Cond_Small(hparams.Data.patch_count,
                                             len(dataset['catIds']), task_eye,
                                             input_size).to(device)
    #
    predictions_random = gen_scanpaths(random_generator,
                                       env_test,
                                       dataloader,
                                       hparams.Data.patch_num,
                                       hparams.Data.max_traj_length,
                                       hparams.Data.im_w,
                                       hparams.Data.im_h,
                                       num_sample=10,
                                       fixations=fixations)

    random_value = metrics.dtw(predictions_random, test_set, hparams.Data.im_w, hparams.Data.im_h)
    print("Random DTW:", random_value)

    random_lcs, random_lcs_count = metrics.averageLCS(predictions_random, test_set, hparams.Data.patch_size)
    print("Random LCS:", random_lcs, random_lcs_count)

    random_mean_cdf, _ = utils.compute_search_cdf(predictions_random, bbox_annos,
                                                  hparams.Data.max_traj_length)
    print("Random Mean CDF:", random_mean_cdf)
    # CDF-AUC
    random_cdf_auc = metrics.compute_cdf_auc(random_mean_cdf)
    print("Random CDF AUC:", random_cdf_auc)

    for it in range(5):
        for type_init in ["real", "first", "random"]:
            env_test.init = type_init
            # generate scanpaths
            print('sample scanpaths for init type', type_init, 'and iteration n.' + str(it) + '...')
            predictions = gen_scanpaths(generator,
                                        env_test,
                                        dataloader,
                                        hparams.Data.patch_num,
                                        hparams.Data.max_traj_length,
                                        hparams.Data.im_w,
                                        hparams.Data.im_h,
                                        num_sample=10,
                                        fixations=fixations)

            # value = metrics.compute_mm(human_scanpaths_train, predictions, hparams.Data.im_w, hparams.Data.im_h)
            current_value = metrics.dtw(predictions, test_set,
                                        hparams.Data.im_w, hparams.Data.im_h)
            print("Current DTW:", current_value)
            current_lcs, current_lcs_count = metrics.averageLCS(predictions, test_set, hparams.Data.patch_size)
            print("Current LCS:", current_lcs, current_lcs_count)
            current_mean_cdf, _ = utils.compute_search_cdf(predictions, bbox_annos,
                                                           hparams.Data.max_traj_length)
            print("Current Mean CDF:", current_mean_cdf)
            # CDF-AUC
            current_cdf_auc = metrics.compute_cdf_auc(current_mean_cdf)
            print("Current CDF AUC:", current_cdf_auc)

            if len(cdf[type_init]) == 0:
                cdf[type_init] = np.array(current_mean_cdf)
            # else:
            #    cdf[type_init] = np.concatenate(cdf[type_init], np.array(current_mean_cdf))
            cdf_auc[type_init] = np.append(cdf_auc[type_init], current_cdf_auc)
            value[type_init] = np.append(value[type_init], current_value)
            lcs[type_init] = np.append(lcs[type_init], current_lcs)
            lcs_count[type_init] = np.append(lcs_count[type_init], current_lcs_count)

            for elem in predictions:
                filename = elem['task'] + "/" + elem['name']
                key = elem['task'] + '_' + elem['name']
                # print(filename)
                # print("../dataset/" + elem['name'].split('.')[0] + "/cut_" + elem['name'])
                image = cv.imread("../dataset/" + elem['name'].split('.')[0] + "/cut_" + elem['name'])

                X = elem['X']
                Y = elem['Y']

                image = cv.resize(image, (hparams.Data.im_w[elem['name']], hparams.Data.im_h[elem['name']]))

                task = bbox_annos[key][0]
                pt1 = (task[0], task[1])
                pt2 = (task[0] + task[2], task[1] + task[3])
                image = cv.rectangle(image, pt1, pt2, (0, 255, 255), 2)

                for i in range(len(X)):
                    x = int(X[i])
                    y = int(Y[i])
                    cv.putText(image, str(i), (x, y), cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 3)
                    cv.circle(image, (x, y), 4, (0, 0, 0))
                    if i > 0:
                        xprec = int(X[i - 1])
                        yprec = int(Y[i - 1])
                        cv.line(image, (xprec, yprec), (x, y), (0, 0, 0), 3)

                os.makedirs("./results/" + elem['task'] + "/", exist_ok=True)
                cv.imwrite("./results/" + elem['task'] + "/" + elem['name'][:-4] + "_" + type_init +
                           "_" + str(it) + ".jpg", image)
    value["real"] = np.mean(value["real"])
    value["random"] = np.mean(value["random"])
    value["first"] = np.mean(value["first"])

    lcs["real"] = np.mean(lcs["real"])
    lcs["random"] = np.mean(lcs["random"])
    lcs["first"] = np.mean(lcs["first"])

    lcs_count["real"] = np.mean(lcs_count["real"])
    lcs_count["random"] = np.mean(lcs_count["random"])
    lcs_count["first"] = np.mean(lcs_count["first"])

    cdf_auc["real"] = np.mean(cdf_auc["real"])
    cdf_auc["random"] = np.mean(cdf_auc["random"])
    cdf_auc["first"] = np.mean(cdf_auc["first"])

    print("CDF-AUC:", cdf_auc)
    print("BTW:", value)
    print("LCS:", lcs)
    print("LCS Count:", lcs_count)

    print("CDF:", cdf)

    print("Human BTW:", real_value)
    print("Human LCSS:", real_lcs)
    print("Human LCSS Count:", real_lcs_count)
    print("Human Mean CDF:", real_mean_cdf)
    print("Human CDF AUC:", real_cdf_auc)

    print("Random BTW:", random_value)
    print("Random LCSS:", random_lcs)
    print("Random LCSS Count:", random_lcs_count)
    print("Random Mean CDF:", random_mean_cdf)
    print("Random CDF AUC:", random_cdf_auc)

# cdf["real"] = np.mean(cdf["real"], axis=1)
# cdf["random"] = np.mean(cdf["random"], axis=1)
# cdf["first"] = np.mean(cdf["first"], axis=1)
