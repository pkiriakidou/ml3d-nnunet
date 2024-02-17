import inspect
import os
from copy import deepcopy
from os.path import join
from typing import Generator, Union, Tuple, List, Optional

import numpy as np
import torch
import torch.nn as nn
from batchgenerators.utilities.file_and_folder_operations import maybe_mkdir_p, save_json
from torch.utils.data import Dataset

from inference.export_prediction import export_prediction_from_logits, \
    convert_predicted_logits_to_segmentation_with_correct_shape
from inference.predict_from_raw_data import nnUNetPredictor
from nnunetv2.imageio.simpleitk_reader_writer import SimpleITKIO
from utilities.json_export import recursive_fix_for_json_export


class nnUNetPredictorWrapper(nn.Module):
    def __init__(self, model, wrapped):
        super().__init__()
        self.wrapped: nnUNetPredictor = wrapped
        self.model = model


        #img, props = SimpleITKIO().read_images([join(nnUNet_raw, 'Dataset003_Liver/imagesTr/liver_63_0000.nii.gz')])
        #ret = predictor.predict_single_npy_array(img, props, None, None, False)
#
        #iterator = predictor.get_data_iterator_from_raw_npy_data([img], None, [props], None, 1)



    def forward(self, input: tuple[dict, torch.Tensor]):
        ofile: str = input[0]['ofile']
        data = input[1]
        properties = input[0]['data_properties']
        prediction = self.wrapped.predict_logits_from_preprocessed_data(data).cpu()



        if ofile is not None:
            # this needs to go into background processes
            # export_prediction_from_logits(prediction, properties, configuration_manager, plans_manager,
            #                               dataset_json, ofile, save_probabilities)
            print('sending off prediction to background worker for resampling and export')
            ret = export_prediction_from_logits(prediction, properties, self.wrapped.configuration_manager, self.wrapped.plans_manager,
                      self.wrapped.dataset_json, ofile, False)
        else:
            # convert_predicted_logits_to_segmentation_with_correct_shape(prediction, plans_manager,
            #                                                             configuration_manager, label_manager,
            #                                                             properties,
            #                                                             save_probabilities)
            print('sending off prediction to background worker for resampling')
            ret= convert_predicted_logits_to_segmentation_with_correct_shape(prediction, self.plans_manager,
                         self.configuration_manager, self.label_manager,
                         properties,
                         False)

        if ofile is not None:
            print(f'done with {os.path.basename(ofile)}')
        else:
            print(f'\nDone with image of shape {data.shape}:')


        return ret

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            if name == "wrapped":
                raise AttributeError()
            return getattr(self.wrapped, name)

class nnUNetPredictorDataset(Dataset):
    def __init__(self, wrapped):
        self.data_iterator: Generator = None
        self.wrapped: nnUNetPredictor = wrapped
        #for preprocessed in data_iterator:
        #    data = preprocessed['data']
        #    if isinstance(data, str):
        #        delfile = data
        #        data = torch.from_numpy(np.load(data))
        #        os.remove(delfile)
#
        #    ofile = preprocessed['ofile']
        #    if ofile is not None:
        #        print(f'\nPredicting {os.path.basename(ofile)}:')
        #    else:
        #        print(f'\nPredicting image of shape {data.shape}:')
#
        #    print(f'perform_everything_on_device: {self.perform_everything_on_device}')
#
        #    properties = preprocessed['data_properties']
#
#
        #    #prediction = self.predict_logits_from_preprocessed_data(data).cpu()

    def predict_from_files(self,
                           list_of_lists_or_source_folder: Union[str, List[List[str]]],
                           output_folder_or_list_of_truncated_output_files: Union[str, None, List[str]],
                           save_probabilities: bool = False,
                           overwrite: bool = True,
                           num_processes_preprocessing: int = 1,
                           num_processes_segmentation_export: int = 1,
                           folder_with_segs_from_prev_stage: str = None,
                           num_parts: int = 1,
                           part_id: int = 0):
        """
        This is nnU-Net's default function for making predictions. It works best for batch predictions
        (predicting many images at once).
        """
        if isinstance(output_folder_or_list_of_truncated_output_files, str):
            output_folder = output_folder_or_list_of_truncated_output_files
        elif isinstance(output_folder_or_list_of_truncated_output_files, list):
            output_folder = os.path.dirname(output_folder_or_list_of_truncated_output_files[0])
        else:
            output_folder = None

        ########################
        # let's store the input arguments so that its clear what was used to generate the prediction
        if output_folder is not None:
            my_init_kwargs = {}
            for k in inspect.signature(self.predict_from_files).parameters.keys():
                my_init_kwargs[k] = locals()[k]
            my_init_kwargs = deepcopy(
                my_init_kwargs)  # let's not unintentionally change anything in-place. Take this as a
            recursive_fix_for_json_export(my_init_kwargs)
            maybe_mkdir_p(output_folder)
            save_json(my_init_kwargs, join(output_folder, 'predict_from_raw_data_args.json'))

            # we need these two if we want to do things with the predictions like for example apply postprocessing
            save_json(self.wrapped.dataset_json, join(output_folder, 'dataset.json'), sort_keys=False)
            save_json(self.wrapped.plans_manager.plans, join(output_folder, 'plans.json'), sort_keys=False)
        #######################

        # check if we need a prediction from the previous stage
        if self.wrapped.configuration_manager.previous_stage_name is not None:
            assert folder_with_segs_from_prev_stage is not None, \
                f'The requested configuration is a cascaded network. It requires the segmentations of the previous ' \
                f'stage ({self.wrapped.configuration_manager.previous_stage_name}) as input. Please provide the folder where' \
                f' they are located via folder_with_segs_from_prev_stage'

        # sort out input and output filenames
        list_of_lists_or_source_folder, output_filename_truncated, seg_from_prev_stage_files = \
            self.wrapped._manage_input_and_output_lists(list_of_lists_or_source_folder,
                                                output_folder_or_list_of_truncated_output_files,
                                                folder_with_segs_from_prev_stage, overwrite, part_id, num_parts,
                                                save_probabilities)
        if len(list_of_lists_or_source_folder) == 0:
            return
        self.data_iterator = self.wrapped._internal_get_data_iterator_from_lists_of_filenames(list_of_lists_or_source_folder,
                                                                                 seg_from_prev_stage_files,
                                                                                 output_filename_truncated,
                                                                                 num_processes_preprocessing)

    def __len__(self):
        return len([x for x in self.data_iterator])#len(self.data_iterator)

    def __next__(self):
        preprocessed = next(self.data_iterator)
        if preprocessed is None: return None
        data = preprocessed['data']
        if isinstance(data, str):
            delfile = data
            data = torch.from_numpy(np.load(data))
            os.remove(delfile)
        return tuple([preprocessed, data])

    def __getitem__(self, idx):
        return next(self)
