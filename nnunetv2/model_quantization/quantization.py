from pytorch_quantization import nn as quant_nn
import pytorch_quantization
from pytorch_quantization import calib
from pytorch_quantization.tensor_quant import QuantDescriptor
from torch import nn, ao
from torch.ao.quantization.backend_config import get_tensorrt_backend_config_dict
from torch.utils.data import Dataset

from generic_pytorch_wrapper import nnUNetPredictorDataset, nnUNetPredictorWrapper
from nnunetv2.paths import nnUNet_results, nnUNet_raw
import torch
from batchgenerators.utilities.file_and_folder_operations import join
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from pytorch_quantization import quant_modules
import time

def first_attempt(): #idk
    print(torch.cuda.is_available())

    # quant_modules.initialize()
    # quant_desc_input = QuantDescriptor(calib_method='max')
    # quant_nn.QuantConv2d.set_default_quant_desc_input(quant_desc_input)
    # quant_nn.QuantLinear.set_default_quant_desc_input(quant_desc_input)

    # instantiate the nnUNetPredictor
    predictor = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=True,
        perform_everything_on_device=True,
        device=torch.device('cuda', 0),
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=True
    )
    # initializes the network architecture, loads the checkpoint
    predictor.initialize_from_trained_model_folder(
        join(nnUNet_results, 'Dataset009_Spleen/nnUNetTrainer__nnUNetPlans__3d_fullres'),
        use_folds=(0,),
        checkpoint_name='checkpoint_final.pth',
    )
    model: nn.Module = predictor.network
    import os

    def print_model_size(mdl):
        torch.save(mdl.state_dict(), "tmp.pt")
        print("%.2f MB" % (os.path.getsize("tmp.pt") / 1e6))
        os.remove('tmp.pt')

    backend = 'fbgemm'
    model.qconfig = torch.quantization.get_default_qconfig(backend)
    torch.backends.quantized.engine = backend
    model_static_quantized = torch.quantization.prepare(model, inplace=False)
    model_static_quantized = torch.quantization.convert(model_static_quantized, inplace=False)
    print_model_size(model_static_quantized)

    torch.save({
        'model_state_dict': model.state_dict(),
    }, 'before_quant.pth')
    for name, module in model.named_modules():
        if name.endswith('_quantizer'):
            module.enable_calib()
            module.disable_quant()  # Use full precision data to calibrate

    # variant 1: give input and output folders
    input_dir = join(nnUNet_raw, 'Dataset009_Spleen/imagesTs')
    output_dir = join(nnUNet_raw, 'Dataset009_Spleen/imagesTs_predlowres')

    def call_predict():

        predictor.predict_from_files([[join(input_dir, 'spleen_1_0000.nii.gz')]],
                                     [join(input_dir, 'spleen_1.nii.gz')],  #
                                     save_probabilities=False, overwrite=True,
                                     num_processes_preprocessing=2, num_processes_segmentation_export=2,
                                     folder_with_segs_from_prev_stage=None, num_parts=1, part_id=0)

    # call_predict()

    #
    for name, module in model.named_modules():
        if name.endswith('_quantizer'):
            module.load_calib_amax(strict=False)
            module.disable_calib()
            module.enable_quant()
    predictor.list_of_parameters = [model.state_dict()]
    model.cuda()
    torch.save({
        'model_state_dict': model.state_dict(),
    }, 'after_quant.pth')

    # print('amax print:\n', [x for x in model.state_dict().keys() if 'amax' in x])
    # call_predict()

    # with pytorch_quantization.enable_onnx_export():
    #    # enable_onnx_checker needs to be disabled. See notes below.
    #    torch.onnx.export(
    #        model, , "quant_resnet50.onnx", verbose=True, opset_version=10, enable_onnx_checker=False
    #    )

def chinese_fx_attempt(): # errors
    predictor = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=True,
        perform_everything_on_device=True,
        device= torch.device('cuda', 0),
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=True
    )
    # initializes the network architecture, loads the checkpoint
    predictor.initialize_from_trained_model_folder(
        join(nnUNet_results, 'Dataset009_Spleen/nnUNetTrainer__nnUNetPlans__3d_fullres'),
        use_folds=(0,),
        checkpoint_name='checkpoint_final.pth',
    )
    model: nn.Module = predictor.network
    import os

    def print_model_size(mdl):
        torch.save(mdl.state_dict(), "tmp.pt")
        print("%.2f MB" % (os.path.getsize("tmp.pt") / 1e6))
        os.remove('tmp.pt')

    input_dir = join(nnUNet_raw, 'Dataset009_Spleen/imagesTs')
    output_dir = join(nnUNet_raw, 'Dataset009_Spleen/imagesTs_predlowres')
    dataset = nnUNetPredictorDataset(predictor)
    dataset.predict_from_files([[join(input_dir, 'spleen_1_0000.nii.gz')]],
                                    [join(input_dir, 'spleen_1.nii.gz')],  #
                                    save_probabilities=False, overwrite=True,
                                    num_processes_preprocessing=2, num_processes_segmentation_export=2,
                                    folder_with_segs_from_prev_stage=None, num_parts=1, part_id=0)
    wrapper = nnUNetPredictorWrapper(predictor.network, predictor)

    from torch.quantization import get_default_qconfig
    from torch.quantization.quantize_fx import prepare_fx, convert_fx

    def calibrate(model: nn.Module, data_loader: Dataset):  # 校准功能函数
        model.eval()
        with torch.no_grad():
            for image in data_loader:
                model(image)
    def call_predict():

        predictor.predict_from_files([[join(input_dir, 'spleen_1_0000.nii.gz')]],
                                     [join(input_dir, 'spleen_1.nii.gz')],  #
                                     save_probabilities=False, overwrite=True,
                                     num_processes_preprocessing=2, num_processes_segmentation_export=2,
                                     folder_with_segs_from_prev_stage=None, num_parts=1, part_id=0)

    qconfig = ao.quantization.qconfig.QConfig(
       activation=ao.quantization.observer.HistogramObserver.with_args(
           qscheme=torch.per_tensor_symmetric, dtype=torch.qint8
       ),
       weight=ao.quantization.observer.default_per_channel_weight_observer
    )
    prepared = prepare_fx(wrapper, {"": qconfig,
                                    "object_type": [  # 这里设置反卷积的量化规则，注意看维度的per-channel量化ch_axis=1
                                        (torch.nn.ConvTranspose2d,
                                         ao.quantization.qconfig.QConfig(
                                             activation=ao.quantization.observer.HistogramObserver.with_args(
                                                 qscheme=torch.per_tensor_symmetric, dtype=torch.qint8,
                                             ),
                                             weight=ao.quantization.observer.PerChannelMinMaxObserver.with_args(
                                                 ch_axis=1, dtype=torch.qint8, qscheme=torch.per_channel_symmetric)))
                                    ]
                                    },
                         example_inputs=next(dataset),
                         backend_config=get_tensorrt_backend_config_dict()
                         )

    #calibrate(prepared, dataset)  # 校准数据集进行标准
    call_predict()
    quantized_model = convert_fx(prepared)  # 把校准后的模型转化为量化版本模型
    wrapper.model.fuse_model()

    # Specify quantization configuration
    # Start with simple min/max range estimation and per-tensor quantization of weights
    #predictor.network.qconfig = torch.ao.quantization.default_qconfig
    #print(predictor.network.qconfig)
    #torch.ao.quantization.prepare(predictor.network, inplace=True)
#
    ## Calibrate first
    #print('Post Training Quantization Prepare: Inserting Observers')
    ## print('\n Inverted Residual Block:After observer insertion \n\n', wrapper.features[1].conv)
#
    ## Calibrate with the training set
    #predictor.list_of_parameters = [model.state_dict()]
    #predictor.predict_from_files([[join(input_dir, 'spleen_1_0000.nii.gz')]],
    #                             [join(input_dir, 'spleen_1.nii.gz')],  #
    #                             save_probabilities=False, overwrite=True,
    #                             num_processes_preprocessing=2, num_processes_segmentation_export=2,
    #                             folder_with_segs_from_prev_stage=None, num_parts=1, part_id=0)
    ## calibrate(wrapper, dataset)
    #print('Post Training Quantization: Calibration done')
#
    ## Convert to quantized model
    #quantized_model = torch.ao.quantization.convert(predictor.network, inplace=True)
    print_model_size(quantized_model)

def static_quant(): #reduced size, can not run as not on gpu

    # quant_modules.initialize()
    # quant_desc_input = QuantDescriptor(calib_method='max')
    # quant_nn.QuantConv2d.set_default_quant_desc_input(quant_desc_input)
    # quant_nn.QuantLinear.set_default_quant_desc_input(quant_desc_input)

    # instantiate the nnUNetPredictor
    predictor = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=True,
        perform_everything_on_device=True,
        device=torch.device('cuda', 0),
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=True
    )
    # initializes the network architecture, loads the checkpoint
    predictor.initialize_from_trained_model_folder(
        join(nnUNet_results, 'Dataset009_Spleen/nnUNetTrainer__nnUNetPlans__3d_fullres'),
        use_folds=(0,),
        checkpoint_name='checkpoint_final.pth',
    )
    model: nn.Module = predictor.network
    import os

    def print_model_size(mdl):
        torch.save(mdl.state_dict(), "tmp.pt")
        print("%.2f MB" % (os.path.getsize("tmp.pt") / 1e6))
        os.remove('tmp.pt')

    print_model_size(model)
    backend = 'fbgemm'
    model.qconfig = torch.quantization.get_default_qconfig(backend)
    for name, module in model.named_modules():
        if 'transpconvs' in name:
            module.qconfig = None
    # print(model.qconfig)
    model_static_quantized = torch.quantization.prepare(model, inplace=False)
    model_static_quantized = torch.quantization.convert(model_static_quantized, inplace=False)
    print_model_size(model_static_quantized)
    predictor.network = model_static_quantized

    # variant 1: give input and output folders
    input_dir = join(nnUNet_raw, 'Dataset009_Spleen/imagesTs')
    output_dir = join(nnUNet_raw, 'Dataset009_Spleen/imagesTs_predlowres')

    def call_predict():

        predictor.predict_from_files([[join(input_dir, 'spleen_1_0000.nii.gz')]],
                                     [join(input_dir, 'spleen_1.nii.gz')],  #
                                     save_probabilities=False, overwrite=True,
                                     num_processes_preprocessing=2, num_processes_segmentation_export=2,
                                     folder_with_segs_from_prev_stage=None, num_parts=1, part_id=0)

    call_predict()

def new_dynamic_test(): #cant see speedup
    print(torch.cuda.is_available())

    # instantiate the nnUNetPredictor
    predictor = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=True,
        perform_everything_on_device=True,
        device=torch.device('cuda', 0),
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=True
    )
    # initializes the network architecture, loads the checkpoint
    predictor.initialize_from_trained_model_folder(
        join(nnUNet_results, 'Dataset009_Spleen/nnUNetTrainer__nnUNetPlans__3d_fullres'),
        use_folds=(0,),
        checkpoint_name='checkpoint_final.pth',
    )
    model: nn.Module = predictor.network
    import os

    input_dir = join(nnUNet_raw, 'Dataset009_Spleen/imagesTs')
    output_dir = join(nnUNet_raw, 'Dataset009_Spleen/imagesTs_predlowres')

    def print_model_size(mdl):
        torch.save(mdl.state_dict(), "tmp.pt")
        print("%.2f MB" % (os.path.getsize("tmp.pt") / 1e6))
        os.remove('tmp.pt')

    def call_predict(type=""):
        tic = time.perf_counter()

        files = ['spleen_1', 'spleen_7', 'spleen_11']
        input_files = [[join(input_dir, file + '_0000.nii.gz')] for file in files]
        output_files = [join(input_dir, file + '.nii.gz') for file in files]
        # [[join(input_dir, 'spleen_1_0000.nii.gz')]],
        # [join(input_dir, 'spleen_1.nii.gz')],
        predictor.predict_from_files(input_files,
                                     output_files,  #
                                     save_probabilities=False, overwrite=True,
                                     num_processes_preprocessing=4, num_processes_segmentation_export=4,
                                     folder_with_segs_from_prev_stage=None, num_parts=1, part_id=0)
        toc = time.perf_counter()
        print(f"{type}: Model finished in {toc - tic:0.4f} seconds")

    call_predict("normal")

    quant_modules.initialize()
    quant_desc_input = QuantDescriptor(calib_method='max')
    quant_nn.QuantConv2d.set_default_quant_desc_input(quant_desc_input)
    quant_nn.QuantLinear.set_default_quant_desc_input(quant_desc_input)

    for name, module in model.named_modules():
        if name.endswith('_quantizer'):
            module.enable_calib()
            module.disable_quant()  # Use full precision data to calibrate

    # variant 1: give input and output folders

    call_predict("calib")

    #
    for name, module in model.named_modules():
        if name.endswith('_quantizer'):
            module.load_calib_amax(strict=False)
            module.disable_calib()
            module.enable_quant()
    predictor.list_of_parameters = [model.state_dict()]
    model.cuda()
    call_predict("quant")
    # print_model_size(model)
if __name__ == '__main__':
    new_dynamic_test()