import torch
import mlflow
import hiddenlayer as HL

from model.collectdata_mdsA import collect_data
from model.alt_loss_A import Loss
from model.training import trainNet, select_gpu
from model.utilities import load_full_state, count_parameters, Params, save_to_mlflow

from model.autoencoder_models import UNet
from model.autoencoder_models import UNetPlusPlus

args = Params(
    batch_size=64,
    device = select_gpu(2),
    epochs=50,
    lr=4e-4,
    experiment_name='UNet', 
    asymmetry_parameter=2.5
)

train_loader = collect_data(
    '/share/lazy/sokoloff/ML-data_A/Aug14_80K_train.h5',
    '/share/lazy/sokoloff/ML-data_AA/Oct03_80K_train.h5',
    '/share/lazy/sokoloff/ML-data_AA/Oct03_40K_train.h5',
    '/share/lazy/will/ML_mdsA/June30_2020_80k_1.h5',
    '/share/lazy/will/ML_mdsA/June30_2020_80k_3.h5',
    '/share/lazy/will/ML_mdsA/June30_2020_80k_4.h5',
    '/share/lazy/will/ML_mdsA/June30_2020_80k_5.h5',
    '/share/lazy/will/ML_mdsA/June30_2020_80k_6.h5',
#     '/share/lazy/will/ML_mdsA/June30_2020_80k_7.h5',
#     '/share/lazy/will/ML_mdsA/June30_2020_80k_8.h5',
#     '/share/lazy/will/ML_mdsA/June30_2020_80k_9.h5',
    batch_size=args['batch_size'],
    masking=True,
    shuffle=False,
    load_XandXsq=False,
#     device = args['device'], 
    load_xy=False
)

val_loader = collect_data(
    '/share/lazy/sokoloff/ML-data_AA/Oct03_20K_val.h5',
    batch_size=args['batch_size'],
    slice=slice(256 * 39),
    masking=True, 
    shuffle=False,
    load_XandXsq=False,
    load_xy=False)


mlflow.tracking.set_tracking_uri('file:/share/lazy/pv-finder_model_repo')
mlflow.set_experiment(args['experiment_name'])

model = UNet().to(args['device'])
# Fuse Conv, bn and relu
model.fuse_model()

optimizer = torch.optim.Adam(model.parameters(), lr=args['lr'])
loss = Loss(epsilon=1e-5,coefficient=2.5)

model.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
torch.quantization.prepare_qat(model, inplace=True)

# load_full_state(model, '/share/lazy/pv-finder_model_repo/17/27d5279c4a0641ecb3807f400b25a9a8/artifacts/run_stats.pyt')

run_name = 'QAT u-net'

train_iter = enumerate(trainNet(model, optimizer, loss, train_loader, val_loader, args['epochs'], notebook=False))
with mlflow.start_run(run_name = run_name) as run:
    for epoch, result in train_iter:
        
        if epoch > 3:
            # Freeze quantizer parameters
            model.apply(torch.quantization.disable_observer)
        if epoch > 2:
            # Freeze batch norm mean and variance estimates
            model.apply(torch.nn.intrinsic.qat.freeze_bn_stats)
        
        save_to_mlflow({
            'Metric: Training loss':result.cost,
            'Metric: Validation loss':result.val,
            'Metric: Efficiency':result.eff_val.eff_rate,
            'Metric: False positive rate':result.eff_val.fp_rate,
            'Param: Asymmetry':args['asymmetry_parameter'],
            'Param: Epochs':args['epochs'],
        }, step=epoch)
