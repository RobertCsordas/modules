import framework
import tasks
import os
import torch
# torch.backends.cudnn.enabled = False

def register_args(parser: framework.helpers.ArgumentParser):
    parser.add_argument("-batch_size", default=128)
    parser.add_argument("-lr", default=1e-3)
    parser.add_argument("-wd", default=0.0)
    parser.add_argument("-lr_warmup", default=0)
    parser.add_argument("-mask_lr", default="1e-2", parser=parser.float_or_none_parser)
    parser.add_argument("-test_interval", default=1000)
    parser.add_argument("-n_digits", default=1)
    parser.add_argument("-state_size", default=128)
    parser.add_argument("-n_layers", default=2)
    parser.add_argument("-stop_after", default="None", parser=parser.int_or_none_parser)
    parser.add_argument("-task", default="tuple")
    parser.add_argument("-mask_loss_weight", default=1e-4)
    parser.add_argument("-step_per_mask", default=10000)
    parser.add_argument("-bias_no_mask", default=False)
    parser.add_argument("-dropout", default=0.0)
    parser.add_argument("-tuple.mode", default="together", choice=["together", "only_one", "same_output",
                                                                   "same_input", "same_io", "only_one_io"])
    parser.add_argument("-tuple.tuple2_delay", default=0)
    parser.add_argument("-grad_clip", default="1.0", parser=parser.float_or_none_parser)
    parser.add_argument("-scan.analyze_splits", default="jump,length,turn_left", parser=parser.str_list_parser)
    parser.add_argument("-scan.train_split", default="simple", parser=parser.str_list_parser)
    parser.add_argument("-scale_mask_loss", default=False)
    parser.add_argument("-mask_init", default=2.0)
    parser.add_argument("-analysis.enable", default=True)
    parser.add_argument("-analysis.plot_masks", default=True)
    parser.add_argument("-analysis.only_verify_masks", default=False)
    parser.add_argument("-analysis.skip_layernorm", default=False)
    parser.add_argument("-layer_sizes", default="800,800,256", parser=parser.int_list_parser)
    parser.add_argument("-class_removal.keep_last_layer", default=False)
    parser.add_argument("-encoder_decoder.embedding_size", default=16)
    parser.add_argument("-transfer.mask_new_init", default=2.0)
    parser.add_argument("-transfer.mask_used_init", default=2.0)
    parser.add_argument("-transfer.reset_first_layer", default=True)
    parser.add_argument("-transformer.n_heads", default=4)
    parser.add_argument("-transformer.use_paper_lr_schedule", default=False)
    parser.add_argument("-transformer.ff_multiplier", default=2.0)
    parser.add_argument("-transformer.encoder_n_layers", default=3)
    parser.add_argument("-transformer.decoder_n_layers", default=3)
    parser.add_argument("-cnn.dropout", default=True)
    parser.add_argument("-test_batch_size", default="None", parser=parser.int_or_none_parser)
    parser.add_argument("-mask_batch_size", default="None", parser=parser.int_or_none_parser)
    parser.add_argument("-dm_math.tasks", default="algebra__linear_1d", parser=parser.str_list_parser)
    parser.add_argument("-dm_math.train_splits", default="easy,medium,hard",
                        parser=parser.str_list_parser)
    parser.add_argument("-dm_math.masks_splits", default="easy", parser=parser.str_list_parser)
    parser.add_argument("-mask_stability.measure_on", default="minimal", choice=["minimal", "all"])
    parser.add_argument("-masking.warmup.nsteps", default=0)
    parser.add_argument("-inv_mask_exclude_io", default="none", choice=["none", "fullmask", "ones"])
    parser.add_argument("-restore_pretrained", type=str)
    parser.add_argument("-test_pretrained", default=1)
    parser.add_argument("-train_baseline", default=False, help="Train the model on easy task and test on hard,"
                                                               "no masking")
    parser.add_argument("-lr_sched.steps", default="", parser=parser.int_list_parser)
    parser.add_argument("-lr_sched.gamma", default=0.1)
    parser.add_argument("-optimizer", default="adam", choice=["adam", "sgd"])


    parser.add_profile([parser.Profile("scan", {
        "task": "scan",
        "n_layers": 2,
        "state_size": 200,
        "lr": 1e-3,
        "grad_clip": "5",
        "stop_after": 15000,
        "step_per_mask": 15000,
        "bias_no_mask": 1,
        "batch_size": 256,
        "mask_loss_weight": 3e-5,
        "dropout": 0.5
    }),

    parser.Profile("add_mul", {
        "task": "addmul",
        "stop_after": 20000,
        "n_layers": 2,
        "state_size": 256,
        "n_digits": 2,
        "mask_loss_weight": 0.0001,
        "step_per_mask": 20000,
        "bias_no_mask": 1
    }),

    parser.Profile("trafo_scan", {
        "task": "trafo_scan",
        "state_size": 100,
        "mask_loss_weight": 0.0006,
        "test_batch_size": 2048
    }, include="scan"),

    parser.Profile("tuple", {
        "task": "tuple",
        "n_digits": 2,
        "mask_lr": 0.01,
        "mask_loss_weight": 0.0001,
        "state_size": 256,
        "step_per_mask": 20000,
        "stop_after": 20000
    }),

    parser.Profile("cifar10_class_removal", {
        "task": "cifar10_class_removal",
        "stop_after": 20000,
        "mask_loss_weight": 3e-4,
        "mask_lr": 1e-3,
        "step_per_mask": 20000,
        "class_removal.keep_last_layer": 1
    }),

    parser.Profile("cifar10_resnet_hp", {
        "task": "cifar10_resnet_hp_class_removal",
        "wd": 1e-4,
        "batch_size": 128,
        "optimizer": "sgd",
        "lr": 0.1,
        "lr_sched.steps": "32000,48000",
        "stop_after": 64000,
        "mask_loss_weight": 0.00002,
        "mask_lr": 0.03,
        "mask_batch_size": 256,
        "step_per_mask": 30000,
        "class_removal.keep_last_layer": 1
    }),

    parser.Profile("deepmind_math", {
        "task": "deepmind_math",
        "lr": 1e-4,
        "stop_after": 50000,
        "step_per_mask": 20000,
        "bias_no_mask": 1,
        "batch_size": 256,
        "mask_loss_weight": 0.001,
        "state_size": 512,
        "transformer.n_heads": 8,
        "transformer.ff_multiplier": 4,
        "transformer.encoder_n_layers": 6,
        "transformer.decoder_n_layers": 6,
        "test_batch_size": 1024,
        "grad_clip": 0.1
    })])


def save_weights(helper: framework.helpers.TrainingHelper, task: tasks.Task):
    data = {
        "VERSION": 1,
        "parameters": task.model.model_parameters.state_dict(),
        "state": task.model.model.state_dict()
    }

    torch.save(data, os.path.join(helper.dirs.model_weights, "model.pth"))


def load_weights(helper: framework.helpers.TrainingHelper, task: tasks.Task):
    pretrained = os.path.join(os.path.expanduser(helper.opt.restore_pretrained),
                              str(helper.opt.sweep_id_for_grid_search), "model.pth")
    assert os.path.isfile(pretrained), f"Failed to load pretrained weights. File {pretrained} not found."
    print(f"Loading pretrained weights from {pretrained}...")

    data = torch.load(pretrained)
    ver = data.get("VERSION", 0)
    if ver == 0:
        task.model.model_parameters.load_state_dict(data)
    else:
        task.model.model.load_state_dict(data["state"])
        task.model.model_parameters.load_state_dict(data["parameters"])


def main():
    helper = framework.helpers.TrainingHelper(wandb_project_name="modules",
                                                   register_args=register_args, extra_dirs=["export", "model_weights"])

    def invalid_task_error(self):
        assert False, f"Invalid task: {helper.opt.task}"

    constructors = {
        "tuple": tasks.TupleTask,
        "tuple_ff": tasks.TupleTaskFeedforward,
        "scan": tasks.ScanTask,
        "trafo_scan": tasks.TransformerScanTask,
        "addmul_ff": tasks.AddMulFeedforward,
        "addmul": tasks.AddMulTask,
        "cifar10_class_removal": tasks.Cifar10ClassRemovalTask,
        "cifar10_resnet_hp_class_removal": tasks.Cifar10ResnetHPClassRemovalTask,
        "cifar10_mask_stability": tasks.Cifar10MaskStabilityTask,
        "cifar10_grad_cos_distance": tasks.Cifar10GradCosDistanceTask,
        "permuted_mnist": tasks.PermutedMnistTask,
        "deepmind_math": tasks.DeepmindMathTask,
        "tuple_ff_copyweight": tasks.TupleFeedforwardCopyweightTask
    }

    task = constructors.get(helper.opt.task, invalid_task_error)(helper)
    if helper.opt.restore_pretrained:
        assert not task.helper.opt.train_baseline
        load_weights(helper, task)
        if helper.opt.test_pretrained:
            helper.summary.log({f"load_validation/{k}": v for k, v in task.validate().items()})
        print("Done. Skipping training...")
    else:
        if task.helper.opt.train_baseline:
            task.set_baseline_mode()

        task.train()

        print("Training finished. Saving model...")
        save_weights(helper, task)

    if task.helper.opt.analysis.enable and not task.helper.opt.train_baseline:
        task.post_train()

    helper.finish()


if __name__ == "__main__":
    main()