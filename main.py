import framework
import tasks


def register_args(parser: framework.helpers.ArgumentParser):
    parser.add_argument("-batch_size", default=128)
    parser.add_argument("-lr", default=1e-3)
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
    parser.add_argument("-scan.analyze_splits", default="jump,turn_left,length", parser=parser.str_list_parser)
    parser.add_argument("-scale_mask_loss", default=False)
    parser.add_argument("-mask_init", default=2.0)
    parser.add_argument("-layer_sizes", default="800,800,256", parser=parser.int_list_parser)
    parser.add_argument("-class_removal.keep_last_layer", default=False)
    parser.add_argument("-encoder_decoder.embedding_size", default=16)
    parser.add_argument("-transfer.mask_new_init", default=2.0)
    parser.add_argument("-transfer.mask_used_init", default=2.0)


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
    }), parser.Profile("tuple", {
        "task": "tuple",
        "n_digits": 2,
        "mask_lr": 0.01,
        "mask_loss_weight": 0.0001,
        "state_size": 256,
        "step_per_mask": 20000,
        "stop_after": 20000
    })])


def main():
    helper = framework.helpers.TrainingHelper(wandb_project_name="modules",
                                                   register_args=register_args, extra_dirs=["export"])

    def invalid_task_error(self):
        assert False, f"Invalid task: {helper.opt.task}"

    constructors = {
        "tuple": tasks.TupleTask,
        "tuple_ff": tasks.TupleTaskFeedforward,
        "scan": tasks.ScanTask,
        "addmul_ff": tasks.AddMulFeedforward,
        "addmul": tasks.AddMulTask,
        "cifar10_class_removal": tasks.Cifar10ClassRemovalTask,
        "permuted_mnist": tasks.PermutedMnistTask
    }

    task = constructors.get(helper.opt.task, invalid_task_error)(helper)

    task.train()
    task.post_train()

    helper.finish()


if __name__ == "__main__":
    main()