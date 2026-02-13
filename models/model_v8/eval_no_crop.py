"""Runs model_v8 evaluation without the mandatory center crop."""

from eval import EvalConfig, main


def main_no_crop() -> None:
    cfg = EvalConfig(
        center_crop=False,
        plot_path='eval_predictions_v8_no_crop.png',
        errors_csv='eval_predictions_v8_no_crop.csv',
    )
    print('[eval_no_crop] center_crop disabled for this run')
    main(cfg)


if __name__ == '__main__':
    main_no_crop()
