from src.models.arg_parser import parse_args

from src.ssl.trainer import SSLTrainer


def main():
    hparams = parse_args()
    ssl_trainer = SSLTrainer(hparams)
    ssl_trainer.fit_ssl()


if __name__ == "__main__":
    main()
