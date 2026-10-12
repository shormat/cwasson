import argparse
from Cwasson import Cwasson

def main():
    parser = argparse.ArgumentParser(description="Translate French to English with a trained Cwasson checkpoint.")
    parser.add_argument("checkpoint", help="Path to a .pth checkpoint saved during training")
    args = parser.parse_args()

    SRC_PATH = "shuffled_fr.txt"
    TGT_PATH = "shuffled_en.txt"
    SP_MODEL_PATH = "shuffled.model"
    MAX_LEN = 128
    NUM_EPOCHS = 10
    BATCH_SIZE = 32
    LR = 3e-4
    N_HEADS = 4
    D_MODEL = 256
    D_FF = 1024
    N_LAYERS = 3
    DROPOUT = 0.2
    SAVE_PATH = "checkpoints/"
    USE_WANDB = False
    USE_ACCELERATOR = False
    TRAIN_RATIO = 0.01
    VAL_RATIO = 0.98
    TEST_RATIO = 0.01

    trainer = Cwasson(
        src_path=SRC_PATH,
        tgt_path=TGT_PATH,
        sp_model_path=SP_MODEL_PATH,
        max_len=MAX_LEN,
        num_epochs=NUM_EPOCHS,
        batch_size=BATCH_SIZE,
        lr=LR,
        n_heads=N_HEADS,
        d_model=D_MODEL,
        d_ff=D_FF,
        n_layers=N_LAYERS,
        dropout=DROPOUT,
        save_path=SAVE_PATH,
        use_wandb=USE_WANDB,
        use_accelerator=USE_ACCELERATOR,
        train_ratio=TRAIN_RATIO,
        val_ratio=VAL_RATIO,
        test_ratio=TEST_RATIO,
    )

    trainer.load_model(args.checkpoint)

    text = "Bonjour, Comment ça va ?"
    translation = trainer.translate_text(text)
    print(f"Translation: {translation}")

    trainer.interactive_translation()


if __name__ == "__main__":
    main()
