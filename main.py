import torch.multiprocessing as mp

def main():
    import torch
    from Cwasson import Cwasson
    
    SRC_PATH = "shuffled_fr.txt"
    TGT_PATH = "shuffled_en.txt"
    SP_MODEL_PATH = "shuffled.model"
    MAX_LEN = 128
    NUM_EPOCHS = 20
    BATCH_SIZE = 32
    LR = 3e-4
    N_HEADS = 4
    D_MODEL = 256
    D_FF = 1024
    N_LAYERS = 3
    DROPOUT = 0.2
    SAVE_PATH = "checkpoints/"
    USE_WANDB = False
    TRAIN_RATIO = 0.90
    VAL_RATIO = 0.05
    TEST_RATIO = 0.05
    USE_ACCELERATOR = True
    
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

    trainer.run()


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
