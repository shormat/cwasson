import os
import torch
import torch.nn as nn
import torch.distributed as dist
import sacrebleu
from Tokenizer import Tokenizer
from Loader import DatasetLoader
from torch.utils.data import DataLoader
from Transformer import Transformer
from sacrebleu.metrics import BLEU
from datetime import datetime
from accelerate import Accelerator
from WandbLogger import WandbLogger

"""
READ ME :
    The code with plain pyTorch is not complete since we used the accelerate library to accelerate the training process 
    due to a lack of time.
    So the functions to read are the ones with the ACCELERATOR_ prefix.
"""

def gather_list(data_list):
        world_size = dist.get_world_size()
        gathered = [None] * world_size
        dist.all_gather_object(gathered, data_list)
        flat = []
        
        for part in gathered:
            flat.extend(part)
            
        return flat

class Cwasson():
    def __init__(self, 
                 src_path,
                 tgt_path, 
                 sp_model_path, 
                 max_len=128, 
                 num_epochs=5, 
                 batch_size=32, 
                 lr=3e-4, 
                 n_heads=8, 
                 d_model=512, 
                 d_ff=2048, 
                 n_layers=6, 
                 dropout=0.1, 
                 train_ratio=0.8, 
                 val_ratio=0.1, 
                 test_ratio=0.1, 
                 save_path="checkpoints/", 
                 use_wandb=False,
                 use_accelerator=False):

        # Constants
        self.SRC_PATH = src_path
        self.TGT_PATH = tgt_path
        self.SP_MODEL_PATH = sp_model_path
        self.MAX_LEN = max_len
        self.NUM_EPOCHS = num_epochs
        self.BATCH_SIZE = batch_size
        self.LR = lr
        self.N_HEADS = n_heads
        self.D_MODEL = d_model
        self.D_FF = d_ff
        self.N_LAYERS = n_layers
        self.DROPOUT = dropout
        self.TRAIN_RATIO = abs(train_ratio)
        self.VAL_RATIO = abs(val_ratio)
        self.TEST_RATIO = abs(test_ratio)
        self.SAVE_PATH = save_path
        self.USE_ACCELERATOR = use_accelerator

        if (int(self.TRAIN_RATIO + self.VAL_RATIO + self.TEST_RATIO)) != 1:
            raise ValueError("Train, validation, and test ratios must sum to 1.")
        
        #Saves
        os.makedirs(self.SAVE_PATH, exist_ok=True)
        os.makedirs(f"{self.SAVE_PATH}/train", exist_ok=True)
        os.makedirs(f"{self.SAVE_PATH}/val", exist_ok=True)
        os.makedirs(f"{self.SAVE_PATH}/epochs", exist_ok=True)
        os.makedirs(f"{self.SAVE_PATH}/test", exist_ok=True)

        #Data Loading
        self.tokenizer = Tokenizer(self.SP_MODEL_PATH, self.MAX_LEN)
        self.dataset = DatasetLoader(self.SRC_PATH, self.TGT_PATH, self.tokenizer)

        train_set, test_set, val_set = self.dataset.get_splits(self.TRAIN_RATIO, self.TEST_RATIO, self.VAL_RATIO)
        
        self.train_loader = DataLoader(train_set, 
                                       batch_size=self.BATCH_SIZE,
                                       shuffle=True,
                                       num_workers=4,
                                       prefetch_factor=2,
                                       pin_memory=True,
                                       persistent_workers=True)
        
        self.test_loader = DataLoader(test_set, 
                                     batch_size=self.BATCH_SIZE,
                                     shuffle=False,
                                     num_workers=4,
                                     prefetch_factor=2,
                                     pin_memory=True,
                                     persistent_workers=True)
        
        self.val_loader = DataLoader(val_set, 
                                     batch_size=self.BATCH_SIZE,
                                     shuffle=False,
                                     num_workers=4,
                                     prefetch_factor=2,
                                     pin_memory=True,
                                     persistent_workers=True)

        
        if self.USE_ACCELERATOR:
            self.accelerator = Accelerator()
            self.print = self.accelerator.print
            self.print(f"use :{self.accelerator.device} ")
            
            self.model = Transformer(self.tokenizer.voc_size,
                        self.tokenizer.pad_id,
                        d_model=self.D_MODEL, 
                        n_layers=self.N_LAYERS, 
                        n_heads=self.N_HEADS, 
                        d_ff=self.D_FF, 
                        max_len=self.MAX_LEN, 
                        dropout=self.DROPOUT)
            
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.LR)
            self.criterion = nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_id, label_smoothing=0.1)

            self.model, self.optimizer, self.train_loader, self.test_loader, self.val_loader = self.accelerator.prepare(self.model,
                                                                                                                        self.optimizer,
                                                                                                                        self.train_loader,
                                                                                                                        self.test_loader,
                                                                                                                        self.val_loader)

            if self.accelerator.is_main_process:
                print(f"Running on {self.accelerator.num_processes} GPUs")
                
        else:
            self.print = print
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model = Transformer(self.tokenizer.voc_size,
                                    self.tokenizer.pad_id,
                                    d_model=self.D_MODEL, 
                                    n_layers=self.N_LAYERS, 
                                    n_heads=self.N_HEADS, 
                                    d_ff=self.D_FF, 
                                    max_len=self.MAX_LEN, 
                                    dropout=self.DROPOUT).to(self.device)
            
            if torch.cuda.device_count() > 1:
                self.print(f"Using {torch.cuda.device_count()} GPUs!")
                self.model = nn.DataParallel(self.model)


            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.LR)
            
            self.criterion = nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_id)


        if not self.USE_ACCELERATOR or (self.USE_ACCELERATOR and self.accelerator.is_main_process):
            self.wandb_logger = WandbLogger(entity_name="les-patissiers", project_name="cwasson",
                                            config={
                                                "num_epochs": self.NUM_EPOCHS,
                                                "batch_size": self.BATCH_SIZE,
                                                "learning_rate": self.LR,
                                                "n_heads": self.N_HEADS,
                                                "d_model": self.D_MODEL,
                                                "d_ff": self.D_FF,
                                                "n_layers": self.N_LAYERS,
                                                "dropout": self.DROPOUT
                                            }, enabled=use_wandb, directory=self.SAVE_PATH)

        # for evaluations
        self.bleu = BLEU(effective_order=True)

    def run(self):
        if self.USE_ACCELERATOR:
            train = self.ACCELERATOR_train
            test = self.ACCELERATOR_test
        else:
            train = self.DP_train
            test = self.DP_test

        for epoch in range(1, self.NUM_EPOCHS + 1):
            avg_train_loss = train(epoch)
            avg_val_loss, blue_score = test(epoch)

            if not self.USE_ACCELERATOR or (self.USE_ACCELERATOR and self.accelerator.is_main_process):
                self.print(f"Epoch {epoch}/{self.NUM_EPOCHS} - Train Loss: {avg_train_loss:.4f} - Val Loss: {avg_val_loss:.4f} - BLEU: {blue_score:.4f}")
                self.save_model(epoch=epoch, step=0, save_epoch=True) 
                self.wandb_logger.log({
                    "train_loss": avg_train_loss,
                    "val_loss": avg_val_loss,
                    "bleu": blue_score,
                    "epoch": epoch
                })

    def ACCELERATOR_train(self, epoch):
        self.model.train()

        step = 0
        total_loss = 0.0
        best_loss = float('inf')
        for batch in self.train_loader:
            src, tgt = batch
            src = src.to(self.accelerator.device, non_blocking=True)
            tgt = tgt.to(self.accelerator.device, non_blocking=True)
            decoder_input = tgt[:, :-1]

            target_output = tgt[:, 1:]
          
            output = self.model(src, decoder_input)

            output_dim = output.shape[-1]
            output = output.contiguous().view(-1, output_dim)
            target_output = target_output.contiguous().view(-1)

            loss = self.criterion(output, target_output)    

            self.optimizer.zero_grad()
            self.accelerator.backward(loss)
            self.optimizer.step()

            if step % 500 == 0 and self.accelerator.is_main_process:
                self.print(f"Epoch {epoch} - Step {step}/{len(self.train_loader)} - Loss: {loss.item():.4f}")
                self.wandb_logger.log({
                    "train_loss": loss.item(),
                    "epoch": epoch,
                    "step": step
                })

            if loss.item() < best_loss and self.accelerator.is_main_process:
                best_loss = loss.item()
                self.save_model(epoch=epoch, step=step, is_train=True)

            step += 1
            total_loss += loss.item()

        return total_loss / len(self.train_loader)

    def DP_train(self, epoch):
        self.model.train()

        step = 0
        total_loss = 0.0
        best_loss = float('inf')
        for batch in self.train_loader:
            src, tgt = batch
            src = src.to(self.device, non_blocking=True)
            tgt = tgt.to(self.device, non_blocking=True)

            decoder_input = tgt[:, :-1]

            target_output = tgt[:, 1:]
        
            output = self.model(src, decoder_input)
            
            output_dim = output.shape[-1]
            output = output.contiguous().view(-1, output_dim)
            target_output = target_output.contiguous().view(-1)

            loss = self.criterion(output, target_output)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if step % 500 == 0:
                self.print(f"Epoch {epoch} - Step {step}/{len(self.train_loader)} - Loss: {loss.item():.4f}")
                self.wandb_logger.log({
                    "train_loss": loss.item(),
                    "epoch": epoch,
                    "step": step
                })

            # save the model
            if loss.item() < best_loss:
                best_loss = loss.item()
                self.save_model(epoch=epoch, step=step, is_train=True)

            step += 1
            total_loss += loss.item()

        return total_loss / len(self.train_loader)
    
    @torch.no_grad()
    def ACCELERATOR_test(self, epoch, max_eval_batches=None):
        accelerator = self.accelerator
        self.model.eval()

        total_loss = 0.0
        step = 0

        all_losses = []
        local_refs = []
        local_hyps = []


        for batch in self.test_loader:
            if max_eval_batches and step >= max_eval_batches:
                break

            src, tgt = batch
            src = src.to(accelerator.device, non_blocking=True)
            tgt = tgt.to(accelerator.device, non_blocking=True)

            decoder_input = tgt[:, :-1]
            target_output = tgt[:, 1:]

            output = self.model(src, decoder_input)

            loss = self.criterion(
                output.contiguous().view(-1, output.shape[-1]),
                target_output.contiguous().view(-1)
            )

            gathered_loss = accelerator.gather(loss.detach())
            all_losses.append(gathered_loss)

            for i in range(src.size(0)):
                ref_tokens = tgt[i].cpu().numpy().tolist()
                hyp = self.translate(src[i].cpu().numpy().tolist())
                if hyp is not None:
                    local_refs.append(self.tokenizer.sp.decode(ref_tokens))
                    local_hyps.append(hyp)

            if step % 500 == 0 and accelerator.is_main_process:
                src0 = src[0].cpu().numpy().tolist()
                tgt0 = tgt[0].cpu().numpy().tolist()
                self.print(f"Input:      {self.tokenizer.sp.decode(src0)}")
                self.print(f"Reference:  {self.tokenizer.sp.decode(tgt0)}")
                self.print(f"Hypothesis: {self.translate(src0)}")
                self.wandb_logger.log({
                    "test_loss": loss.item(),
                    "epoch": epoch,
                    "step": step
                })

            step += 1

        all_losses = torch.cat(all_losses)
        avg_loss = all_losses.mean().item()

        if accelerator.distributed_type != "NO":
            gathered_refs = gather_list(local_refs)
            gathered_hyps = gather_list(local_hyps)
        else:
            gathered_refs = local_refs
            gathered_hyps = local_hyps

        corpus_bleu = 0.0
        if accelerator.is_main_process and gathered_refs and gathered_hyps:
            corpus_bleu = sacrebleu.corpus_bleu(gathered_hyps, [gathered_refs]).score

        if accelerator.distributed_type != "NO":
            accelerator.wait_for_everyone()

        return avg_loss, corpus_bleu

    @torch.no_grad()
    def DP_test(self, epoch, max_eval_batches=None):
        self.model.eval()
        total_loss = 0.0
        step = 0

        refs = []
        hyps = []

        for batch in self.test_loader:
            if max_eval_batches and step >= max_eval_batches:
                break

            src, tgt = batch
            src = src.to(self.device, non_blocking=True)
            tgt = tgt.to(self.device, non_blocking=True)

            decoder_input = tgt[:, :-1]
            target_output = tgt[:, 1:]

            output = self.model(src, decoder_input)
            loss = self.criterion(output.contiguous().view(-1, output.shape[-1]),
                                  target_output.contiguous().view(-1))
            total_loss += loss.item()

            # Generate translations for BLEU calculation
            for i in range(src.size(0)):
                src_tokens = src[i].cpu().numpy().tolist()
                ref_tokens = tgt[i].cpu().numpy().tolist()
                
                ref = self.tokenizer.sp.decode(ref_tokens)
                refs.append(ref)
                
                hyp = self.translate(src_tokens)
                if hyp is not None: 
                    hyps.append(hyp)

            if step % 500 == 0:
                src_in = src[0].cpu().numpy().tolist()
                tgt_in = tgt[0].cpu().numpy().tolist()
                self.print(f"Input: {self.tokenizer.sp.decode(src_in)}")
                self.print(f"Reference: {self.tokenizer.sp.decode(tgt_in)}")
                self.print(f"Hypothesis: {self.translate(src_in)}")

                self.wandb_logger.log({
                    "test_loss": loss.item(),
                    "epoch": epoch,
                    "step": step
                })

            step += 1

        avg_loss = total_loss / step

        corpus_bleu = 0
        if refs and hyps:
            formatted_refs = [[ref] for ref in refs]
            corpus_bleu = self.bleu.corpus_score(hyps, formatted_refs).score

        return avg_loss, corpus_bleu

    def save_model(self, epoch=1, step=0, is_train=False, is_test=False, is_val=False, save_epoch=False):
        if self.USE_ACCELERATOR and not self.accelerator.is_main_process:
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if self.USE_ACCELERATOR:
            model_to_save = self.accelerator.unwrap_model(self.model)
            state_dict = model_to_save.state_dict()
        else:
            if isinstance(self.model, nn.DataParallel):
                state_dict = self.model.module.state_dict()
            else:
                state_dict = self.model.state_dict()

        if is_train:
            save_path = f"{self.SAVE_PATH}/train/model_epoch_{epoch}_step_{step}_{timestamp}.pth"
        elif is_test:
            save_path = f"{self.SAVE_PATH}/test/model_epoch_{epoch}_step_{step}_{timestamp}.pth"
        elif is_val:
            save_path = f"{self.SAVE_PATH}/val/model_epoch_{epoch}_step_{step}_{timestamp}.pth"
        elif save_epoch:
            save_path = f"{self.SAVE_PATH}/epochs/model_epoch_{epoch}_{timestamp}.pth"
        
        if self.USE_ACCELERATOR:
            self.accelerator.save(state_dict, save_path)
        else:
            torch.save(state_dict, save_path)


    def translate(self, src_tokens, max_len=128):
        if self.USE_ACCELERATOR:
            return self.ACCELERATOR_translate(src_tokens, max_len)
            
        self.model.eval()

        if len(src_tokens) > max_len - 2: 
            src_tokens = src_tokens[:max_len - 2]

        src = [self.tokenizer.bos_id] + src_tokens + [self.tokenizer.eos_id]
        
        if len(src) < max_len:
            src = src + [self.tokenizer.pad_id] * (max_len - len(src))
        
        src = torch.LongTensor(src).unsqueeze(0).to(self.device)

        tgt = torch.LongTensor([[self.tokenizer.bos_id]]).to(self.device)

        for _ in range(max_len):
            src_mask = (src != self.tokenizer.pad_id).unsqueeze(1).unsqueeze(2)
            tgt_mask = torch.ones(1, tgt.size(1), tgt.size(1)).bool().to(self.device)
            tgt_mask = tgt_mask & torch.tril(torch.ones(1, tgt.size(1), tgt.size(1))).bool().to(self.device)

            with torch.no_grad():
                output = self.model(src, tgt)

            next_token = output.argmax(2)[:, -1].unsqueeze(1)
            tgt = torch.cat([tgt, next_token], dim=1)

            if next_token.item() == self.tokenizer.eos_id:
                break

        output_tokens = tgt.squeeze().tolist()[1:]
        if output_tokens and output_tokens[-1] == self.tokenizer.eos_id:
            output_tokens = output_tokens[:-1]

        if not output_tokens:
            return "Translation error: No output generated"

        try:
            return self.tokenizer.sp.decode(output_tokens)
        except Exception as e:
            return f"Decoding error: {str(e)}, tokens: {output_tokens}"

    @torch.no_grad()
    def ACCELERATOR_translate(self, src_tokens, max_len=128):
        """Accelerated version of translate that properly handles device placement and process management."""
        self.model.eval()

        if len(src_tokens) > max_len - 2:  # -2 for BOS and EOS tokens
            src_tokens = src_tokens[:max_len - 2]

        src = [self.tokenizer.bos_id] + src_tokens + [self.tokenizer.eos_id]
        
        # Pad to max_len like in training
        if len(src) < max_len:
            src = src + [self.tokenizer.pad_id] * (max_len - len(src))
        
        src = torch.LongTensor(src).unsqueeze(0).to(self.accelerator.device)
        tgt = torch.LongTensor([[self.tokenizer.bos_id]]).to(self.accelerator.device)

        for _ in range(max_len):
            src_mask = (src != self.tokenizer.pad_id).unsqueeze(1).unsqueeze(2)
            tgt_mask = torch.ones(1, tgt.size(1), tgt.size(1)).bool().to(self.accelerator.device)
            tgt_mask = tgt_mask & torch.tril(torch.ones(1, tgt.size(1), tgt.size(1))).bool().to(self.accelerator.device)

            with torch.no_grad():
                output = self.model(src, tgt)

            next_token = output.argmax(2)[:, -1].unsqueeze(1)
            tgt = torch.cat([tgt, next_token], dim=1)

            if next_token.item() == self.tokenizer.eos_id:
                break

        output_tokens = tgt.squeeze().tolist()[1:]
        if output_tokens and output_tokens[-1] == self.tokenizer.eos_id:
            output_tokens = output_tokens[:-1]

        if not output_tokens:
            return "Translation error: No output generated"

        try:
            # Only decode and print on main process
            if self.accelerator.is_main_process:
                return self.tokenizer.sp.decode(output_tokens)
            return None
        except Exception as e:
            if self.accelerator.is_main_process:
                return f"Decoding error: {str(e)}, tokens: {output_tokens}"
            return None

    def load_model(self, model_path):
        """Load a saved model from the given path."""
        if self.USE_ACCELERATOR:
            # For accelerator, we need to unwrap the model first
            model_to_load = self.accelerator.unwrap_model(self.model)
            state_dict = torch.load(model_path, map_location=self.accelerator.device)
            model_to_load.load_state_dict(state_dict)
        else:
            state_dict = torch.load(model_path, map_location=self.device)
            if isinstance(self.model, nn.DataParallel):
                self.model.module.load_state_dict(state_dict)
            else:
                self.model.load_state_dict(state_dict)
        
        self.model.eval()
        self.print(f"Model loaded successfully from {model_path}")

    def translate_text(self, text, max_len=128):
        """Translate a given text using the loaded model."""
        src_tokens = self.tokenizer.sp.encode(text)
        
        if len(src_tokens) > max_len - 2:  # -2 for BOS and EOS tokens
            src_tokens = src_tokens[:max_len - 2]
        
        translation = self.translate(src_tokens, max_len=max_len)
        
        return translation

    def interactive_translation(self):
        """Start an interactive translation session."""
        self.print("\nStarting interactive translation session...")
        self.print("Type 'quit' or 'exit' to end the session")
        self.print("Type your text to translate:\n")
        
        while True:
            try:
                text = input("> ").strip()
                if text.lower() in ['quit', 'exit']:
                    break
                    
                if not text:
                    continue
                    
                translation = self.translate_text(text)
                self.print(f"\nTranslation: {translation}\n")
                
            except KeyboardInterrupt:
                self.print("\nExiting translation session...")
                break
            except Exception as e:
                self.print(f"\nError during translation: {str(e)}\n")

