import os, traceback, shutil
import numpy as np
import torch
import torch.nn as nn
from typing import Union, Optional
from torch.nn.parallel import DistributedDataParallel
from huggingface_hub import PyTorchModelHubMixin
from ..utils import human_format


class TrainerCallback:
    def on_epoch_start(self, model: nn.Module, epoch: int) -> None:
        pass

    def on_epoch_end(self, model: nn.Module, epoch: int) -> Union[bool, None]:
        pass

    def on_batch_start(self, model: nn.Module, batch_idx: int, batch: dict[str, torch.Tensor]) -> None:
        pass

    def on_batch_end(self, model: nn.Module, batch_idx: int, loss: float, batch: dict[str, torch.Tensor]) -> \
            Union[
                bool, None]:
        pass

    def on_training_end(self, model: nn.Module) -> None:
        pass

    def on_validation_end(self, model: nn.Module, epoch: int, val_loss: float, val_metrics: dict) -> Union[
        bool, None]:
        pass


class PrintLossCallback(TrainerCallback):
    def __init__(self, batch_log_interval: int = 100, joint_mode: bool = False, batches_per_epoch: int = None):
        self.epoch_means = []
        self.epoch_losses = []
        self.batch_group_losses = []
        self.batch_log_interval = batch_log_interval
        self.joint_mode = joint_mode
        self.batches_per_epoch = batches_per_epoch

    def on_batch_start(self, model: nn.Module, batch_idx: int, batch: dict[str, torch.Tensor]) -> None:
        pass

    def on_batch_end(self, model: nn.Module, batch_idx: int, loss: int,
                     batch: dict[str, torch.Tensor]) -> None:
        self.batch_group_losses.append(loss)
        self.epoch_losses.append(loss)

        if batch_idx != 0 and batch_idx % self.batch_log_interval == 0:
            batch_group_mean = np.stack(self.batch_group_losses).mean()
            self.batch_group_losses = []
            if self.batches_per_epoch is not None:
                print(
                    f'Batch {batch_idx} / {self.batches_per_epoch} - loss: {loss}, last {self.batch_log_interval} batches mean loss: {batch_group_mean:.4f}')
            else:
                print(
                    f'Batch {batch_idx} - loss: {loss}, last {self.batch_log_interval} batches mean loss: {batch_group_mean:.4f}')

    def on_epoch_start(self, model: nn.Module, epoch: int) -> None:
        self.epoch_losses = []
        print(f'Start epoch: {epoch}')

    def on_epoch_end(self, model: nn.Module, epoch: int) -> None:
        epoch_mean = np.stack(self.epoch_losses).mean()
        print(f'Epoch {epoch} - mean loss: {epoch_mean:.4f}')
        self.epoch_means.append(epoch_mean)

    def on_training_end(self, model: nn.Module) -> None:
        print(f'Finished training! All losses:')
        print(self.epoch_means)

    def on_validation_end(self, model: nn.Module, epoch: int, val_loss: float, val_metrics: dict) -> None:
        if self.joint_mode:
            print(f"Epoch {epoch} - encoder loss: {val_metrics['loss']['encoder']:.4f}")
            print(f"Epoch {epoch} - decoder loss: {val_metrics['loss']['decoder']:.4f}")
        print(f"Epoch {epoch} - validation Loss: {val_loss:.4f}")


class PrintAccuracyCallback(TrainerCallback):
    def __init__(self, joint_mode: bool = False):
        self.joint_mode = joint_mode

    def on_validation_end(self, model: nn.Module, epoch: int, val_loss: float, val_metrics: dict) -> None:
        if self.joint_mode:
            print(f"Epoch {epoch} - encoder node accuracy: {val_metrics['accuracy']['node_encoder']:.4f}")
            print(f"Epoch {epoch} - decoder node accuracy: {val_metrics['accuracy']['node_decoder']:.4f}")
            print(f"Epoch {epoch} - encoder accuracy: {val_metrics['accuracy']['encoder']:.4f}")
            print(f"Epoch {epoch} - decoder accuracy: {val_metrics['accuracy']['decoder']:.4f}")
        else:
            print(f"Epoch {epoch} - node accuracy: {val_metrics['node_accuracy']:.4f}")
            print(f"Epoch {epoch} - accuracy: {val_metrics['accuracy']:.4f}")


class TokenCounterCallback(TrainerCallback):
    def __init__(self, limit: int, batch_log_interval: int = 100):
        self.total_tokens = 0
        self.limit = limit
        self.batch_log_interval = batch_log_interval

    def on_batch_end(self, model: nn.Module, batch_idx: int, loss: int,
                     batch: dict[str, torch.Tensor]) -> bool:
        attention_mask = batch['attention_mask']
        batch_tokens = attention_mask.sum().item()
        self.total_tokens += batch_tokens
        if batch_idx != 0 and batch_idx % self.batch_log_interval == 0:
            print(f'Total processed tokens: {human_format(self.total_tokens)}')

        should_stop_training = self.total_tokens >= self.limit
        if should_stop_training:
            print(f'Reached a limit of {human_format(self.limit)} processed tokens - stopping training')
        return should_stop_training

    def on_training_end(self, model: nn.Module) -> None:
        print(f'Total training tokens: {human_format(self.total_tokens)}')

    def get_total_tokens(self):
        return self.total_tokens


class ModelSaveCallback(TrainerCallback):
    def __init__(
            self,
            save_dir: str,
            max_keep: int = 3,
            push_to_hub: bool = False,
            hub_model_id: str = None,
            private_repo: bool = False,
            hf_token: str = None,
            push_checkpoint_weights: bool = True,
            final_commit_message: str = None,
            save_checkpoint_after_n_batches: int = None,
            push_batch_checkpoint: bool = False,
            display_exc_trace: bool = False,
            use_ddp: bool = False,
    ):
        self.save_dir = save_dir
        self.max_keep = max_keep
        self.best_loss = float('inf')
        self.ckpt_paths = []
        self.push_to_hub = push_to_hub
        self.hub_model_id = hub_model_id
        self.private_repo = private_repo
        self.hf_token = hf_token
        self.push_checkpoint_weights = push_checkpoint_weights
        self.final_commit_message = final_commit_message
        self.save_checkpoint_after_n_batches = save_checkpoint_after_n_batches
        self.push_batch_checkpoint = push_batch_checkpoint
        self.finished_epochs = 0
        self.display_exc_trace = display_exc_trace
        self.rank = int(os.environ['RANK']) if use_ddp else 0

    def on_batch_end(self, model: nn.Module, batch_idx: int, loss: int, batch: dict[str, torch.Tensor]) -> Union[
        bool, None]:
        if self.rank == 0 and self.save_checkpoint_after_n_batches is not None and batch_idx != 0 and batch_idx % self.save_checkpoint_after_n_batches == 0:
            if isinstance(model, DistributedDataParallel):
                model = next(model.children())
            try:
                if model.save_pretrained is not None:
                    ckpt_path = os.path.join(
                        self.save_dir,
                        'batch_checkpoint'
                    )
                    path_exists = os.path.exists(ckpt_path)
                    if not path_exists:
                        os.makedirs(ckpt_path)
                    model.save_pretrained(save_directory=ckpt_path)
                else:
                    path_exists = os.path.exists(self.save_dir)
                    if not path_exists:
                        os.makedirs(self.save_dir)
                    ckpt_path = os.path.join(
                        self.save_dir,
                        'batch_checkpoint.pt'
                    )
                    os.remove(ckpt_path)
                    torch.save(model.state_dict(), ckpt_path)
            except Exception as e:
                print(f"Error saving batch checkpoint: {str(e)}")
                if self.display_exc_trace:
                    traceback.print_exc()
            try:
                if self.push_to_hub and self.push_batch_checkpoint and model.push_to_hub is not None and self.hub_model_id:
                    model.push_to_hub(
                        repo_id=self.hub_model_id,
                        token=self.hf_token,
                        private=self.private_repo,
                    )
            except Exception as e:
                print(f"Error pushing batch checkpoint: {str(e)}")
                if self.display_exc_trace:
                    traceback.print_exc()

    def on_validation_end(
            self,
            model: Union[nn.Module, PyTorchModelHubMixin],
            epoch: int,
            val_loss: float,
            val_metrics: dict
    ):
        if self.rank == 0:
            self.finished_epochs += 1
            if val_loss < self.best_loss:
                self.best_loss = val_loss
                if isinstance(model, DistributedDataParallel):
                    model = next(model.children())
                try:
                    if model.save_pretrained is not None:
                        ckpt_path = os.path.join(
                            self.save_dir,
                            f'epoch_{epoch}_val_loss_{val_loss:.4f}'
                        )
                        path_exists = os.path.exists(ckpt_path)
                        if not path_exists:
                            os.makedirs(ckpt_path)
                        model.save_pretrained(save_directory=ckpt_path)
                    else:
                        path_exists = os.path.exists(self.save_dir)
                        if not path_exists:
                            os.makedirs(self.save_dir)
                        ckpt_path = os.path.join(
                            self.save_dir,
                            f'epoch_{epoch}_val_loss_{val_loss:.4f}.pt'
                        )
                        torch.save(model.state_dict(), ckpt_path)
                    self.ckpt_paths.append(ckpt_path)

                    # Keep only N best checkpoints
                    if len(self.ckpt_paths) > self.max_keep:
                        oldest_path = self.ckpt_paths.pop(0)
                        if model.save_pretrained is not None:
                            shutil.rmtree(oldest_path)
                        else:
                            os.remove(oldest_path)
                except Exception as e:
                    print(f"Error saving epoch checkpoint: {str(e)}")
                    if self.display_exc_trace:
                        traceback.print_exc()

                try:
                    if self.push_to_hub and self.push_checkpoint_weights and model.push_to_hub is not None and self.hub_model_id:
                        model.push_to_hub(
                            repo_id=self.hub_model_id,
                            commit_message=f'Epoch {epoch} - Val loss {val_loss:.4f}',
                            token=self.hf_token,
                            private=self.private_repo,
                        )
                except Exception as e:
                    print(f"Error pushing epoch checkpoint: {str(e)}")
                    if self.display_exc_trace:
                        traceback.print_exc()

    def on_training_end(self, model: Union[nn.Module, PyTorchModelHubMixin]):
        if self.rank == 0:
            if isinstance(model, DistributedDataParallel):
                model = next(model.children())
            try:
                # Save final model
                if model.save_pretrained is not None:
                    ckpt_path = os.path.join(
                        self.save_dir,
                        'final_model'
                    )
                    model.save_pretrained(save_directory=ckpt_path)
                else:
                    ckpt_path = os.path.join(self.save_dir, 'final_model.pt')
                    torch.save(model.state_dict(), ckpt_path)
                print(f"Final model saved to {ckpt_path}")
            except Exception as e:
                print(f"Error saving final model: {str(e)}")
                if self.display_exc_trace:
                    traceback.print_exc()
            try:
                if self.push_to_hub and model.push_to_hub is not None:
                    model.push_to_hub(
                        repo_id=self.hub_model_id,
                        commit_message=self.final_commit_message or f'Final pre-trained model, after {self.finished_epochs} epochs',
                        token=self.hf_token,
                        private=self.private_repo,
                    )
                print(f"Model uploaded to repo: {self.hub_model_id}")
            except Exception as e:
                print(f"Error pushing final model: {str(e)}")
                if self.display_exc_trace:
                    traceback.print_exc()


class JointModelSaveCallback(TrainerCallback):
    def __init__(
            self,
            save_dir: str,
            max_keep: int = 3,
            push_to_hub: bool = False,
            hub_model_decoder: str = None,
            hub_model_encoder: str = None,
            hub_model_head: str = None,
            private_repo: bool = False,
            hf_token: str = None,
            push_checkpoint_weights: bool = True,
            final_commit_message: str = None,
            save_checkpoint_after_n_batches: int = None,
            push_batch_checkpoint: bool = False,
            mlm_mode: bool = False,
            display_exc_trace: bool = False,
            use_ddp: bool = False,
    ):
        self.save_dir = save_dir
        self.max_keep = max_keep
        self.best_loss = float('inf')
        self.ckpt_paths = []
        self.push_to_hub = push_to_hub
        self.hub_model_decoder = hub_model_decoder
        self.hub_model_encoder = hub_model_encoder
        self.hub_model_head = hub_model_head
        self.private_repo = private_repo
        self.hf_token = hf_token
        self.push_checkpoint_weights = push_checkpoint_weights
        self.final_commit_message = final_commit_message
        self.save_checkpoint_after_n_batches = save_checkpoint_after_n_batches
        self.push_batch_checkpoint = push_batch_checkpoint
        self.finished_epochs = 0
        self.mlm_mode = mlm_mode
        self.display_exc_trace = display_exc_trace
        self.rank = int(os.environ['RANK']) if use_ddp else 0

    def _save_batch(self, model: Union[nn.Module, PyTorchModelHubMixin], component: str, hub_id: str = None):
        try:
            if model.save_pretrained is not None:
                ckpt_path = os.path.join(
                    self.save_dir,
                    component,
                    'batch_checkpoint'
                )
                path_exists = os.path.exists(ckpt_path)
                if not path_exists:
                    os.makedirs(ckpt_path)
                model.save_pretrained(save_directory=ckpt_path)
            else:
                comp_path = os.path.join(
                    self.save_dir,
                    component
                )
                path_exists = os.path.exists(comp_path)
                if not path_exists:
                    os.makedirs(comp_path)
                ckpt_path = os.path.join(
                    comp_path,
                    'batch_checkpoint.pt'
                )
                os.remove(ckpt_path)
                torch.save(model.state_dict(), ckpt_path)
        except Exception as e:
            print(f"Error saving batch checkpoint: {str(e)}")
            if self.display_exc_trace:
                traceback.print_exc()
        try:
            if self.push_to_hub and self.push_batch_checkpoint and model.push_to_hub is not None and hub_id:
                model.push_to_hub(
                    repo_id=hub_id,
                    token=self.hf_token,
                    private=self.private_repo,
                )
        except Exception as e:
            print(f"Error pushing batch checkpoint: {str(e)}")
            if self.display_exc_trace:
                traceback.print_exc()

    def on_batch_end(self, model: nn.Module, batch_idx: int, loss: int, batch: dict[str, torch.Tensor]) -> Union[
        bool, None]:
        if self.rank == 0 and self.save_checkpoint_after_n_batches is not None and batch_idx != 0 and batch_idx % self.save_checkpoint_after_n_batches == 0:
            if isinstance(model, DistributedDataParallel):
                model = next(model.children())

            if not self.mlm_mode:
                # Set single memory item for saving
                batch_size = model.decoder.model.stm.batch_size
                model.decoder.model.stm.single_memory()

                self._save_batch(model.encoder, 'encoder', hub_id=self.hub_model_encoder)
                self._save_batch(model.decoder, 'decoder', hub_id=self.hub_model_decoder)
                self._save_batch(model.mlm_head, 'head', hub_id=self.hub_model_head)
                # Set back batched memory
                model.decoder.model.stm.batched_memory(batch_size)
            else:
                self._save_batch(model.encoder, 'encoder', hub_id=self.hub_model_encoder)
                self._save_batch(model.mlm_head, 'head', hub_id=self.hub_model_head)


    def _save_validation(self, model: Union[nn.Module, PyTorchModelHubMixin], component: str, epoch: int,
                         val_loss: float, hub_id: str = None):
        try:
            if model.save_pretrained is not None:
                ckpt_path = os.path.join(
                    self.save_dir,
                    component,
                    f'epoch_{epoch}_val_loss_{val_loss:.4f}'
                )
                path_exists = os.path.exists(ckpt_path)
                if not path_exists:
                    os.makedirs(ckpt_path)
                model.save_pretrained(save_directory=ckpt_path)
            else:
                comp_path = os.path.join(
                    self.save_dir,
                    component
                )
                path_exists = os.path.exists(comp_path)
                if not path_exists:
                    os.makedirs(comp_path)
                ckpt_path = os.path.join(
                    comp_path,
                    f'epoch_{epoch}_val_loss_{val_loss:.4f}.pt'
                )
                torch.save(model.state_dict(), ckpt_path)
            self.ckpt_paths.append(ckpt_path)

            # Keep only N best checkpoints
            if len(self.ckpt_paths) > self.max_keep:
                oldest_path = self.ckpt_paths.pop(0)
                if model.save_pretrained is not None:
                    shutil.rmtree(oldest_path)
                else:
                    os.remove(oldest_path)
        except Exception as e:
            print(f"Error saving epoch checkpoint: {str(e)}")
            if self.display_exc_trace:
                traceback.print_exc()

        try:
            if self.push_to_hub and self.push_checkpoint_weights and model.push_to_hub is not None and hub_id:
                model.push_to_hub(
                    repo_id=hub_id,
                    commit_message=f'Epoch {epoch} - Val loss {val_loss:.4f}',
                    token=self.hf_token,
                    private=self.private_repo,
                )
        except Exception as e:
            print(f"Error pushing epoch checkpoint: {str(e)}")
            if self.display_exc_trace:
                traceback.print_exc()

    def on_validation_end(
            self,
            model: Union[nn.Module, PyTorchModelHubMixin],
            epoch: int,
            val_loss: float,
            val_metrics: dict
    ):
        if self.rank == 0:
            self.finished_epochs += 1
            if val_loss < self.best_loss:
                self.best_loss = val_loss
                if isinstance(model, DistributedDataParallel):
                    model = next(model.children())
                if not self.mlm_mode:
                    # Set single memory item for saving
                    batch_size = model.decoder.model.stm.batch_size
                    model.decoder.model.stm.single_memory()

                    self._save_validation(model.encoder, 'encoder', epoch, val_loss, hub_id=self.hub_model_encoder)
                    self._save_validation(model.decoder, 'decoder', epoch, val_loss, hub_id=self.hub_model_decoder)
                    self._save_validation(model.mlm_head, 'head', epoch, val_loss, hub_id=self.hub_model_head)
                    # Set back batched memory
                    model.decoder.model.stm.batched_memory(batch_size)
                else:
                    self._save_validation(model.encoder, 'encoder', epoch, val_loss, hub_id=self.hub_model_encoder)
                    self._save_validation(model.mlm_head, 'head', epoch, val_loss, hub_id=self.hub_model_head)

    def _save_final(self, model: Union[nn.Module, PyTorchModelHubMixin], component: str, hub_id: str = None):
        try:
            # Save final model
            if model.save_pretrained is not None:
                ckpt_path = os.path.join(
                    self.save_dir,
                    component,
                    'final_model'
                )
                path_exists = os.path.exists(ckpt_path)
                if not path_exists:
                    os.makedirs(ckpt_path)
                model.save_pretrained(save_directory=ckpt_path)
            else:
                comp_path = os.path.join(
                    self.save_dir,
                    component
                )
                path_exists = os.path.exists(comp_path)
                if not path_exists:
                    os.makedirs(comp_path)
                ckpt_path = os.path.join(comp_path, 'final_model.pt')
                torch.save(model.state_dict(), ckpt_path)
            print(f"Final model saved to {ckpt_path}")
        except Exception as e:
            print(f"Error saving final model: {str(e)}")
            if self.display_exc_trace:
                traceback.print_exc()
        try:
            if self.push_to_hub and model.push_to_hub is not None and hub_id:
                model.push_to_hub(
                    repo_id=hub_id,
                    commit_message=self.final_commit_message or f'Final pre-trained model, after {self.finished_epochs} epochs',
                    token=self.hf_token,
                    private=self.private_repo,
                )
        except Exception as e:
            print(f"Error pushing final model: {str(e)}")
            if self.display_exc_trace:
                traceback.print_exc()

    def on_training_end(self, model: Union[nn.Module, PyTorchModelHubMixin]):
        if self.rank == 0:
            if isinstance(model, DistributedDataParallel):
                model = next(model.children())
            if not self.mlm_mode:
                # Set single memory item for saving
                batch_size = model.decoder.model.stm.batch_size
                model.decoder.model.stm.single_memory()

                self._save_final(model.encoder, 'encoder', hub_id=self.hub_model_encoder)
                self._save_final(model.decoder, 'decoder', hub_id=self.hub_model_decoder)
                self._save_final(model.mlm_head, 'head', hub_id=self.hub_model_head)

                # Set back batched memory
                model.decoder.model.stm.batched_memory(batch_size)
            else:
                self._save_final(model.encoder, 'encoder', hub_id=self.hub_model_encoder)
                self._save_final(model.mlm_head, 'head', hub_id=self.hub_model_head)

class EarlyStoppageCallback(TrainerCallback):
    def __init__(self, num_plateau_epochs: int = 3) -> None:
        super().__init__()
        self.num_plateau_epochs = num_plateau_epochs
        self.best_loss = 9999.0
        self.best_loss_epoch = 0

    def on_validation_end(
            self,
            model: nn.Module,
            epoch: int,
            val_loss: float,
            val_metrics: dict
    ):
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.best_loss_epoch = epoch
        elif epoch - self.best_loss_epoch >= self.num_plateau_epochs:
            return True
        return None


class MrlTrainerCallback:
    def on_epoch_start(self, actor: nn.Module, epoch: int, stage_epochs: int, global_epoch: int,
                       global_epochs: int, curriculum_config: dict) -> None:
        pass

    def on_epoch_end(self, actor: nn.Module, epoch: int, stage_epochs: int, policy_loss: float,
                     critic_loss: float, global_epoch: int, global_epochs: int) -> None:
        pass

    def on_episode_collected(self, actor: nn.Module, batch_idx: int, episode_trajectories: list[dict],
                             reward: float) -> None:
        pass

    def on_reward(self, actor: nn.Module, rewards: list[float], generated: str, reference: str, saved_data: str, eval_mode: bool) -> None:
        pass

    def on_update_epoch_start(self, actor: nn.Module, critic: nn.Module, global_epoch: int, update_epoch: int) -> None:
        pass

    def on_batch_updated(self, actor: nn.Module, epoch: int, step: int, policy_loss: float) -> None:
        pass

    def on_critic_updated(self, actor: nn.Module, critic: nn.Module, epoch: int, step: int,
                          critic_loss: float) -> None:
        pass

    def on_update_epoch_end(self, actor: nn.Module, critic: nn.Module, global_epoch: int, update_epoch: int, policy_loss: float, critic_loss: float) -> None:
        pass

    def on_training_end(self, actor: nn.Module, critic: nn.Module, curriculum_config: dict) -> None:
        pass

    def on_eval_end(self, actor: nn.Module, critic: nn.Module, epoch: int, eval_mean_reward: float) -> Union[bool, None]:
        pass

    def on_eval_episode_end(self, actor: nn.Module, epoch: int, batch_idx: int, reward: float) -> None:
        pass


class MrlPrintCallback(MrlTrainerCallback):
    def __init__(self, update_steps_interval: int = 10) -> None:
        super(MrlPrintCallback, self).__init__()
        self.update_steps_interval = update_steps_interval
        self.policy_losses = []
        self.critic_losses = []

    def on_epoch_start(self, actor: nn.Module, epoch: int, stage_epochs: int, curriculum_config: dict,
                       global_epoch: int, global_epochs: int) -> None:
        print(
            f'Starting epoch {epoch}/{stage_epochs - 1} (stage) | {global_epoch}/{global_epochs} (global) for {curriculum_config["steps"]} steps in {curriculum_config["strategy"]} strategy.')

    def on_epoch_end(self, actor: nn.Module, epoch: int, stage_epochs: int, policy_loss: float,
                     critic_loss: float, global_epoch: int, global_epochs: int) -> None:
        print(f'Finished epoch {epoch}/{stage_epochs - 1} (stage) | {global_epoch}/{global_epochs} (global)')
        print(f'Policy mean loss: {policy_loss} | Critic mean loss: {critic_loss}')

    def on_episode_collected(self, actor: nn.Module, batch_idx: int, episode_trajectories: list[dict],
                             reward: float) -> None:
        print(f'Collected {batch_idx} episode | mean reward {reward}')

    def on_reward(self, actor: nn.Module, rewards: list[float], generated: dict[str, torch.Tensor],
                  reference: dict[str, torch.Tensor], saved_data: dict[str, torch.Tensor], eval_mode: bool) -> None:
        print(f"{'Eval' if eval_mode else 'Train'} | Mean reward: {sum(rewards) / len(rewards)} | All collected rewards: {rewards}")

    def on_update_epoch_start(self, actor: nn.Module, critic: nn.Module, global_epoch: int, update_epoch: int) -> None:
        print(f'Epoch {global_epoch} | Starting update epoch {update_epoch}')

    def on_batch_updated(self, actor: nn.Module, epoch: int, step: int, policy_loss: float) -> None:
        if step != 0 and step % self.update_steps_interval == 0:
            loss = sum(self.policy_losses) / len(self.policy_losses)
            self.policy_losses = []
            print(f'Epoch {epoch} | Steps {step - self.update_steps_interval} - {step} - mean policy loss {loss} | current policy loss {policy_loss}')
        else:
            self.policy_losses.append(policy_loss)

    def on_critic_updated(self, actor: nn.Module, critic: nn.Module, epoch: int, step: int,
                          critic_loss: float) -> None:
        if step != 0 and step % self.update_steps_interval == 0:
            loss = sum(self.critic_losses) / len(self.critic_losses)
            self.critic_losses = []
            print(f'Epoch {epoch} | Steps {step - self.update_steps_interval} - {step} - mean critic loss {loss} | current critic loss {critic_loss}')
        else:
            self.critic_losses.append(critic_loss)

    def on_update_epoch_end(self, actor: nn.Module, critic: nn.Module, global_epoch: int, update_epoch: int, policy_loss: float, critic_loss: float) -> None:
        print(f'Epoch {global_epoch} | Update epoch {update_epoch} - mean policy loss {policy_loss} | mean critic loss {critic_loss}')

    def on_training_end(self, actor: nn.Module, critic: nn.Module, curriculum_config: dict) -> None:
        print(f'Finished training for {curriculum_config["steps"]} steps in {curriculum_config["strategy"]} strategy.')

    def on_eval_end(self, actor: nn.Module, critic: nn.Module, epoch: int, eval_mean_reward: float) -> None:
        print(f'Eval epoch {epoch} - mean reward {eval_mean_reward}')

    def on_eval_episode_end(self, actor: nn.Module, epoch: int, batch_idx: int, reward: float) -> None:
        print(f'Eval epoch {epoch} / Episode {batch_idx} - mean reward {reward}')


class MrlEarlyStoppageCallback(MrlTrainerCallback):
    def __init__(self, num_plateau_epochs: int = 2, threshold: Optional[float] = None) -> None:
        super().__init__()
        self.num_plateau_epochs = num_plateau_epochs
        self.best_reward = -9999.0
        self.best_reward_epoch = 0
        self.threshold = threshold

    def on_eval_end(self, _actor: nn.Module, _critic: nn.Module, epoch: int, eval_mean_reward: float) -> Union[bool, None]:
        if self.threshold is not None:
            if eval_mean_reward > self.threshold:
                return True

        if eval_mean_reward > self.best_reward:
            self.best_reward = eval_mean_reward
            self.best_reward_epoch = epoch
        elif epoch - self.best_reward_epoch >= self.num_plateau_epochs:
            return True
        return None

class MrlModelSaveCallback(MrlTrainerCallback):
    def __init__(
            self,
            save_dir: str,
            max_keep: int = 3,
            push_to_hub: bool = False,
            hub_model_decoder: str = None,
            hub_model_encoder: str = None,
            hub_model_memory_attention: str = None,
            hub_model_critic: str = None,
            private_repo: bool = False,
            hf_token: str = None,
            push_checkpoint_weights: bool = True,
            final_commit_message: str = None,
            display_exc_trace: bool = False,
            use_ddp: bool = False,
    ):
        self.save_dir = save_dir
        self.max_keep = max_keep
        self.best_reward = float('-inf')
        self.ckpt_paths = []
        self.push_to_hub = push_to_hub
        self.hub_model_decoder = hub_model_decoder
        self.hub_model_encoder = hub_model_encoder
        self.hub_model_memory_attention = hub_model_memory_attention
        self.hub_model_critic = hub_model_critic
        self.private_repo = private_repo
        self.hf_token = hf_token
        self.push_checkpoint_weights = push_checkpoint_weights
        self.final_commit_message = final_commit_message
        self.finished_epochs = 0
        self.display_exc_trace = display_exc_trace
        self.rank = int(os.environ['RANK']) if use_ddp else 0

    def _save_eval(self, model: Union[nn.Module, PyTorchModelHubMixin], component: str, epoch: int,
                         reward: float, hub_id: str = None):
        try:
            if model.save_pretrained is not None:
                ckpt_path = os.path.join(
                    self.save_dir,
                    component,
                    f'epoch_{epoch}_eval_reward_{reward:.4f}'
                )
                path_exists = os.path.exists(ckpt_path)
                if not path_exists:
                    os.makedirs(ckpt_path)
                model.save_pretrained(save_directory=ckpt_path)
            else:
                comp_path = os.path.join(
                    self.save_dir,
                    component
                )
                path_exists = os.path.exists(comp_path)
                if not path_exists:
                    os.makedirs(comp_path)
                ckpt_path = os.path.join(
                    comp_path,
                    f'epoch_{epoch}_eval_reward_{reward:.4f}.pt'
                )
                torch.save(model.state_dict(), ckpt_path)
            self.ckpt_paths.append(ckpt_path)

            # Keep only N best checkpoints
            if len(self.ckpt_paths) > self.max_keep:
                oldest_path = self.ckpt_paths.pop(0)
                if model.save_pretrained is not None:
                    shutil.rmtree(oldest_path)
                else:
                    os.remove(oldest_path)
        except Exception as e:
            print(f"Error saving epoch checkpoint: {str(e)}")
            if self.display_exc_trace:
                traceback.print_exc()

        try:
            if self.push_to_hub and self.push_checkpoint_weights and model.push_to_hub is not None and hub_id:
                model.push_to_hub(
                    repo_id=hub_id,
                    commit_message=f'Epoch {epoch} - Eval reward {reward:.4f}',
                    token=self.hf_token,
                    private=self.private_repo,
                )
        except Exception as e:
            print(f"Error pushing epoch checkpoint: {str(e)}")
            if self.display_exc_trace:
                traceback.print_exc()

    def on_eval_end(self, actor: nn.Module, critic: nn.Module, epoch: int, eval_mean_reward: float) -> None:
        if self.rank == 0:
            self.finished_epochs += 1
            if eval_mean_reward > self.best_reward:
                self.best_reward = eval_mean_reward
                if isinstance(actor, DistributedDataParallel):
                    actor = next(actor.children())
                self._save_eval(actor.encoder, 'encoder', epoch, eval_mean_reward, hub_id=self.hub_model_encoder)
                self._save_eval(actor.decoder, 'decoder', epoch, eval_mean_reward, hub_id=self.hub_model_decoder)
                self._save_eval(actor.memory_attention, 'memory_attention', epoch, eval_mean_reward, hub_id=self.hub_model_memory_attention)
                if isinstance(critic, DistributedDataParallel):
                    critic = next(critic.children())
                self._save_eval(critic, 'critic', epoch, eval_mean_reward, hub_id=self.hub_model_critic)

    def _save_final(self, model: Union[nn.Module, PyTorchModelHubMixin], component: str, hub_id: str = None):
        try:
            # Save final model
            if model.save_pretrained is not None:
                ckpt_path = os.path.join(
                    self.save_dir,
                    component,
                    'final_model'
                )
                path_exists = os.path.exists(ckpt_path)
                if not path_exists:
                    os.makedirs(ckpt_path)
                model.save_pretrained(save_directory=ckpt_path)
            else:
                comp_path = os.path.join(
                    self.save_dir,
                    component
                )
                path_exists = os.path.exists(comp_path)
                if not path_exists:
                    os.makedirs(comp_path)
                ckpt_path = os.path.join(comp_path, 'final_model.pt')
                torch.save(model.state_dict(), ckpt_path)
            print(f"Final model saved to {ckpt_path}")
        except Exception as e:
            print(f"Error saving final model: {str(e)}")
            if self.display_exc_trace:
                traceback.print_exc()
        try:
            if self.push_to_hub and model.push_to_hub is not None and hub_id:
                model.push_to_hub(
                    repo_id=hub_id,
                    commit_message=self.final_commit_message or f'Model after full curriculum stage, after {self.finished_epochs} epochs',
                    token=self.hf_token,
                    private=self.private_repo,
                )
        except Exception as e:
            print(f"Error pushing final model: {str(e)}")
            if self.display_exc_trace:
                traceback.print_exc()

    def on_training_end(self, actor: nn.Module, critic: nn.Module, curriculum_config: dict) -> None:
        if self.rank == 0:
            if isinstance(actor, DistributedDataParallel):
                actor = next(actor.children())
            self._save_final(actor.encoder, 'encoder', hub_id=self.hub_model_encoder)
            self._save_final(actor.decoder, 'decoder', hub_id=self.hub_model_decoder)
            self._save_final(actor.memory_attention, 'memory_attention', hub_id=self.hub_model_memory_attention)
            if isinstance(critic, DistributedDataParallel):
                critic = next(critic.children())
            self._save_final(critic, 'critic', hub_id=self.hub_model_critic)

class MrlGeneratedTokensCallback(MrlTrainerCallback):
    def __init__(self, steps_log_interval: int = 100):
        self.total_tokens = 0
        self.steps_log_interval = steps_log_interval
        self.step = 0

    def on_reward(self, actor: nn.Module, rewards: list[float], generated: dict[str, torch.Tensor],
                  reference: dict[str, torch.Tensor], saved_data: dict[str, torch.Tensor], eval_mode: bool) -> None:
        self.step += 1
        attention_mask = generated['attention_mask']
        batch_tokens = attention_mask.sum().item()
        self.total_tokens += batch_tokens
        if self.step != 0 and self.step % self.steps_log_interval == 0:
            print(f'Total processed tokens: {human_format(self.total_tokens)}')

    def on_training_end(self, actor: nn.Module, critic: nn.Module, curriculum_config: dict) -> None:
        print(f'Total training tokens: {human_format(self.total_tokens)}')

    def get_total_tokens(self):
        return self.total_tokens

