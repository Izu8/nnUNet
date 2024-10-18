import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric, SurfaceDistanceMetric
import math
from torch import optim
import sys
import os
import wandb
from monai.inferers import sliding_window_inference
from monai.transforms import AsDiscrete, Compose
from monai.data import decollate_batch
import matplotlib.pyplot as plt
import pandas as pd
import torch.distributed as dist
import gc
import random
from torch._dynamo import OptimizedModule





class CVLoop:
    def __init__(self,model,config,current_epoch,total_epoch,iteration_num,train_dataloader,val_dataloader,
                 loss,optimizer,lr_scheduler,fabric,run_name="",infer_device="cpu",sw_batch_num=2,fold=None,
                 val_interval=10,best_val_score=None,
                 result_dir="",net_param_dir="",summary_path="",):
        self.model = model
        self.current_epoch = current_epoch
        self.total_epoch = total_epoch
        self.iteration_num = iteration_num
        self.lr = config["lr"]
        self.loss = loss
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.dice_metric = DiceMetric(include_background=False, reduction="mean_batch", get_not_nans=False)
        self.assd_metric = SurfaceDistanceMetric(include_background=False,symmetric=True,reduction="mean_batch",distance_metric='euclidean',get_not_nans=False)
        #self.scaler = grad_scaler
        self.train_loader = train_dataloader
        self.val_loader = val_dataloader
        self.fabric = fabric
        self.out_channels = config["class_num"]
        self.label_names = list(config["label_names"].keys())[1:]
        self.post_label = AsDiscrete(to_onehot=config["class_num"])
        self.post_pred = AsDiscrete(argmax=True, to_onehot=config["class_num"])
        self.sw_batch_num = sw_batch_num
        self.rand_crop_size = config["model_params"]["image_size"]
        self.infer_device = (infer_device if infer_device == "cpu" else self.fabric.device)
        self.fold = fold
        self.val_interval = val_interval
        self.init_run_name = run_name
        self.run_name = run_name + "_fold" + str(fold)
        self.early_stopping = Early_stopping(best_score=best_val_score)
        self.result_df = pd.read_csv(f"/win/scallop/user/ogura/nnUNet/datasets/nnUnet_results/{result_dir}/{summary_path}",encoding="utf-8")
        self.config = config
        self.result_dir = result_dir
        self.net_param_dir = net_param_dir
        self.summary_path = summary_path
        self.world_size = 1
        self.init_row =[self.config["voxel_size"],self.config["model_type"],self.config["pretrain"],infer_device,self.config["rand_crop_num"],self.sw_batch_num,fold]
        
        self.class_list = {
                "bolus": [],
                "nank": [],
                "zetu": [],
                "skull": [],
                "mandible": [],
                "cervical_spine": [],
                "hyoid_bone": [],
                "thyroid_cartilag": []
                }
        self.frame_num = []
            
        wandb.init(project="Swallowing-Segmentation-8structures", name=self.run_name+f"_epoch{self.total_epoch}",config=config)
        
    
    def train(self,over_write=True):
        if self.fold == 0 and self.current_epoch == 0:
            os.makedirs(f"/win/scallop/user/ogura/nnUNet/net_params/{self.net_param_dir}/{self.init_run_name+'_'+self.config['plan']}",exist_ok=over_write)
            os.makedirs(f"/win/scallop/user/ogura/nnUNet/datasets/nnUnet_results/{self.result_dir}/{self.init_run_name+'_'+self.config['plan']}",exist_ok=over_write)
        
        if self.current_epoch == 0:
            os.makedirs(f"/win/scallop/user/ogura/nnUNet/net_params/{self.net_param_dir}/{self.init_run_name+'_'+self.config['plan']}/fold{self.fold}",exist_ok=over_write)
            os.makedirs(f"/win/scallop/user/ogura/nnUNet/datasets/nnUnet_results/{self.result_dir}/{self.init_run_name+'_'+self.config['plan']}/fold{self.fold}",exist_ok=over_write)

        
        for epoch in tqdm(range(self.current_epoch+1,self.current_epoch+self.total_epoch+1)):
            self.lr_scheduler.step()
            current_lr = self.optimizer.param_groups[0]["lr"]
            wandb.log({"epoch":epoch,"lr":current_lr})
            
            if epoch % self.val_interval == 0:
                
                train_mean_epoch_loss = self.train_epoch()
                val_dice_metrics, val_assd_metrcic = self.validate_epoch(epoch)
            
                print(f"【Epoch:{epoch}】\n Train loss:{train_mean_epoch_loss:.7f} \
                \n Val dice score:{np.mean(val_dice_metrics):.7f}, Val ASSD {np.mean(val_assd_metrcic):.7f}")

                
                wandb.log({"epoch":epoch,"train/epoch loss": train_mean_epoch_loss,"val/mean dice score": np.nanmean(val_dice_metrics)})

                row = []
                val_dict = {"epoch":epoch}
                row.append(epoch)
                each_dice = []
                each_assd = []
                
                for label_name, val_dice, val_assd in zip(self.label_names,val_dice_metrics,val_assd_metrcic):
                    val_dict[f"val/{label_name} dice score"] = val_dice
                    val_dict[f"val/{label_name} ASSD"] = val_assd
                    each_dice.append(val_dice)
                    each_assd.append(val_assd)
                
                wandb.log(val_dict)
                row += each_dice
                row += each_assd
                row = self.init_row + row
                df = pd.DataFrame(data=[row],columns=self.result_df.columns)
                df.to_csv(f"/win/scallop/user/ogura/nnUNet/datasets/nnUnet_results/{self.result_dir}/{self.summary_path}",mode="a",index=False,header=False)
                
                checkpoint = {"epoch":epoch,
                            "state_dict":self.model.state_dict(),
                            "optimizer":self.optimizer.state_dict(),
                            "scheduler":self.lr_scheduler.state_dict(),
                            "best val score":np.nanmean(val_dice_metrics)}
                

                torch.save(checkpoint,f"/win/scallop/user/ogura/nnUNet/net_params/{self.net_param_dir}/{self.init_run_name+'_'+self.config['plan']}/fold{self.fold}/{self.run_name}_epoch{epoch}model.pth")
                self.early_stopping(np.nanmean(val_dice_metrics))
                if self.early_stopping.earlystop == True:
                    print("--Model was saved!--")
                    torch.save(checkpoint,f"/win/scallop/user/ogura/nnUNet/net_params/{self.net_param_dir}/{self.init_run_name+'_'+self.config['plan']}/fold{self.fold}/{self.run_name}_bestmodel.pth")
                    self.early_stopping.set_false()
            
            else:
                train_mean_epoch_loss = self.train_epoch()
                print(f"【Epoch:{epoch}】\n Train loss:{train_mean_epoch_loss:.7f}")
                wandb.log({"epoch":epoch,"train/epoch loss": train_mean_epoch_loss})
        
        

        #torch.save(checkpoint,f"../net_params/8stractures/{self.init_run_name}/fold{self.fold}/{self.run_name}_epoch{self.n_epochs}.pth")
        print(f"--fold{self.fold} was finished!")
        wandb.finish()
        
            
    
    def train_epoch(self):
        self.model.train()
        epoch_loss = 0
        print("train loop")
        train_dataloader_iter = iter(self.train_loader)
        for idx in range(self.iteration_num):
            
            try:
                batch = next(train_dataloader_iter)
            except StopIteration:
                train_dataloader_iter = iter(self.train_loader)
                batch = next(train_dataloader_iter)
                
            images = batch["image"]
            labels = batch["label"]

            #if isinstance(labels, list):
               # labels = [i.to(self.device) for i in labels]
            #else:
                #labels = labels.to(self.device)
            
            #print(f"image:{images.device}")
            #print(f"model:{next(self.model.parameters()).is_cuda}")
            #print(f"label:{labels[0].device}")
            
            logit = self.model(images)
            with self.fabric.autocast():
                loss = self.loss(logit, labels)

            epoch_loss += loss.item()
            wandb.log({"train/loss": loss.item()})
            

            self.optimizer.zero_grad()
            self.fabric.backward(loss)
            self.optimizer.step()
            
        
            del images, labels, logit, loss
            torch.cuda.empty_cache()
        
        mean_epoch_loss = epoch_loss / self.iteration_num
        
        return mean_epoch_loss
   
    def validate_epoch(self,epoch):
        self.set_deep_supervision_enabled(False)
        self.model.eval()
        epoch_loss = 0
        print("validation loop")
        
        with torch.no_grad():
            for frame, batch in enumerate(self.val_loader):
                
                print(f"val iter:{frame}")
                print(f"sw:{self.infer_device}")
                
                images = batch["image"]
                labels = batch["label"].to(self.infer_device)
                
                with torch.cuda.amp.autocast():
                    logit = sliding_window_inference(images, self.rand_crop_size, self.sw_batch_num, self.model, mode="gaussian",sigma_scale=1. / 8, 
                                                     device=self.infer_device)
                    #loss = self.loss(logit, labels)
                
                #epoch_loss += loss.item()
                output_convert, labels_convert = self.convert_onehot(logit,labels)
                
                
                
                dice_result = self.dice_metric(y_pred=output_convert, y=labels_convert)[0]
                assd_result = self.assd_metric(y_pred=output_convert, y=labels_convert)[0]
                
                for idx, class_name in enumerate(self.class_list.keys()):
                    dice_value = dice_result[idx].item()
                    assd_value = assd_result[idx].item()
                    self.class_list[class_name].append(dice_value)
                self.frame_num.append(frame+1)
            
                
            
                del images, labels, logit,output_convert, labels_convert
                torch.cuda.empty_cache()
                gc.collect()

            #mean_epoch_loss = epoch_loss / len(self.val_loader)
            dice_metric = self.dice_metric.aggregate().tolist()
            assd_metric = self.assd_metric.aggregate().tolist()
            self.dice_metric.reset()
            self.assd_metric.reset()
    
        result_df = pd.DataFrame(data=[],columns=[])
        result_df["frame number"]= self.frame_num
        for class_name in self.class_list.keys():
            result_df[class_name] = self.class_list[class_name]
        
        result_df.to_csv(f"/win/scallop/user/ogura/nnUNet/datasets/nnUnet_results/{self.result_dir}/{self.init_run_name+'_'+self.config['plan']}/fold{self.fold}/result_epoch{epoch}_df.csv",index=False)   
        self.reset_dict()
        
        self.set_deep_supervision_enabled(True)

        return dice_metric, assd_metric
    
    def convert_onehot(self, outputs, true_labels):
        labels_list = decollate_batch(true_labels)
        labels_convert = [self.post_label(label_tensor) for label_tensor in labels_list]
        outputs_list = decollate_batch(outputs)
        output_convert = [self.post_pred(pred_tensor) for pred_tensor in outputs_list]
        return output_convert, labels_convert
    
    def set_deep_supervision_enabled(self, enabled: bool):
        """
        This function is specific for the default architecture in nnU-Net. If you change the architecture, there are
        chances you need to change this as well!
        """

        self.model.decoder.deep_supervision = enabled

    
    def reset_dict(self):
        self.class_list = {
                "bolus": [],
                "nank": [],
                "zetu": [],
                "skull": [],
                "mandible": [],
                "cervical_spine": [],
                "hyoid_bone": [],
                "thyroid_cartilag": []
                }
        self.frame_num = []
        


class Early_stopping:
    def __init__(self, patience=1, verbose=False, delta=0, best_score=None, smaller_better=False):
        self.best_score = best_score
        self.current_score = None
        self.counter = 0
        self.patiece = patience
        self.delta = delta
        self.earlystop = False
        self.verbose = verbose
        self.smaller_better = smaller_better
    def __call__(self, val_metric):
        self.current_score = val_metric
        
        if self.smaller_better:
            if self.best_score == None:
                self.best_score = val_metric
            
            elif self.current_score - self.best_score < self.delta:
                self.best_score = self.current_score
                self.counter += 1
                if self.counter == self.patiece:
                    self.earlystop = True
        
        else:
            if self.best_score == None:
                self.best_score = val_metric
            
            elif self.current_score - self.best_score > self.delta:
                self.best_score = self.current_score
                self.counter += 1
                if self.counter == self.patiece:
                    self.earlystop = True
        

    def Print_score(self):
        if self.verbose == True:
            print(f"Val loss(Current):{self.current_score}")
            
    def set_false(self):
        self.earlystop = False
        self.counter = 0
    
    def get_best_score(self):
        return self.best_score




def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


