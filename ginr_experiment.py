from imports import *
from models import *
from utils import *


class GINR_Experiment():
    def __init__(self, experiment_name, config):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        self.experiment_name = experiment_name
        self.config = deepcopy(config)
        
        output_dir = './ginr_checkpoints/' + self.experiment_name
        os.makedirs(output_dir, exist_ok=True)
        
        self.output_dir = output_dir
        self.checkpoint_path = os.path.join(output_dir, 'checkpoint.pth.tar')
        self.config_path = os.path.join(output_dir, 'config.txt')
        
        self.history = []
        self.train_loss = []
        self.train_auroc = []
        self.test_loss = []
        self.test_auroc = []

        dataset_args = deepcopy(self.config['dataset_args'])

        dataset_args.update(train=True)
        self.train_loader = DataLoader(
            self.config['dataset_class'](**dataset_args),
            batch_size=self.config['batch_size'],
            shuffle=True,
            pin_memory=True,
        )

        dataset_args.update(train=False)
        self.test_loader = DataLoader(
            self.config['dataset_class'](**dataset_args),
            batch_size=self.config['batch_size'],
            shuffle=False,
            pin_memory=True,
        )
        
        self.model = self.config['model_class'](
            self.train_loader.dataset.n_fourier, 
            self.train_loader.dataset.target_dim,
            **self.config['model_args'],
        ).to(self.device)
        self.optimizer = self.config['optimizer_class'](self.model.parameters(), **self.config["optimizer_args"])
        
        self.loss_fn = self.config['loss_fn'](weight=self.train_loader.dataset.label_weight.to(self.device))
        
        self.config['model_class'] = self.model
        self.config['optimizer_class'] = self.optimizer
        self.config['loss_fn'] = self.loss_fn
        
        # load checkpoint and check compatibility 
        if os.path.isfile(self.config_path):
            with open(self.config_path, 'r') as f:
               if f.read()[:-1] != repr(self):
                   raise ValueError(
                       "Cannot create this experiment: "
                       "Checkpoint found with same name but different config."
                   )
                    
            self.load()
        else:
            self.save()
            
    @property
    def epoch(self):
        return len(self.history)
    
    def setting(self):
        return self.config
    
    def __repr__(self):
        string = ''
        
        for key, val in self.setting().items():
            string += '{}: {}\n'.format(key, val)
            
        return string
    
    def state_dict(self):
        return {
            'model' : self.model.state_dict(),
            'optimizer' : self.optimizer.state_dict(),
            'history' : self.history,
            'train loss' : self.train_loss,
            'train auroc' : self.train_auroc,
            'test loss' : self.test_loss,
            'test auroc' : self.test_auroc,
        }
    
    def load_state_dict(self, checkpoint):
        self.model.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        
        self.history = checkpoint['history']
        self.train_loss = checkpoint['train loss']
        self.train_auroc = checkpoint['train auroc']
        self.test_loss = checkpoint['test loss']
        self.test_auroc = checkpoint['test auroc']
        
    def save(self):
        torch.save(self.state_dict(), self.checkpoint_path)
        
        with open(self.config_path, 'w') as f:
            print(self, file=f)
            
    def load(self):
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        
        self.load_state_dict(checkpoint)
        
        del checkpoint
        
    def plot(self, clear=False):
        if clear:
            display.display(plt.clf())
            display.clear_output(wait=True)
        
        fig, axes = plt.subplots(2, 2, figsize=(10, 5), constrained_layout=True)
        
        axes = axes.flatten()
        
        axes[0].clear()
        axes[1].clear()
        axes[2].clear()
        axes[3].clear()
        
        axes[0].plot(self.train_loss)
        axes[0].set_title('Training loss', size=16, color='teal')
        axes[0].set_xlabel('Epochs', size=16, color='teal')
        axes[0].set_ylim(0, 10)
        axes[0].grid()
        
        axes[1].plot(self.test_loss)
        axes[1].set_title('Testing loss', size=16, color='teal')
        axes[1].set_xlabel('Epochs', size=16, color='teal')
        axes[1].set_ylim(0, 10)
        axes[1].grid()

        axes[2].plot(self.train_auroc)
        axes[2].set_title('Training AUROC', size=16, color='teal')
        axes[2].set_xlabel('Epochs', size=16, color='teal')
        axes[2].set_ylim(0, 1)
        axes[2].grid()
        
        axes[3].plot(self.test_auroc)
        axes[3].set_title('Testing AUROC', size=16, color='teal')
        axes[3].set_xlabel('Epochs', size=16, color='teal')
        axes[3].set_ylim(0, 1)
        axes[3].grid()
        
        plt.show()
        
    def train_epoch(self):
        self.model.train()
        
        losses = []
        
        for inputs, targets in self.train_loader:
            inputs = inputs.to(self.device)
            inputs.requires_grad_()

            targets = targets.to(self.device)

            pred = self.model(inputs.abs()).permute(0, 1)

            loss = self.loss_fn(pred, targets)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            losses.append(loss.detach().cpu().item())

        return np.mean(losses)


    def test_epoch(self):
        self.model.eval()
        
        losses = []
        
        with torch.no_grad():
            for inputs, targets in self.test_loader:
                inputs = inputs.to(self.device)

                targets = targets.to(self.device)

                pred = self.model(inputs.abs()).permute(0, 1)
                
                loss = self.loss_fn(pred, targets)

                losses.append(loss.detach().cpu().item())
        
        return np.mean(losses)

    def auroc(self, train, save_predictions=False):
        split_label = "Train" if train else "Test"
        loader = self.train_loader if train else self.test_loader

        predictions = []
        labels = []
        objects = []

        self.model.eval()

        with torch.no_grad():
            for i in range(loader.dataset.num_objects):
                points, targets = loader.dataset.get_object_points(i)
                points = points.to(self.device)
                targets = targets.to(self.device)

                objects.append(loader.dataset.obj_to_output_file[i][0])
                predictions.append(self.model(points.abs()))
                labels.append(targets)

            all_predictions = torch.cat(predictions, dim=0)
            all_labels = torch.cat(labels, dim=0)

            AUROC = MulticlassAUROC(num_classes=(all_labels.max() + 1).detach().cpu().item(), average="weighted")
            auroc = AUROC(all_predictions, all_labels).detach().cpu().item()

            if save_predictions:
                predictions_dir = os.path.join(self.output_dir, "predictions")
                os.makedirs(predictions_dir, exist_ok=True)

                for o, p in zip(objects, predictions):
                    np.save(os.path.join(predictions_dir, o), p.argmax(dim=1).detach().cpu().numpy())

                with open(os.path.join(predictions_dir, split_label + " AUROC.txt"), "w") as f:
                    for o, p, l in zip(objects, predictions, labels):
                        s = "{}, MulticlassAUROC:".format(o) + " {}".format(AUROC(p, l).detach().cpu().item()) + "\n"
                        f.write(s)

            return auroc
    
    def train(self, num_epochs, show_plot):
        if show_plot:
            self.plot(clear=False)
        
        print(self)
        print("Number of model parameters: {}".format(count_parameters(self.model)))
        print("Start/Continue training from epoch {0}".format(self.epoch))
        
        while self.epoch < num_epochs:            
            self.train_loss.append(self.train_epoch())
            self.train_auroc.append(self.auroc(train=True, save_predictions=False))

            if (self.epoch % 10) == 0:
                self.test_loss.append(self.test_epoch())
                self.test_auroc.append(self.auroc(train=False, save_predictions=False))
            
            self.history.append(self.epoch + 1)
            
            if show_plot and (self.epoch + 1) % 100 == 0:
                self.plot(clear=True)
            elif not show_plot:
                print("Epoch: {0}".format(self.epoch))
                print("Train Loss: {:.4f}".format(self.train_loss[-1]))
                print("Test Loss: {:.4f}".format(self.test_loss[-1]))
                print("Train AUROC: {:.4f}".format(self.train_auroc[-1]))
                print("Test AUROC: {:.4f}".format(self.test_auroc[-1]))
                print()
            
            thread = Thread(target=self.save)
            thread.start()
            thread.join()
            
        print("Finished training\n")

        print("Saving predictions\n")
        _ = self.auroc(train=True, save_predictions=True)
        _ = self.auroc(train=False, save_predictions=True)
        print("Finished\n")
