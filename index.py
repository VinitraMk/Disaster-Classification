import os
import torch
from torch.utils.data.dataset import random_split
from torch.utils.data.dataloader import DataLoader
from azureml.core import Workspace, Experiment as AzExperiment, Environment, ScriptRunConfig
from azureml.core.dataset import Dataset
from azureml.data import OutputFileDatasetConfig
from datetime import date
from azure.storage.blob import BlobServiceClient, BlobClient
import pandas as pd

#custom imports
from experiments.preprocessor import Preprocessor
from models.ann import ANN
from experiments.DisasterDataset import DisasterDataset
from helper.utils import get_config, get_filename, get_model_params, get_preproc_params, save_model, save_tensor

class Index:

    model_args = None
    preproc_args = None
    config = None
    preprocessor = None
    train_dataset = None
    test_dataset = None

    def __init__(self):
        if not(os.getenv('ROOT_DIR')):
            os.environ['ROOT_DIR'] = os.getcwd()

    def start_program(self):
        self.__define_args()
        self.__preprocess_data()
        self.__make_model()
        self.__prepare_datasets()
        self.__make_azure_resources()
        self.__start_training()
        
    def __define_args(self):
        self.model_args = get_model_params()
        self.config = get_config()
        self.preproc_args = get_preproc_params()

    def __preprocess_data(self):
        self.preprocessor = Preprocessor()
        self.train_dataset = DisasterDataset(f'{self.config["input_path"]}\\train.csv')
        self.test_dataset = DisasterDataset(f'{self.config["input_path"]}\\test.csv', False)
        self.preprocessor.start_preprocessing(self.train_dataset)
        #dataloader = DataLoader(train_dataset, batch_size = self.model_args["batch_size"], shuffle=False, collate_fn=preprocessor.collate_batch)

    def __make_model(self):
        self.model = ANN(self.preprocessor.get_vocab_size(), 64, 2)
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(self.model.parameters(), lr = self.model_args['lr'])
        model_obj = {
            'model_state': self.model.state_dict(),
            'criterion': criterion,
            'optimizer_state': optimizer.state_dict()
        }
        model_path = f"{self.config['processed_io_path']}\\models"
        save_model(model_obj, model_path, 'ann', True)

    def __prepare_datasets(self):
        num_train = int(len(self.train_dataset) * self.preproc_args["train_validation_split"])
        split_train_, split_valid_ = random_split(self.train_dataset, [num_train, len(self.train_dataset) - num_train])
        self.train_loader = DataLoader(split_train_, batch_size = self.model_args["batch_size"], shuffle = True, collate_fn=self.preprocessor.collate_batch)
        self.valid_loader = DataLoader(split_valid_, batch_size = self.model_args["batch_size"], shuffle = True, collate_fn=self.preprocessor.collate_batch)
        self.test_loader = DataLoader(self.test_dataset, batch_size = self.model_args["batch_size"], shuffle=False, collate_fn=self.preprocessor.collate_batch)

    def __make_azure_resources(self):
        print('\nConfiguring Azure Resources...')
        # Configuring workspace
        print('\tConfiguring Workspace...')
        today = date.today()
        todaystring = today.strftime("%d-%m-%Y")
        self.azws = Workspace.from_config()

        print('\tConfiguring Environment...')
        self.azenv = Environment.get(workspace=self.azws, name="vinazureml-env")
        self.azexp = AzExperiment(workspace=self.azws, name=f'{todaystring}-experiments')
        print('\tGetting default blob store...\n')
        self.def_blob_store = self.azws.get_default_datastore()

    def __copy_model_chkpoint(self):
        local_model_path = f"{self.config['internal_output_path']}/ann.pt"
        upload_model_path = f"{self.config['processed_io_path']}/models/ann.pt"
        if (os.path.isfile(local_model_path)):
            os.system(f"cp {local_model_path} {upload_model_path}")

    def __train_model_in_azure(self, is_first_batch = False, is_last_batch = False):
        self.__copy_model_chkpoint()
        self.def_blob_store.upload(src_dir='./processed_io', target_path="input/", overwrite=True)
        input_data = Dataset.File.from_files(path=(self.def_blob_store, '/input'))
        input_data = input_data.as_named_input('input').as_mount()
        output = OutputFileDatasetConfig(destination=(self.def_blob_store, '/output'))
        config = ScriptRunConfig(
            source_directory='./models/training_scripts',
            script='train_nn.py',
            arguments=[input_data, output, 'ann', self.preprocessor.get_vocab_size(), is_first_batch, is_last_batch],
            compute_target="mikasa",
            environment=self.azenv
        )

        run = self.azexp.submit(config)
        run.wait_for_completion(show_output = True, raise_on_error = True)

    def __download_output(self):
        config = get_config()
        STORAGE_ACCOUNT_URL = 'https://mlintro1651836008.blob.core.windows.net/'
        MAIN_OUTPUT_CONTAINER = 'azureml-blobstore-31aeaa24-564c-4aa8-bdf4-fc4b5707bd1b/output'
        MODEL_CONTAINER = f'{MAIN_OUTPUT_CONTAINER}/models'
        LOCAL_MODEL_PATH = f"{config['internal_output_path']}\\ann_model.pt"
        blob_service_client = BlobServiceClient(account_url=STORAGE_ACCOUNT_URL, credential=os.environ["AZURE_STORAGE_CONNECTIONKEY"])
        blob_client_model = blob_service_client.get_blob_client(MODEL_CONTAINER, 'ann_model.pt', snapshot=None)
        self.__download_blob(LOCAL_MODEL_PATH, blob_client_model)


    def __download_blob(self, local_filename, blob_client_instance):
        with open(local_filename, "wb") as my_blob:
            blob_data = blob_client_instance.download_blob()
            blob_data.readinto(my_blob)

    def __evaluate_model(self, data_loader, is_test_dataset = False):
        if is_test_dataset:
            print('\nGetting test data predictions...')
        else:
            print('\nEvaluating model with validation set...')
        model_state_path = f"{self.config['internal_output_path']}\\ann_model.pt"
        model_object = torch.load(model_state_path)
        self.model.load_state_dict(model_object["model_state"])
        self.model.eval()
        total_acc, total_count, accu_val = 0, 0, 0
        if is_test_dataset:
            results_df = pd.DataFrame([])
        with torch.no_grad():
            for i, batch in enumerate(data_loader):
                predicted_probs = self.model(batch[1], batch[3])
                predicted_labels = predicted_probs.argmax(1)
                if not(is_test_dataset):
                    total_acc += (predicted_labels == batch[0]).sum().item()
                    total_count += batch[0].size(0)
                else:
                    predicted_df = pd.DataFrame(predicted_labels.numpy(), columns=['target'])
                    predicted_ids = pd.DataFrame(batch[2].numpy(), columns=['id'])
                    predicted_df = pd.concat([predicted_ids, predicted_df], axis = 1)
                    results_df.append(predicted_df, ignore_index = True)
            if not(is_test_dataset):
                accu_val = total_acc / total_count
            else:
                print(results_df.head(20))
                csv_output_path = f"{self.config['output_path']}\\{get_filename('ann')}.csv"
                results_df.to_csv(csv_output_path, index = False)
        return accu_val
        
   
    def __start_training(self):
        for epoch in range(1, self.model_args["num_epochs"] + 1):
            total_acc, total_count = 0, 0
            print('\nRunning epoch', epoch)
            for i, batch  in enumerate(self.train_loader):
                print(f'\tSending batch {i} for training...')
                save_tensor(batch[1], 'texts')
                save_tensor(batch[0], 'labels')
                save_tensor(batch[3], 'offsets')
                is_last_batch = (i == (len(self.train_loader) - 1))
                is_first_batch = i == 0
                self.__train_model_in_azure(is_first_batch, is_last_batch)
            self.__download_output()
            accu_val = self.__evaluate_model(self.valid_loader)
            print(f'Accuracy after epoch {epoch}:', accu_val)
        self.__evaluate_model(self.test_loader, True)
            


if __name__ == "__main__":
    index = Index()
    index.start_program()