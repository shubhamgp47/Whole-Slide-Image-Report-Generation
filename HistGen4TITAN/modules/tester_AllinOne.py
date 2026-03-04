import logging
import os
from abc import abstractmethod

import cv2
import pandas as pd
import torch

from modules.utils import generate_heatmap
from tqdm import tqdm
import logging

class BaseTester(object):
    def __init__(self, model, criterion, metric_ftns, args):
        self.args = args

        logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                            datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        # setup GPU device if available, move model into configured device
        self.device, device_ids = self._prepare_device(args.n_gpu)
        self.model = model.to(self.device)
        if len(device_ids) > 1:
            self.model = torch.nn.DataParallel(model, device_ids=device_ids)

        self.criterion = criterion
        self.metric_ftns = metric_ftns

        self.epochs = self.args.epochs
        self.save_dir = self.args.save_dir
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        self._load_checkpoint(args.load)

    @abstractmethod
    def test(self):
        raise NotImplementedError

    @abstractmethod
    def plot(self):
        raise NotImplementedError

    def _prepare_device(self, n_gpu_use):
        n_gpu = torch.cuda.device_count()
        if n_gpu_use > 0 and n_gpu == 0:
            self.logger.warning(
                "Warning: There\'s no GPU available on this machine," "training will be performed on CPU.")
            n_gpu_use = 0
        if n_gpu_use > n_gpu:
            self.logger.warning(
                "Warning: The number of GPU\'s configured to use is {}, but only {} are available " "on this machine.".format(
                    n_gpu_use, n_gpu))
            n_gpu_use = n_gpu
        device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
        list_ids = list(range(n_gpu_use))
        return device, list_ids

    def _load_checkpoint(self, load_path):
        load_path = str(load_path)
        self.logger.info("Loading checkpoint: {} ...".format(load_path))
        checkpoint = torch.load(load_path)
        self.model.load_state_dict(checkpoint['state_dict'])



#OG
class Tester(BaseTester):
    def __init__(self, model, criterion, metric_ftns, args, test_dataloader):
        super(Tester, self).__init__(model, criterion, metric_ftns, args)
        self.test_dataloader = test_dataloader

    def test(self):
        self.logger.info('Start to evaluate in the test set.')
        log = dict()
        self.model.eval()
        with torch.no_grad():
            test_gts, test_res, test_ids = [], [], []
            for batch_idx, (images_id, images, reports_ids, reports_masks) in tqdm(enumerate(self.test_dataloader)):
                images_id, images, reports_ids, reports_masks = images_id[0], images.to(self.device), reports_ids.to(
                    self.device), reports_masks.to(self.device)
                output = self.model(images, mode='sample')
                reports = self.model.tokenizer.decode_batch(output.cpu().numpy())
                ground_truths = self.model.tokenizer.decode_batch(reports_ids[:, 1:].cpu().numpy())
                test_res.extend(reports)
                test_gts.extend(ground_truths)
                test_ids.append(images_id)
            
            test_met = self.metric_ftns({i: [gt] for i, gt in enumerate(test_gts)},
                                        {i: [re] for i, re in enumerate(test_res)})
            log.update(**{'test_' + k: v for k, v in test_met.items()})
            print(log)

            # Convert to pandas DataFrame
            test_res_df = pd.DataFrame(test_res, columns=['Generated Reports'])
            test_gts_df = pd.DataFrame(test_gts, columns=['Ground Truths'])

            # Create DataFrame for IDs
            test_ids_df = pd.DataFrame(test_ids, columns=['Case ID'])

            # Merge the DataFrames
            merged_df = pd.concat([test_ids_df, test_res_df, test_gts_df], axis=1)

            # Save the merged DataFrame to a CSV file
            merged_df.to_csv(os.path.join(self.save_dir, "gen_vs_gt.csv"), index=False)
            test_res_df.to_csv(os.path.join(self.save_dir, "res.csv"), index=False)
            test_gts_df.to_csv(os.path.join(self.save_dir, "gts.csv"), index=False)

            # Save evaluation metrics to results.csv
            metrics_df = pd.DataFrame([log])  # Wrap in list to make it a single-row DataFrame
            metrics_df.to_csv(os.path.join(self.save_dir, "results.csv"), index=False)

        return log

    #OG
    def plot(self):
        assert self.args.batch_size == 1 and self.args.beam_size == 1
        self.logger.info('Start to plot attention weights in the test set.')
        os.makedirs(os.path.join(self.save_dir, "attentions"), exist_ok=True)
        mean = torch.tensor((0.485, 0.456, 0.406))
        std = torch.tensor((0.229, 0.224, 0.225))
        mean = mean[:, None, None]
        std = std[:, None, None]

        self.model.eval()
        with torch.no_grad():
            for batch_idx, (images_id, images, reports_ids, reports_masks) in tqdm(enumerate(self.test_dataloader)):
                images, reports_ids, reports_masks = images.to(self.device), reports_ids.to(
                    self.device), reports_masks.to(self.device)
                output = self.model(images, mode='sample')
                image = torch.clamp((images[0].cpu() * std + mean) * 255, 0, 255).int().cpu().numpy()
                report = self.model.tokenizer.decode_batch(output.cpu().numpy())[0].split()
                attention_weights = [layer.src_attn.attn.cpu().numpy()[:, :, :-1].mean(0).mean(0) for layer in
                                     self.model.encoder_decoder.model.decoder.layers]
                for layer_idx, attns in enumerate(attention_weights):
                    assert len(attns) == len(report)
                    for word_idx, (attn, word) in enumerate(zip(attns, report)):
                        os.makedirs(os.path.join(self.save_dir, "attentions", "{:04d}".format(batch_idx),
                                                 "layer_{}".format(layer_idx)), exist_ok=True)

                        heatmap = generate_heatmap(image, attn)
                        cv2.imwrite(os.path.join(self.save_dir, "attentions", "{:04d}".format(batch_idx),
                                                 "layer_{}".format(layer_idx), "{:04d}_{}.png".format(word_idx, word)),
                        heatmap)
                        

'''
    #Experiment
    def plot(self):
        import numpy as np
        import cv2
        
        assert self.args.batch_size == 1 and self.args.beam_size == 1
        self.logger.info('Start to plot attention weights in the test set.')
        os.makedirs(os.path.join(self.save_dir, "attentions"), exist_ok=True)
        mean = torch.tensor((0.485, 0.456, 0.406))
        std = torch.tensor((0.229, 0.224, 0.225))
        mean = mean[:, None, None]
        std = std[:, None, None]
        
        def generate_heatmap(image, weights):
            """
            Modified version of original generate_heatmap to handle non-square attention weights
            """
            image = image.transpose(1, 2, 0)
            height, width, _ = image.shape
            
            # Fix for non-square weights: pad to next larger square
            size = int(np.ceil(np.sqrt(weights.shape[0])))
            padded_len = size * size
            
            # Pad weights with zeros to make perfect square
            padded_weights = np.zeros(padded_len)
            padded_weights[:weights.shape[0]] = weights
            
            # Now reshape safely (preserving original logic)
            weights = padded_weights.reshape(size, size)
            weights = weights - np.min(weights)
            weights = weights / np.max(weights)
            weights = cv2.resize(weights, (width, height))
            weights = np.uint8(255 * weights)
            heatmap = cv2.applyColorMap(weights, cv2.COLORMAP_JET)
            result = heatmap * 0.5 + image * 0.5
            return result
        
        def process_attention(attn_tensor, report_len):
            """
            Process attention tensor to match report length
            attn_tensor shape: [batch, heads, src_len, tgt_len] = [1, 8, 182, 512]
            Returns: [report_len, src_len] - attention weights per word
            """
            # Average across heads and remove batch dimension
            attn_avg_heads = attn_tensor.mean(axis=1)[0]  # Shape: [182, 512]
            
            # Take only the first 'report_len' target positions  
            attn_trimmed = attn_avg_heads[:, :report_len]  # Shape: [182, report_len]
            
            # Transpose to get attention per word: [report_len, 182]
            attn_per_word = attn_trimmed.T
            
            return attn_per_word
        
        self.model.eval()
        with torch.no_grad():
            for batch_idx, (images_id, images, reports_ids, reports_masks) in tqdm(enumerate(self.test_dataloader)):
                images, reports_ids, reports_masks = images.to(self.device), reports_ids.to(
                    self.device), reports_masks.to(self.device)
                output = self.model(images, mode='sample')
                image = torch.clamp((images[0].cpu() * std + mean) * 255, 0, 255).int().cpu().numpy()
                report = self.model.tokenizer.decode_batch(output.cpu().numpy())[0].split()
                report_len = len(report)
                
                self.logger.info(f"Processing batch {batch_idx}: Report length = {report_len}")
                self.logger.info(f"Report words: {report}")
                
                # Extract attention from the correct location we discovered
                if hasattr(self.model.encoder_decoder.attn_mem, 'attn'):
                    attn_tensor = self.model.encoder_decoder.attn_mem.attn.cpu().numpy()
                    self.logger.info(f"Extracted attention tensor shape: {attn_tensor.shape}")
                    
                    # Process attention to match report length
                    attn_per_word = process_attention(attn_tensor, report_len)
                    self.logger.info(f"Processed attention shape: {attn_per_word.shape}")
                    
                    # Verify dimensions match (this should now pass)
                    assert attn_per_word.shape[0] == report_len, f"Attention length {attn_per_word.shape[0]} != report length {report_len}"
                    
                    # Generate visualizations for each word
                    for word_idx, (attn, word) in enumerate(zip(attn_per_word, report)):
                        os.makedirs(os.path.join(self.save_dir, "attentions", "{:04d}".format(batch_idx),
                                                "layer_attn_mem"), exist_ok=True)
                        
                        # Generate heatmap using the fixed function
                        heatmap = generate_heatmap(image, attn)
                        
                        # Save the attention visualization
                        filename = os.path.join(self.save_dir, "attentions", "{:04d}".format(batch_idx),
                                            "layer_attn_mem", "{:04d}_{}.png".format(word_idx, word))
                        cv2.imwrite(filename, heatmap)
                    
                    self.logger.info(f"Successfully generated {len(report)} attention visualizations for batch {batch_idx}")
                    
                else:
                    self.logger.error("No attention tensor found in encoder_decoder.attn_mem")
                    self.logger.error("Check if the model has the expected attention mechanism")'''
                    

class CompetitionTester(BaseTester):
    """
    Modified Tester class for competition submissions where ground truth is not available
    """
    def __init__(self, model, args, test_dataloader):
        # Initialize without criterion and metric_ftns since we won't calculate metrics
        self.args = args

        logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                            datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        # setup GPU device if available, move model into configured device
        self.device, device_ids = self._prepare_device(args.n_gpu)
        self.model = model.to(self.device)
        if len(device_ids) > 1:
            self.model = torch.nn.DataParallel(model, device_ids=device_ids)

        self.epochs = self.args.epochs
        self.save_dir = self.args.save_dir
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        self._load_checkpoint(args.load)
        self.test_dataloader = test_dataloader

    def test(self):
        """
        Generate predictions for competition submission without ground truth
        """
        self.logger.info('Start to generate predictions for competition submission.')
        
        self.model.eval()
        with torch.no_grad():
            test_predictions = []
            test_ids = []
            
            for batch_idx, batch_data in tqdm(enumerate(self.test_dataloader)):
                # Handle different dataloader formats
                if len(batch_data) == 2:  # Only images_id and images (competition format)
                    images_id, images = batch_data
                    images_id = images_id[0] if isinstance(images_id, (list, tuple)) else images_id
                    images = images.to(self.device)
                elif len(batch_data) == 4:  # Full format with ground truth (fallback)
                    images_id, images, _, _ = batch_data
                    images_id = images_id[0] if isinstance(images_id, (list, tuple)) else images_id
                    images = images.to(self.device)
                else:
                    raise ValueError(f"Unexpected batch format with {len(batch_data)} elements")
                
                # Generate predictions
                output = self.model(images, mode='sample')
                #Debugging print statements
                '''token_ids = output.cpu().numpy()[0]  # Assuming batch size = 1
                print(f"[{batch_idx}] Token IDs: {token_ids}")

                decoded = self.model.tokenizer.decode(token_ids)
                print(f"[{batch_idx}] Decoded: {decoded}")'''
                #Debugging print statements ended on above line
                reports = self.model.tokenizer.decode_batch(output.cpu().numpy())
                
                test_predictions.extend(reports)
                test_ids.extend([images_id] if not isinstance(images_id, list) else images_id)
            
            # Create DataFrame for submission
            submission_df = pd.DataFrame({
                'case_id': test_ids,
                'generated_report': test_predictions
            })
            
            # Save predictions for submission
            submission_path = os.path.join(self.save_dir, "competition_predictions.csv")
            submission_df.to_csv(submission_path, index=False)
            
            # Also save just the predictions in the original format for compatibility
            predictions_df = pd.DataFrame(test_predictions, columns=['Generated Reports'])
            predictions_df.to_csv(os.path.join(self.save_dir, "predictions_only.csv"), index=False)
            
            self.logger.info(f'Generated {len(test_predictions)} predictions')
            self.logger.info(f'Predictions saved to: {submission_path}')
            
            return {'num_predictions': len(test_predictions)}

    def plot(self):
        """
        Generate attention visualizations (optional for competition)
        """
        assert self.args.batch_size == 1 and self.args.beam_size == 1
        self.logger.info('Start to plot attention weights.')
        os.makedirs(os.path.join(self.save_dir, "attentions"), exist_ok=True)
        mean = torch.tensor((0.485, 0.456, 0.406))
        std = torch.tensor((0.229, 0.224, 0.225))
        mean = mean[:, None, None]
        std = std[:, None, None]

        self.model.eval()
        with torch.no_grad():
            for batch_idx, batch_data in tqdm(enumerate(self.test_dataloader)):
                # Handle different batch formats
                if len(batch_data) == 2:
                    images_id, images = batch_data
                    images = images.to(self.device)
                elif len(batch_data) == 4:
                    images_id, images, _, _ = batch_data
                    images = images.to(self.device)
                
                output = self.model(images, mode='sample')
                image = torch.clamp((images[0].cpu() * std + mean) * 255, 0, 255).int().cpu().numpy()
                report = self.model.tokenizer.decode_batch(output.cpu().numpy())[0].split()
                attention_weights = [layer.src_attn.attn.cpu().numpy()[:, :, :-1].mean(0).mean(0) for layer in
                                     self.model.encoder_decoder.model.decoder.layers]
                for layer_idx, attns in enumerate(attention_weights):
                    assert len(attns) == len(report)
                    for word_idx, (attn, word) in enumerate(zip(attns, report)):
                        os.makedirs(os.path.join(self.save_dir, "attentions", "{:04d}".format(batch_idx),
                                                 "layer_{}".format(layer_idx)), exist_ok=True)

                        heatmap = generate_heatmap(image, attn)
                        cv2.imwrite(os.path.join(self.save_dir, "attentions", "{:04d}".format(batch_idx),
                                                 "layer_{}".format(layer_idx), "{:04d}_{}.png".format(word_idx, word)),
                                    heatmap)
