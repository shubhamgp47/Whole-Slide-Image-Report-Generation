import numpy as np
import torch
import torch.nn as nn
from modules.visual_extractor import VisualExtractor
from modules.histgen_module import BaseTitanHistGen

class HistGenTitanModel(nn.Module):
    def __init__(self, args, tokenizer):
        super(HistGenTitanModel, self).__init__()
        self.args = args
        self.tokenizer = tokenizer
        
        # Create simplified encoder-decoder without CMC module
        self.encoder_decoder = BaseTitanHistGen(args, tokenizer)  # New simplified version
        
        # TITAN embedding projection layer
        # This was for baseline training 1. Removing this for 2nd training (hypothesis is The 768→512 compression is losing critical information)
        #self.titan_projection = torch.nn.Linear(args.titan_embedding_dim, args.projection_dim)

        #Added this for second training. This is used in training 2,3,4
        #self.titan_projection = nn.Identity()  # No transformation

        #Now, for training 5
        self.titan_expansion = nn.Linear(args.titan_embedding_dim, args.projection_dim)  # 768->1536 expansion
        self.titan_compression = nn.Linear(args.projection_dim, args.d_model)  # 1536->d_model compression, e.g. 768

        
        # Optional: Additional projection to match decoder expected dimensions
        # Commented this too for second training(maybe we meant 5th?). Used in training 2,3,4
        '''if args.projection_dim != args.d_model:
            self.decoder_projection = torch.nn.Linear(args.projection_dim, args.d_model)
        else:
            self.decoder_projection = None'''
            
        # Remove decoder projection too
        self.decoder_projection = None
            
        self.forward = self.forward_titan_pathology
        
        # No visual extractor needed for TITAN embeddings
        # self.visual_extractor = None

    def __str__(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + '\nTrainable parameters: {}'.format(params)

    #For Training 1,2,3,4
    '''def forward_titan_pathology(self, slide_embeddings, targets=None, mode='train', update_opts={}):
        """
        Forward pass for TITAN-based HistGen
        Args:
            slide_embeddings: Pre-computed TITAN slide embeddings [batch_size, titan_embedding_dim]
            targets: Target report tokens for training
            mode: 'train' or 'sample'
        """
        
        # Project TITAN embeddings to decoder input dimension
        projected_features = self.titan_projection(slide_embeddings)  # [batch_size, projection_dim]
        
        # Optional additional projection to decoder dimension
        # Commented for training 2,3,4
        #if self.decoder_projection is not None:
            #projected_features = self.decoder_projection(projected_features)  # [batch_size, d_model]
        
        # For decoder, we need both fc_feats (global) and att_feats (attention features)
        # Since TITAN gives us slide-level embeddings, we use them for both
        fc_feats = projected_features  # Global slide features [batch_size, d_model]
        # Commented for training 4, Hypothesis: Single slide embedding lacks spatial diversity that decoder expects.
        # 
        #att_feats = projected_features.unsqueeze(1)  # Add sequence dimension [batch_size, 1, d_model]
        att_feats = slide_embeddings.unsqueeze(1).repeat(1, 8, 1)  # [batch_size, 8, 768]
        
        if mode == 'train':
            output = self.encoder_decoder(fc_feats, att_feats, targets, mode='forward')
            return output
        elif mode == 'sample':
            output, output_probs = self.encoder_decoder(fc_feats, att_feats, mode='sample')
            return output
        else:
            raise ValueError(f"Invalid mode: {mode}")'''
    
    #For Training 5    
    def forward_titan_pathology(self, slide_embeddings, targets=None, mode='train', update_opts={}):
        x = self.titan_expansion(slide_embeddings)  # [batch_size, 1536]
        x = torch.relu(x)  # non-linearity
        projected_features = self.titan_compression(x)  # [batch_size, d_model], e.g. 768
        
        fc_feats = projected_features  # [batch_size, d_model]
        
        # Use expanded features for att_feats - add sequence dimension
        att_feats = x.unsqueeze(1).repeat(1, 8, 1)  # [batch_size, 8, 1536]
        
        # Optionally compress att_feats to d_model if attention layers expect it (check your att_embed input dim)
        # att_feats = self.titan_compression(att_feats.view(-1, att_feats.size(-1))).view(att_feats.size(0), att_feats.size(1), -1)
        
        if mode == 'train':
            output = self.encoder_decoder(fc_feats, att_feats, targets, mode='forward')
            return output
        elif mode == 'sample':
            output, output_probs = self.encoder_decoder(fc_feats, att_feats, mode='sample')
            return output
        else:
            raise ValueError(f"Invalid mode: {mode}")
