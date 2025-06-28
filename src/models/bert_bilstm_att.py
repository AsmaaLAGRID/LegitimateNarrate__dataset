
import torch.nn as nn
from transformers import AutoModel

class SequentialSentenceClassifier(nn.Module):
    """
    Projets (batch) 
      => [tokenizer]
      => BERT ([CLS] de chaque phrase)
      => Projection linéaire + ReLU
      => BiLSTM (dépendances séquentielles)
      => MultiHeadAttention (pondération contextuelle)
      => Classifieur (label par phrase)
    """
    
    def __init__(self, 
                 model_name="bert-large-uncased",  
                 num_classes=2,  
                 lstm_hidden_size=256,
                 lstm_layers=2,
                 dropout_rate=0.3,
                 max_seq_length=200,  # Based on dataset statistiques (max=177, avec marge)
                 max_sentences=200):  # Based on dataset statistiques (max=168, avec marge)
        
        super(SequentialSentenceClassifier, self).__init__()
        
        self.max_seq_length = max_seq_length # max token per sentence
        self.max_sentences = max_sentences   # max sentence per project
        self.num_classes = num_classes       # number of classes
        
        # 1. Sentence encoder (BERT)
        self.sentence_encoder = AutoModel.from_pretrained(model_name)
        self.sentence_embedding_dim = self.sentence_encoder.config.hidden_size
        
        # Gel partiel des couches BERT pour éviter l'overfitting
        for param in self.sentence_encoder.embeddings.parameters():
            param.requires_grad = False
        for layer in self.sentence_encoder.encoder.layer[:6]:  # Gel des 6 premières couches
            for param in layer.parameters():
                param.requires_grad = False
        # 2. Dimentionnality reduction post-BERT
        self.sentence_projector = nn.Sequential(
            nn.Linear(self.sentence_embedding_dim, lstm_hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # 3. LSTM bidirectionnel to get the sequentiality between sentences
        self.lstm = nn.LSTM(
            input_size=lstm_hidden_size,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout_rate if lstm_layers > 1 else 0
        )
        
        # 4. Attention mechanism to weight the important sentences
        self.attention = nn.MultiheadAttention(
            embed_dim=lstm_hidden_size * 2,  # *2 car LSTM bidirectionnel
            num_heads=8,
            dropout=dropout_rate,
            batch_first=True
        )
        
        # 5. Classification layer
        self.classifier = nn.Sequential(
            nn.Linear(lstm_hidden_size * 2, lstm_hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(lstm_hidden_size, lstm_hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(lstm_hidden_size // 2, num_classes)
        )

    
    def encode_sentences(self, input_ids, attention_mask):
        """
        Encode les phrases individuellement avec BERT
        """
        batch_size, num_sentences, seq_len = input_ids.shape
        
        # Reshape pour traiter toutes les phrases d'un batch ensemble
        input_ids = input_ids.view(-1, seq_len)
        attention_mask = attention_mask.view(-1, seq_len)
        
        # Encodage BERT
        outputs = self.sentence_encoder(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
        # Utilisation du token [CLS] comme représentation de phrase
        sentence_embeddings = outputs.last_hidden_state[:, 0, :]  # [CLS] token
        
        # Reshape pour retrouver la structure batch x sentences x embedding
        sentence_embeddings = sentence_embeddings.view(batch_size, num_sentences, -1)
        
        return sentence_embeddings
    
    def forward(self, input_ids, attention_mask, sentence_masks=None):
        """
        Forward pass complet
        
        Args:
            input_ids: [batch_size, max_sentences, max_seq_length]
            attention_mask: [batch_size, max_sentences, max_seq_length]
            sentence_masks: [batch_size, max_sentences] - masque pour phrases valides
        """
        batch_size, num_sentences, seq_len = input_ids.shape
        
        # 1. Encodage des phrases
        sentence_embeddings = self.encode_sentences(input_ids, attention_mask)
        
        # 2. Projection dimensionnelle
        sentence_embeddings = self.sentence_projector(sentence_embeddings)
        
        # 3. LSTM pour dépendances séquentielles
        '''if sentence_masks is not None:
            # Calcul des longueurs réelles pour pack_padded_sequence
            lengths = sentence_masks.sum(dim=1).cpu()
            sentence_embeddings = nn.utils.rnn.pack_padded_sequence(
                sentence_embeddings, lengths, batch_first=True, enforce_sorted=False
            )
            lstm_out, (hidden, cell) = self.lstm(sentence_embeddings)
            lstm_out, _ = nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)
        else:'''
        lstm_out, (hidden, cell) = self.lstm(sentence_embeddings)
        
        # 4. Mécanisme d'attention
        attended_out, attention_weights = self.attention(lstm_out, lstm_out, lstm_out)
        
        # 5. Classification pour chaque phrase
        logits = self.classifier(attended_out)
        
        return logits, attention_weights